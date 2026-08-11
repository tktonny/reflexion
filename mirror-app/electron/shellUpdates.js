// Over-the-air updates for the Linux (Ubuntu) mirror — the Electron SHELL itself.
//
// bundleUpdates.js replaces the renderer bundle (a few MB) and covers almost every change: prompts,
// conversation flow, self-check logic, screens, anything inlined from EXPO_PUBLIC_*. It cannot replace the
// shell — Chromium, the main process, network.js, the relay. That is what this is for: the occasional
// 97 MB upgrade, run rarely and deliberately.
//
// WHY NOT electron-updater: it is the standard answer, but it drags a transitive dependency tree into a
// package whose `files` allowlist deliberately excludes node_modules (only `ws` is re-included), and it wants
// its own feed format (latest-linux.yml) alongside the manifest we already serve. On Linux it does not sign
// anything either, so it would buy us the same trust model with more moving parts. An AppImage is a single
// self-contained executable file, which makes replacing it a file copy — so we do that, reusing the exact
// manifest + sha256 verification that bundleUpdates.js already uses.
//
// TRUST: identical to bundleUpdates — HTTPS plus a sha256 from the manifest. That protects against
// corruption and truncation, not against a compromised update host. Because this channel replaces an
// EXECUTABLE rather than a bundle of JS, that distinction matters more here: only ever point
// REFLEXION_UPDATE_BASE at a host you control as tightly as the backend itself.

const crypto = require('crypto')
const fs = require('fs')
const fsp = require('fs/promises')
const path = require('path')

const FETCH_TIMEOUT_MS = 60_000
const MAX_APPIMAGE_BYTES = 400 * 1024 * 1024

function manifestUrl(updateBase) {
  return `${String(updateBase).replace(/\/+$/, '')}/shell-latest.json`
}

/**
 * The AppImage file currently running, or null when this is not an AppImage (dev run, .deb install).
 * AppImageLauncher and the AppImage runtime both set APPIMAGE to the absolute path of the file.
 */
function runningAppImage(env = process.env) {
  const value = env.APPIMAGE
  if (typeof value !== 'string' || !value.trim()) return null
  return value.trim()
}

/** Whether this install can update its own shell at all — reported to the settings screen. */
function shellUpdateSupported(env = process.env) {
  return Boolean(runningAppImage(env))
}

// A shell version is only ever compared and displayed, never used as a path, but keep it tight anyway.
const VERSION_SHAPE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/

function parseManifest(manifest, updateBase) {
  if (!manifest || typeof manifest !== 'object') throw new Error('shell manifest is not an object')
  const { version, sha256, url, arch } = manifest
  if (typeof version !== 'string' || !VERSION_SHAPE.test(version)) throw new Error('shell manifest has no usable version')
  if (typeof sha256 !== 'string' || !/^[0-9a-f]{64}$/i.test(sha256)) throw new Error('shell manifest has no sha256')
  if (typeof url !== 'string' || !url.trim()) throw new Error('shell manifest has no url')
  const resolved = new URL(url, `${String(updateBase).replace(/\/+$/, '')}/`)
  const base = new URL(`${String(updateBase).replace(/\/+$/, '')}/`)
  if (resolved.origin !== base.origin) throw new Error('shell archive is not on the update origin')
  if (resolved.protocol !== 'https:' && resolved.hostname !== '127.0.0.1' && resolved.hostname !== 'localhost') {
    throw new Error('shell archive must be served over https')
  }
  return { version, sha256: sha256.toLowerCase(), url: resolved.toString(), arch: typeof arch === 'string' ? arch : null }
}

/**
 * Download the new AppImage next to the running one and verify it. Does NOT install — `applyDownloaded`
 * does, and only when an operator says so: installing relaunches the app.
 *
 * Downloading beside the target (not into /tmp) so the rename that installs it is on the same filesystem —
 * a cross-device rename would fail, and copying an executable the app is running from is worse.
 */
async function checkAndDownload({ updateBase, currentVersion, arch = process.arch, env = process.env }) {
  if (!updateBase) return { kind: 'disabled', detail: 'No update host is configured for this unit.' }
  const target = runningAppImage(env)
  if (!target) {
    return { kind: 'disabled', detail: 'Shell updates need an AppImage install (this unit runs from a package or a dev build).' }
  }
  try {
    const response = await fetch(manifestUrl(updateBase), { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS), redirect: 'follow' })
    if (!response.ok) throw new Error(`shell manifest returned HTTP ${response.status}`)
    const manifest = parseManifest(await response.json(), updateBase)
    // An x64 AppImage on an arm64 unit would install cleanly and then refuse to execute, leaving a dead
    // appliance, so a mismatch is refused rather than attempted.
    if (manifest.arch && manifest.arch !== arch) {
      return { kind: 'failed', detail: `The published shell is for ${manifest.arch}; this unit is ${arch}.` }
    }
    if (manifest.version === currentVersion) {
      return { kind: 'up_to_date', detail: `Already on shell ${manifest.version}.`, version: manifest.version }
    }

    const staged = `${target}.new`
    const archive = await fetch(manifest.url, { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS), redirect: 'follow' })
    if (!archive.ok) throw new Error(`shell archive returned HTTP ${archive.status}`)
    const buffer = Buffer.from(await archive.arrayBuffer())
    if (buffer.byteLength === 0) throw new Error('shell archive is empty')
    if (buffer.byteLength > MAX_APPIMAGE_BYTES) throw new Error('shell archive is larger than the allowed size')
    const digest = crypto.createHash('sha256').update(buffer).digest('hex')
    if (digest !== manifest.sha256) throw new Error('the downloaded shell did not match its checksum')

    await fsp.writeFile(staged, buffer, { mode: 0o755 })
    return {
      kind: 'downloaded',
      detail: `Shell ${manifest.version} is ready. Applying it will restart the mirror.`,
      version: manifest.version,
      staged,
    }
  } catch (error) {
    // A failed shell check must never take the mirror down; it keeps running what it has.
    const message = error instanceof Error ? error.message : String(error)
    return { kind: 'failed', detail: message.slice(0, 200) }
  }
}

/**
 * Install a staged AppImage: keep the current one as `.old`, move the new one into place, and report that a
 * relaunch is needed. The caller relaunches — ONLY when the mirror is idle.
 *
 * The old file is kept rather than deleted so a unit that will not start can be recovered by hand, which on
 * an appliance in someone's home is the difference between a phone call and a site visit.
 */
async function applyDownloaded({ env = process.env } = {}) {
  const target = runningAppImage(env)
  if (!target) return { ok: false, error: 'Shell updates need an AppImage install.' }
  const staged = `${target}.new`
  if (!fs.existsSync(staged)) return { ok: false, error: 'There is no downloaded shell to apply.' }
  try {
    const backup = `${target}.old`
    await fsp.rm(backup, { force: true })
    // Replacing a running AppImage is safe: the kernel keeps the open inode alive until the process exits.
    await fsp.rename(target, backup)
    await fsp.rename(staged, target)
    await fsp.chmod(target, 0o755)
    return { ok: true, restartRequired: true, backup }
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : 'Could not install the shell update.' }
  }
}

/** Drop a staged download without installing it. */
async function discardDownloaded({ env = process.env } = {}) {
  const target = runningAppImage(env)
  if (!target) return { ok: true }
  await fsp.rm(`${target}.new`, { force: true }).catch(() => undefined)
  return { ok: true }
}

module.exports = {
  MAX_APPIMAGE_BYTES,
  applyDownloaded,
  checkAndDownload,
  discardDownloaded,
  manifestUrl,
  parseManifest,
  runningAppImage,
  shellUpdateSupported,
}
