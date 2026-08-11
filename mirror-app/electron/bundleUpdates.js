// Over-the-air updates for the Linux (Ubuntu) mirror — the renderer bundle.
//
// WHY THIS EXISTS AT ALL: the Android app updates over the air with expo-updates (EAS Update). expo-updates
// has NO web implementation, so inside Electron `Updates.isEnabled` is false and the settings screen simply
// reports "OTA disabled". The Linux fleet therefore had no update path whatsoever — every fix meant
// shipping a 97 MB AppImage by hand. This module is the Linux equivalent of expo-updates, deliberately
// mirroring its semantics so the one settings screen can drive both platforms:
//
//   * only the RENDERER BUNDLE is updatable (dist/ — JS, assets, and every EXPO_PUBLIC_* value inlined into
//     it). The Electron shell, Chromium, and the relay are NOT: those need electron-updater.
//   * checking is MANUAL. The mirror is a kiosk that stays powered for days, so "check on launch" would
//     almost never fire; the opposite — polling and reloading whenever something lands — risks reloading
//     mid-conversation and cutting an elder off in the middle of a check-in. An operator triggers it when
//     they can see the mirror is idle.
//   * downloading and APPLYING are separate steps, for the same reason.
//   * a bundle that does not boot is rolled back automatically, because a bad update on an appliance in
//     someone's home is otherwise a support visit.
//
// LAYOUT under <userData>/:
//   bundles/<version>/          unpacked bundle (index.html + _expo/…)
//   bundles/state.json          { active, previous, pending, lastFailed }
//
// TRUST: the manifest is fetched over HTTPS and every archive is checked against the sha256 in that
// manifest before it is unpacked. That makes the update no more trusted than the host serving it — which is
// the same origin the app already trusts for its API — and stops a corrupted or truncated download from
// bricking the unit. It is NOT a signature: a compromised update host could serve a bundle that this code
// would accept. Signing the manifest with a key pinned in the shell is the next step if the update host
// ever stops being the same trusted origin as the backend.

const crypto = require('crypto')
const fs = require('fs')
const fsp = require('fs/promises')
const path = require('path')
const { execFile } = require('child_process')

const { systemEnv } = require('./systemEnv')

const STATE_FILE = 'state.json'
const BUNDLES_DIR = 'bundles'
// Enough for a JS bundle with fonts and the wake-word asset; small enough that a wrong URL serving
// something enormous cannot fill the appliance's disk.
const MAX_ARCHIVE_BYTES = 160 * 1024 * 1024
const FETCH_TIMEOUT_MS = 30_000

function bundlesRoot(userDataDir) {
  return path.join(userDataDir, BUNDLES_DIR)
}

function statePath(userDataDir) {
  return path.join(bundlesRoot(userDataDir), STATE_FILE)
}

function bundleDir(userDataDir, version) {
  return path.join(bundlesRoot(userDataDir), version)
}

/** Never throws: a corrupt state file must fall back to "running the packaged bundle", not brick the unit. */
function readState(userDataDir) {
  try {
    const parsed = JSON.parse(fs.readFileSync(statePath(userDataDir), 'utf8'))
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch {
    return {}
  }
}

async function writeState(userDataDir, next) {
  await fsp.mkdir(bundlesRoot(userDataDir), { recursive: true })
  await fsp.writeFile(statePath(userDataDir), JSON.stringify(next, null, 2))
}

// A version string becomes a directory name, so it must not be able to escape the bundles root or collide
// with the state file. Rejecting anything outside this alphabet is simpler than sanitising.
const VERSION_SHAPE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/

function validVersion(version) {
  return typeof version === 'string' && VERSION_SHAPE.test(version) && version !== STATE_FILE && !version.includes('..')
}

/**
 * The directory the SPA should be served from: the active OTA bundle when one is installed and intact,
 * otherwise the bundle packaged inside the AppImage.
 *
 * Falls back on a missing directory or a missing index.html rather than trusting state.json, because the
 * consequence of getting this wrong is a blank mirror — and a half-deleted bundle directory is exactly the
 * state an interrupted update leaves behind.
 */
function activeWebDir(userDataDir, packagedDir) {
  const { active } = readState(userDataDir)
  if (!validVersion(active)) return packagedDir
  const dir = bundleDir(userDataDir, active)
  try {
    if (fs.existsSync(path.join(dir, 'index.html'))) return dir
  } catch {
    /* fall through */
  }
  return packagedDir
}

/** Which bundle is running, for the settings screen. */
function currentBundle(userDataDir) {
  const state = readState(userDataDir)
  const active = validVersion(state.active) ? state.active : null
  return {
    version: active,
    embedded: active === null,
    pending: validVersion(state.pending) ? state.pending : null,
    lastFailed: typeof state.lastFailed === 'string' ? state.lastFailed : null,
  }
}

function manifestUrl(updateBase) {
  return `${String(updateBase).replace(/\/+$/, '')}/latest.json`
}

async function fetchJson(url) {
  const response = await fetch(url, { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS), redirect: 'follow' })
  if (!response.ok) throw new Error(`update manifest returned HTTP ${response.status}`)
  return response.json()
}

/**
 * Validate a manifest before acting on it. A malformed manifest must be a clean "cannot update" rather than
 * a partially applied update, so every field the download path relies on is checked here, once.
 */
function parseManifest(manifest, updateBase) {
  if (!manifest || typeof manifest !== 'object') throw new Error('update manifest is not an object')
  const { version, sha256, url } = manifest
  if (!validVersion(version)) throw new Error('update manifest has no usable version')
  if (typeof sha256 !== 'string' || !/^[0-9a-f]{64}$/i.test(sha256)) throw new Error('update manifest has no sha256')
  if (typeof url !== 'string' || !url.trim()) throw new Error('update manifest has no url')
  // A relative url keeps the manifest portable across hosts; an absolute one must stay on the update origin
  // so a manifest cannot redirect the appliance at an arbitrary download host.
  const resolved = new URL(url, `${String(updateBase).replace(/\/+$/, '')}/`)
  const base = new URL(`${String(updateBase).replace(/\/+$/, '')}/`)
  if (resolved.origin !== base.origin) throw new Error('update archive is not on the update origin')
  if (resolved.protocol !== 'https:' && resolved.hostname !== '127.0.0.1' && resolved.hostname !== 'localhost') {
    throw new Error('update archive must be served over https')
  }
  return { version, sha256: sha256.toLowerCase(), url: resolved.toString() }
}

async function downloadArchive(url, destination) {
  const response = await fetch(url, { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS), redirect: 'follow' })
  if (!response.ok) throw new Error(`update archive returned HTTP ${response.status}`)
  const buffer = Buffer.from(await response.arrayBuffer())
  // Checked after buffering rather than streaming: the cap exists to stop a wrong URL from filling the
  // disk, and a bundle this size fits in memory on any unit that can run Chromium.
  if (buffer.byteLength > MAX_ARCHIVE_BYTES) throw new Error('update archive is larger than the allowed size')
  if (buffer.byteLength === 0) throw new Error('update archive is empty')
  await fsp.writeFile(destination, buffer)
  return crypto.createHash('sha256').update(buffer).digest('hex')
}

/** Unpack with the system unzip. Linux appliances have it, and shelling out beats vendoring a zip library.
 *  `systemEnv()` matters here for the same reason it does in network.js: inside an AppImage the injected
 *  LD_LIBRARY_PATH would make the system unzip load the bundle's libraries and fail. */
function unzip(archive, destination) {
  return new Promise((resolve, reject) => {
    execFile('unzip', ['-q', '-o', archive, '-d', destination], { env: systemEnv(), timeout: 120_000 }, (error, _stdout, stderr) => {
      if (error) reject(new Error(`could not unpack the update (${stderr || error.message})`.trim()))
      else resolve()
    })
  })
}

/**
 * Check for an update and install it alongside the running one. Does NOT switch to it — `applyPending`
 * does, so an operator can confirm the mirror is idle before the window reloads.
 *
 * Returns a shaped outcome instead of throwing: a failed update check must never take the mirror down, it
 * keeps running the bundle it has.
 */
async function checkAndDownload({ userDataDir, updateBase, packagedVersion }) {
  if (!updateBase) return { kind: 'disabled', detail: 'No update host is configured for this unit.' }
  let archive
  try {
    const manifest = parseManifest(await fetchJson(manifestUrl(updateBase)), updateBase)
    const current = currentBundle(userDataDir)
    if (manifest.version === current.version || (current.embedded && manifest.version === packagedVersion)) {
      return { kind: 'up_to_date', detail: `Already on bundle ${manifest.version}.`, version: manifest.version }
    }
    // Refuse to reinstall a bundle that already failed to boot here — otherwise an operator pressing the
    // button repeatedly would loop through the same broken update.
    if (manifest.version === current.lastFailed) {
      return { kind: 'failed', detail: `Bundle ${manifest.version} failed to start on this unit before; it will not be reinstalled.` }
    }

    await fsp.mkdir(bundlesRoot(userDataDir), { recursive: true })
    archive = path.join(bundlesRoot(userDataDir), `${manifest.version}.zip`)
    const digest = await downloadArchive(manifest.url, archive)
    if (digest !== manifest.sha256) {
      // Truncated, corrupted, or not the file the manifest describes. Either way it must not be unpacked.
      throw new Error('the downloaded update did not match its checksum')
    }

    const target = bundleDir(userDataDir, manifest.version)
    await fsp.rm(target, { recursive: true, force: true })
    await fsp.mkdir(target, { recursive: true })
    await unzip(archive, target)
    // A bundle with no index.html cannot be served; catching it here means the unit never switches to it.
    if (!fs.existsSync(path.join(target, 'index.html'))) {
      await fsp.rm(target, { recursive: true, force: true })
      throw new Error('the update does not contain an index.html')
    }

    const state = readState(userDataDir)
    await writeState(userDataDir, { ...state, pending: manifest.version })
    return { kind: 'downloaded', detail: `Bundle ${manifest.version} is ready. Restart to apply it.`, version: manifest.version }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return { kind: 'failed', detail: message.slice(0, 200) }
  } finally {
    if (archive) await fsp.rm(archive, { force: true }).catch(() => undefined)
  }
}

/**
 * Promote the downloaded bundle to active. The caller reloads the window afterwards — ONLY when the mirror
 * is idle, because this restarts the UI and would otherwise cut a conversation off mid-sentence.
 *
 * `previous` is kept so `rollback` has somewhere to go.
 */
async function applyPending(userDataDir) {
  const state = readState(userDataDir)
  if (!validVersion(state.pending)) return { ok: false, error: 'There is no downloaded update to apply.' }
  if (!fs.existsSync(path.join(bundleDir(userDataDir, state.pending), 'index.html'))) {
    await writeState(userDataDir, { ...state, pending: null })
    return { ok: false, error: 'The downloaded update is no longer intact; download it again.' }
  }
  await writeState(userDataDir, {
    ...state,
    previous: validVersion(state.active) ? state.active : null,
    active: state.pending,
    pending: null,
    // Cleared on a fresh apply so a version that failed once can be retried after it is fixed upstream.
    lastFailed: null,
    // The renderer clears this once it has booted; a value still here on next launch means it never did.
    booting: state.pending,
  })
  return { ok: true, version: state.pending }
}

/**
 * Called by the renderer once it has actually rendered. This is what makes rollback possible: without a
 * positive signal from inside the new bundle, the shell cannot tell "started fine" from "white screen".
 */
async function markBooted(userDataDir) {
  const state = readState(userDataDir)
  if (!state.booting) return { ok: true }
  await writeState(userDataDir, { ...state, booting: null })
  return { ok: true }
}

/**
 * Go back to the previously active bundle (or the packaged one). Called on startup when the last apply
 * never reported a successful boot, and available manually from the settings screen.
 */
async function rollback(userDataDir, reason = 'requested') {
  const state = readState(userDataDir)
  const failed = validVersion(state.booting) ? state.booting : (validVersion(state.active) ? state.active : null)
  const target = validVersion(state.previous) ? state.previous : null
  await writeState(userDataDir, {
    ...state,
    active: target,
    previous: null,
    booting: null,
    // Remembered so checkAndDownload will not immediately reinstall the bundle that just failed.
    lastFailed: failed,
  })
  if (failed) console.warn(`[electron] rolled back bundle ${failed} (${reason}); now on ${target ?? 'the packaged bundle'}`)
  return { ok: true, from: failed, to: target }
}

/**
 * Run at startup, BEFORE the window is created. If the previous launch applied a bundle that never reported
 * a successful boot, that bundle is broken — a crash loop on an appliance in someone's home is a support
 * visit, so undo it automatically.
 */
async function rollbackIfLastBootFailed(userDataDir) {
  const state = readState(userDataDir)
  if (!validVersion(state.booting)) return { rolledBack: false }
  const result = await rollback(userDataDir, 'previous launch never finished starting')
  return { rolledBack: true, ...result }
}

/** Delete bundles that are neither active, pending, nor the rollback target. */
async function pruneBundles(userDataDir) {
  const state = readState(userDataDir)
  const keep = new Set([state.active, state.pending, state.previous].filter(validVersion))
  let removed = 0
  try {
    for (const entry of await fsp.readdir(bundlesRoot(userDataDir), { withFileTypes: true })) {
      if (!entry.isDirectory() || keep.has(entry.name)) continue
      await fsp.rm(path.join(bundlesRoot(userDataDir), entry.name), { recursive: true, force: true })
      removed += 1
    }
  } catch {
    /* nothing installed yet */
  }
  return { removed }
}

module.exports = {
  MAX_ARCHIVE_BYTES,
  activeWebDir,
  applyPending,
  checkAndDownload,
  currentBundle,
  manifestUrl,
  markBooted,
  parseManifest,
  pruneBundles,
  readState,
  rollback,
  rollbackIfLastBootFailed,
  validVersion,
}
