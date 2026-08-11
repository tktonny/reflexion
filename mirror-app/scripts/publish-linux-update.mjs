#!/usr/bin/env node
// Produce the files an Ubuntu mirror fleet downloads to update itself.
//
// Two independent channels — see electron/bundleUpdates.js and electron/shellUpdates.js:
//
//   latest.json        + bundle-<version>.zip     the renderer bundle (a few MB, the routine channel)
//   shell-latest.json  + shell-<version>.AppImage the Electron shell  (~97 MB, the rare channel)
//
// Both manifests carry a sha256 that the appliance verifies before installing anything, so this script is
// the only place the digest is computed and it must be computed from the exact bytes that get uploaded.
//
// Usage:
//   node scripts/publish-linux-update.mjs                             # bundle only, version from today
//   node scripts/publish-linux-update.mjs --version=2026.08.11-2
//   node scripts/publish-linux-update.mjs --shell="dist-linux/Reflexion Mirror-1.0.0.AppImage" --arch=x64
//
// Output lands in dist-updates/, which is what you rsync to /www/wwwroot/mirror-updates/ on the server.

import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import process from 'node:process'

const ROOT = path.resolve(import.meta.dirname, '..')
const DIST = path.join(ROOT, 'dist')
const OUT = path.join(ROOT, 'dist-updates')

function argument(name) {
  const prefix = `--${name}=`
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length)
}

function fail(message) {
  console.error(`\n✗ ${message}\n`)
  process.exit(1)
}

// Date-based by default: the appliance only ever compares versions for equality, and a date tells an
// operator reading latest.json when the bundle was cut, which a hash would not.
function defaultVersion() {
  const now = new Date()
  const stamp = [now.getFullYear(), now.getMonth() + 1, now.getDate()]
    .map((part, index) => (index === 0 ? String(part) : String(part).padStart(2, '0')))
    .join('.')
  return `${stamp}-${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}`
}

// Must match VERSION_SHAPE in electron/bundleUpdates.js — the version becomes a directory name on the
// appliance, so a value this script accepts but the shell rejects would publish an update nothing installs.
const VERSION_SHAPE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/

const version = (argument('version') || defaultVersion()).trim()
if (!VERSION_SHAPE.test(version)) fail(`--version=${version} is not a usable version (letters, digits, . _ - only).`)

const sha256 = (file) => createHash('sha256').update(fs.readFileSync(file)).digest('hex')

fs.mkdirSync(OUT, { recursive: true })

// --- renderer bundle ---------------------------------------------------------------------------

const shellPath = argument('shell')
if (!argument('shell-only')) {
  if (!fs.existsSync(path.join(DIST, 'index.html'))) {
    fail('dist/index.html is missing — run `npm run electron:export` first.')
  }

  // Stamped INTO the bundle so a unit running the packaged copy knows its own version and does not
  // re-download an identical bundle on every check.
  fs.writeFileSync(path.join(DIST, 'bundle-version.txt'), `${version}\n`)

  const archiveName = `bundle-${version}.zip`
  const archive = path.join(OUT, archiveName)
  fs.rmSync(archive, { force: true })
  // Zipped from INSIDE dist/ so index.html sits at the archive root: the appliance unzips straight into the
  // bundle directory and serves it, with no wrapper directory to strip.
  execFileSync('zip', ['-qr', archive, '.'], { cwd: DIST, stdio: 'inherit' })

  const digest = sha256(archive)
  const manifest = { version, sha256: digest, url: archiveName, publishedAt: new Date().toISOString() }
  fs.writeFileSync(path.join(OUT, 'latest.json'), `${JSON.stringify(manifest, null, 2)}\n`)

  const mb = (fs.statSync(archive).size / 1024 / 1024).toFixed(1)
  console.log(`✓ bundle  ${archiveName}  ${mb} MB  sha256 ${digest.slice(0, 16)}…`)
}

// --- Electron shell ---------------------------------------------------------------------------

if (shellPath) {
  if (!fs.existsSync(shellPath)) fail(`--shell=${shellPath} does not exist.`)
  // The appliance refuses a manifest whose arch does not match its own, because installing the wrong
  // architecture leaves a unit that cannot execute at all.
  const arch = argument('arch') || (/-arm64\.AppImage$/i.test(shellPath) ? 'arm64' : 'x64')
  const shellVersion = (argument('shell-version') || JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8')).version).trim()
  if (!VERSION_SHAPE.test(shellVersion)) fail(`--shell-version=${shellVersion} is not a usable version.`)

  const assetName = `shell-${shellVersion}-${arch}.AppImage`
  fs.copyFileSync(shellPath, path.join(OUT, assetName))
  const digest = sha256(path.join(OUT, assetName))
  const manifest = { version: shellVersion, arch, sha256: digest, url: assetName, publishedAt: new Date().toISOString() }
  // Per-arch manifest name: one fleet can hold both x64 and arm64 units, and each must see only its own.
  fs.writeFileSync(path.join(OUT, `shell-latest-${arch}.json`), `${JSON.stringify(manifest, null, 2)}\n`)
  // The appliance asks for shell-latest.json; nginx maps that per-arch (see the doc). A copy is written
  // under the plain name too so a single-architecture fleet needs no nginx rule at all.
  fs.writeFileSync(path.join(OUT, 'shell-latest.json'), `${JSON.stringify(manifest, null, 2)}\n`)

  const mb = (fs.statSync(path.join(OUT, assetName)).size / 1024 / 1024).toFixed(1)
  console.log(`✓ shell   ${assetName}  ${mb} MB  sha256 ${digest.slice(0, 16)}…  arch ${arch}`)
}

console.log(`\nPublish with:\n  rsync -av ${path.relative(ROOT, OUT)}/ root@<server>:/www/wwwroot/mirror-updates/\n`)
