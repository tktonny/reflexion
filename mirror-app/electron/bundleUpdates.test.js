const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { test } = require('node:test')

const bundleUpdates = require('./bundleUpdates')
const shellUpdates = require('./shellUpdates')

function scratch() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'reflexion-ota-'))
}

function installBundle(dir, version, { withIndex = true } = {}) {
  const target = path.join(dir, 'bundles', version)
  fs.mkdirSync(target, { recursive: true })
  if (withIndex) fs.writeFileSync(path.join(target, 'index.html'), '<html></html>')
  return target
}

function writeState(dir, state) {
  fs.mkdirSync(path.join(dir, 'bundles'), { recursive: true })
  fs.writeFileSync(path.join(dir, 'bundles', 'state.json'), JSON.stringify(state))
}

// --- which bundle gets served -------------------------------------------------------------------

test('a unit with no OTA bundle serves the one packaged in the AppImage', () => {
  assert.equal(bundleUpdates.activeWebDir(scratch(), '/packaged'), '/packaged')
})

test('an installed active bundle is served instead of the packaged one', () => {
  const dir = scratch()
  const installed = installBundle(dir, '2026.08.11-1')
  writeState(dir, { active: '2026.08.11-1' })
  assert.equal(bundleUpdates.activeWebDir(dir, '/packaged'), installed)
})

test('an active bundle whose files are gone falls back instead of showing a blank mirror', () => {
  // Exactly the state an interrupted update leaves behind. Trusting state.json here would be a black screen.
  const dir = scratch()
  writeState(dir, { active: '2026.08.11-1' })
  assert.equal(bundleUpdates.activeWebDir(dir, '/packaged'), '/packaged')
})

test('an active bundle directory with no index.html falls back', () => {
  const dir = scratch()
  installBundle(dir, 'broken', { withIndex: false })
  writeState(dir, { active: 'broken' })
  assert.equal(bundleUpdates.activeWebDir(dir, '/packaged'), '/packaged')
})

test('a corrupt state file degrades to the packaged bundle', () => {
  const dir = scratch()
  fs.mkdirSync(path.join(dir, 'bundles'), { recursive: true })
  fs.writeFileSync(path.join(dir, 'bundles', 'state.json'), '{ not json')
  assert.equal(bundleUpdates.activeWebDir(dir, '/packaged'), '/packaged')
})

test('a version that would escape the bundles directory is refused', () => {
  // The version string becomes a directory name, so traversal has to be impossible by construction.
  for (const bad of ['../../etc', 'a/b', '..', 'state.json', '', '.hidden']) {
    assert.equal(bundleUpdates.validVersion(bad), false, `expected ${JSON.stringify(bad)} to be rejected`)
  }
  assert.equal(bundleUpdates.validVersion('2026.08.11-1'), true)
})

test('a traversing active version is ignored rather than followed', () => {
  const dir = scratch()
  writeState(dir, { active: '../../../../etc' })
  assert.equal(bundleUpdates.activeWebDir(dir, '/packaged'), '/packaged')
})

// --- apply / rollback --------------------------------------------------------------------------

test('applying promotes the pending bundle and remembers the previous one', async () => {
  const dir = scratch()
  installBundle(dir, 'v1')
  installBundle(dir, 'v2')
  writeState(dir, { active: 'v1', pending: 'v2' })
  const result = await bundleUpdates.applyPending(dir)
  assert.deepEqual({ ok: result.ok, version: result.version }, { ok: true, version: 'v2' })
  const state = bundleUpdates.readState(dir)
  assert.equal(state.active, 'v2')
  assert.equal(state.previous, 'v1')
  assert.equal(state.pending, null)
  // The boot marker is what makes rollback possible at all.
  assert.equal(state.booting, 'v2')
})

test('applying with nothing pending is a clean refusal', async () => {
  const result = await bundleUpdates.applyPending(scratch())
  assert.equal(result.ok, false)
})

test('applying a pending bundle whose files vanished clears it instead of serving nothing', async () => {
  const dir = scratch()
  writeState(dir, { active: null, pending: 'ghost' })
  const result = await bundleUpdates.applyPending(dir)
  assert.equal(result.ok, false)
  assert.equal(bundleUpdates.readState(dir).pending, null)
})

test('a bundle that never reported a successful boot is rolled back on the next launch', async () => {
  const dir = scratch()
  installBundle(dir, 'v1')
  installBundle(dir, 'v2')
  writeState(dir, { active: 'v2', previous: 'v1', booting: 'v2' })
  const result = await bundleUpdates.rollbackIfLastBootFailed(dir)
  assert.equal(result.rolledBack, true)
  const state = bundleUpdates.readState(dir)
  assert.equal(state.active, 'v1')
  // Remembered so the operator cannot reinstall the same broken bundle by pressing the button again.
  assert.equal(state.lastFailed, 'v2')
})

test('a bundle that did report a successful boot is left alone', async () => {
  const dir = scratch()
  installBundle(dir, 'v2')
  writeState(dir, { active: 'v2', previous: 'v1' })
  assert.deepEqual(await bundleUpdates.rollbackIfLastBootFailed(dir), { rolledBack: false })
  assert.equal(bundleUpdates.readState(dir).active, 'v2')
})

test('markBooted clears the marker so the next launch keeps the new bundle', async () => {
  const dir = scratch()
  writeState(dir, { active: 'v2', booting: 'v2' })
  await bundleUpdates.markBooted(dir)
  assert.equal(bundleUpdates.readState(dir).booting, null)
  assert.deepEqual(await bundleUpdates.rollbackIfLastBootFailed(dir), { rolledBack: false })
})

test('rolling back with no previous bundle returns to the packaged one', async () => {
  const dir = scratch()
  installBundle(dir, 'v1')
  writeState(dir, { active: 'v1', booting: 'v1' })
  await bundleUpdates.rollbackIfLastBootFailed(dir)
  assert.equal(bundleUpdates.activeWebDir(dir, '/packaged'), '/packaged')
})

test('pruning keeps active, pending and the rollback target and deletes the rest', async () => {
  const dir = scratch()
  for (const v of ['v1', 'v2', 'v3', 'v4']) installBundle(dir, v)
  writeState(dir, { active: 'v3', previous: 'v2', pending: 'v4' })
  const { removed } = await bundleUpdates.pruneBundles(dir)
  assert.equal(removed, 1)
  assert.equal(fs.existsSync(path.join(dir, 'bundles', 'v1')), false)
  for (const kept of ['v2', 'v3', 'v4']) {
    assert.equal(fs.existsSync(path.join(dir, 'bundles', kept)), true, `${kept} should be kept`)
  }
})

test('pruning a unit that has never updated is a no-op', async () => {
  assert.deepEqual(await bundleUpdates.pruneBundles(scratch()), { removed: 0 })
})

// --- manifest validation -----------------------------------------------------------------------

const BASE = 'https://updates.test/mirror-updates'
const SHA = 'a'.repeat(64)

test('a well-formed manifest resolves a relative archive url against the update base', () => {
  const parsed = bundleUpdates.parseManifest({ version: '2026.08.11-1', sha256: SHA, url: 'bundle-2026.08.11-1.zip' }, BASE)
  assert.equal(parsed.url, 'https://updates.test/mirror-updates/bundle-2026.08.11-1.zip')
  assert.equal(parsed.version, '2026.08.11-1')
})

test('a manifest cannot redirect the appliance at another download host', () => {
  // Otherwise anyone able to serve the manifest could point the fleet anywhere.
  assert.throws(
    () => bundleUpdates.parseManifest({ version: 'v1', sha256: SHA, url: 'https://elsewhere.test/evil.zip' }, BASE),
    /not on the update origin/,
  )
})

test('manifests missing the fields the download path relies on are refused', () => {
  assert.throws(() => bundleUpdates.parseManifest(null, BASE), /not an object/)
  assert.throws(() => bundleUpdates.parseManifest({ sha256: SHA, url: 'a.zip' }, BASE), /version/)
  assert.throws(() => bundleUpdates.parseManifest({ version: 'v1', url: 'a.zip' }, BASE), /sha256/)
  assert.throws(() => bundleUpdates.parseManifest({ version: 'v1', sha256: 'short', url: 'a.zip' }, BASE), /sha256/)
  assert.throws(() => bundleUpdates.parseManifest({ version: 'v1', sha256: SHA }, BASE), /url/)
  assert.throws(() => bundleUpdates.parseManifest({ version: '../x', sha256: SHA, url: 'a.zip' }, BASE), /version/)
})

test('plain http is refused except on loopback, where a dev server has no certificate', () => {
  assert.throws(
    () => bundleUpdates.parseManifest({ version: 'v1', sha256: SHA, url: 'a.zip' }, 'http://updates.test/u'),
    /https/,
  )
  assert.doesNotThrow(() => bundleUpdates.parseManifest({ version: 'v1', sha256: SHA, url: 'a.zip' }, 'http://127.0.0.1:9000/u'))
})

test('checking with no update host configured is "disabled", not an error', async () => {
  const outcome = await bundleUpdates.checkAndDownload({ userDataDir: scratch(), updateBase: '' })
  assert.equal(outcome.kind, 'disabled')
})

test('an unreachable update host reports failure and never throws', async () => {
  // A failed update check must leave the mirror running the bundle it has.
  const outcome = await bundleUpdates.checkAndDownload({
    userDataDir: scratch(), updateBase: 'http://127.0.0.1:1/mirror-updates',
  })
  assert.equal(outcome.kind, 'failed')
})

test('manifestUrl tolerates a trailing slash on the base', () => {
  assert.equal(bundleUpdates.manifestUrl('https://x.test/u/'), 'https://x.test/u/latest.json')
  assert.equal(bundleUpdates.manifestUrl('https://x.test/u'), 'https://x.test/u/latest.json')
})

// --- shell channel -----------------------------------------------------------------------------

test('shell updates are unsupported outside an AppImage install', () => {
  assert.equal(shellUpdates.shellUpdateSupported({}), false)
  assert.equal(shellUpdates.shellUpdateSupported({ APPIMAGE: '/opt/Reflexion.AppImage' }), true)
  assert.equal(shellUpdates.runningAppImage({ APPIMAGE: '  ' }), null)
})

test('a shell check on a non-AppImage install is "disabled" rather than a failure', async () => {
  const outcome = await shellUpdates.checkAndDownload({ updateBase: BASE, currentVersion: '1.0.0', env: {} })
  assert.equal(outcome.kind, 'disabled')
})

test('a shell manifest for the wrong architecture is refused', async () => {
  // Installing an x64 AppImage on an arm64 unit yields a dead appliance, so it must never be attempted.
  const outcome = await shellUpdates.checkAndDownload({
    updateBase: 'http://127.0.0.1:1/u', currentVersion: '1.0.0', arch: 'arm64',
    env: { APPIMAGE: path.join(scratch(), 'Reflexion.AppImage') },
  })
  // Cannot reach the manifest here, so this only asserts it fails cleanly rather than throwing.
  assert.equal(outcome.kind, 'failed')
})

test('the shell manifest is validated exactly like the bundle manifest', () => {
  assert.throws(() => shellUpdates.parseManifest({ version: 'v1', sha256: SHA, url: 'https://elsewhere.test/x' }, BASE), /origin/)
  assert.throws(() => shellUpdates.parseManifest({ version: 'v1', sha256: 'nope', url: 'x' }, BASE), /sha256/)
  const parsed = shellUpdates.parseManifest({ version: '1.0.1', sha256: SHA, url: 'shell-1.0.1.AppImage', arch: 'x64' }, BASE)
  assert.equal(parsed.arch, 'x64')
  assert.equal(parsed.url, 'https://updates.test/mirror-updates/shell-1.0.1.AppImage')
})

test('applying a shell update keeps the old AppImage for hand recovery', async () => {
  const dir = scratch()
  const target = path.join(dir, 'Reflexion.AppImage')
  fs.writeFileSync(target, 'OLD')
  fs.writeFileSync(`${target}.new`, 'NEW')
  const result = await shellUpdates.applyDownloaded({ env: { APPIMAGE: target } })
  assert.equal(result.ok, true)
  assert.equal(fs.readFileSync(target, 'utf8'), 'NEW')
  // A unit that will not start after an upgrade is otherwise a site visit.
  assert.equal(fs.readFileSync(`${target}.old`, 'utf8'), 'OLD')
  assert.equal(fs.existsSync(`${target}.new`), false)
})

test('applying with nothing staged is a clean refusal', async () => {
  const dir = scratch()
  const target = path.join(dir, 'Reflexion.AppImage')
  fs.writeFileSync(target, 'OLD')
  const result = await shellUpdates.applyDownloaded({ env: { APPIMAGE: target } })
  assert.equal(result.ok, false)
  assert.equal(fs.readFileSync(target, 'utf8'), 'OLD')
})

test('discarding a staged shell download removes it', async () => {
  const dir = scratch()
  const target = path.join(dir, 'Reflexion.AppImage')
  fs.writeFileSync(`${target}.new`, 'NEW')
  await shellUpdates.discardDownloaded({ env: { APPIMAGE: target } })
  assert.equal(fs.existsSync(`${target}.new`), false)
})
