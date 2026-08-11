// End-to-end OTA test against a real local update host.
//
// The unit tests cover the state machine; this covers the parts that only break in reality: the HTTP fetch,
// the sha256 comparison against bytes actually on the wire, `unzip` producing a servable tree, and the
// refusal paths (bad digest, missing index.html) leaving the unit on its previous bundle rather than
// half-updated. An appliance in someone's home has no operator, so "fails safe" has to be demonstrated.

const assert = require('node:assert/strict')
const { createHash } = require('node:crypto')
const { execFileSync } = require('node:child_process')
const fs = require('node:fs')
const http = require('node:http')
const os = require('node:os')
const path = require('node:path')
const { test } = require('node:test')

const bundleUpdates = require('./bundleUpdates')

function scratch() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'reflexion-ota-e2e-'))
}

/** Build a real zip whose root holds index.html, like a published bundle. */
function makeBundleZip(dir, { withIndex = true } = {}) {
  const staging = path.join(dir, 'staging')
  fs.mkdirSync(path.join(staging, '_expo'), { recursive: true })
  if (withIndex) fs.writeFileSync(path.join(staging, 'index.html'), '<html>mirror</html>')
  fs.writeFileSync(path.join(staging, '_expo', 'entry.js'), 'console.log("bundle")')
  const archive = path.join(dir, 'bundle.zip')
  execFileSync('zip', ['-qr', archive, '.'], { cwd: staging })
  return archive
}

/** Serve a manifest plus an archive over loopback (http is permitted on 127.0.0.1 for exactly this). */
async function startHost({ archive, manifest }) {
  const body = fs.readFileSync(archive)
  const server = http.createServer((req, res) => {
    if (req.url === '/latest.json') {
      res.setHeader('content-type', 'application/json')
      res.end(JSON.stringify(manifest(body)))
      return
    }
    if (req.url === '/bundle.zip') {
      res.setHeader('content-type', 'application/zip')
      res.end(body)
      return
    }
    res.statusCode = 404
    res.end('nope')
  })
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  return { server, base: `http://127.0.0.1:${server.address().port}`, close: () => new Promise((r) => server.close(r)) }
}

const digestOf = (buffer) => createHash('sha256').update(buffer).digest('hex')

test('a published bundle downloads, verifies, installs, and is served after apply', async () => {
  const dir = scratch()
  const archive = makeBundleZip(dir)
  const host = await startHost({
    archive,
    manifest: (body) => ({ version: '2026.08.11-1', sha256: digestOf(body), url: 'bundle.zip' }),
  })
  try {
    const outcome = await bundleUpdates.checkAndDownload({ userDataDir: dir, updateBase: host.base, packagedVersion: null })
    assert.equal(outcome.kind, 'downloaded', outcome.detail)

    // Downloaded but NOT yet live: applying is a separate step so it never happens mid-conversation.
    assert.equal(bundleUpdates.activeWebDir(dir, '/packaged'), '/packaged')

    const applied = await bundleUpdates.applyPending(dir)
    assert.equal(applied.ok, true)
    const served = bundleUpdates.activeWebDir(dir, '/packaged')
    assert.notEqual(served, '/packaged')
    assert.equal(fs.readFileSync(path.join(served, 'index.html'), 'utf8'), '<html>mirror</html>')
    // Assets alongside index.html must survive the unzip, or the mirror loads a blank page.
    assert.equal(fs.existsSync(path.join(served, '_expo', 'entry.js')), true)

    // A second check finds nothing new.
    const again = await bundleUpdates.checkAndDownload({ userDataDir: dir, updateBase: host.base, packagedVersion: null })
    assert.equal(again.kind, 'up_to_date')
  } finally {
    await host.close()
  }
})

test('a bundle whose bytes do not match the manifest digest is refused and nothing is installed', async () => {
  const dir = scratch()
  const archive = makeBundleZip(dir)
  const host = await startHost({
    archive,
    // A truncated or tampered download: the digest is what catches it before anything is unpacked.
    manifest: () => ({ version: '2026.08.11-bad', sha256: 'b'.repeat(64), url: 'bundle.zip' }),
  })
  try {
    const outcome = await bundleUpdates.checkAndDownload({ userDataDir: dir, updateBase: host.base, packagedVersion: null })
    assert.equal(outcome.kind, 'failed')
    assert.match(outcome.detail, /checksum/)
    assert.equal(bundleUpdates.readState(dir).pending ?? null, null)
    assert.equal(fs.existsSync(path.join(dir, 'bundles', '2026.08.11-bad')), false)
    // And no stray archive left behind on the appliance's disk.
    assert.equal(fs.existsSync(path.join(dir, 'bundles', '2026.08.11-bad.zip')), false)
  } finally {
    await host.close()
  }
})

test('an archive with no index.html is rejected rather than becoming a blank mirror', async () => {
  const dir = scratch()
  const archive = makeBundleZip(dir, { withIndex: false })
  const host = await startHost({
    archive,
    manifest: (body) => ({ version: '2026.08.11-noindex', sha256: digestOf(body), url: 'bundle.zip' }),
  })
  try {
    const outcome = await bundleUpdates.checkAndDownload({ userDataDir: dir, updateBase: host.base, packagedVersion: null })
    assert.equal(outcome.kind, 'failed')
    assert.match(outcome.detail, /index\.html/)
    assert.equal(fs.existsSync(path.join(dir, 'bundles', '2026.08.11-noindex')), false)
  } finally {
    await host.close()
  }
})

test('a bundle that never boots is rolled back and then refused on the next check', async () => {
  const dir = scratch()
  const archive = makeBundleZip(dir)
  const host = await startHost({
    archive,
    manifest: (body) => ({ version: '2026.08.11-1', sha256: digestOf(body), url: 'bundle.zip' }),
  })
  try {
    await bundleUpdates.checkAndDownload({ userDataDir: dir, updateBase: host.base, packagedVersion: null })
    await bundleUpdates.applyPending(dir)
    // Simulate the unit restarting without the renderer ever calling markBooted — i.e. a white screen.
    const rolled = await bundleUpdates.rollbackIfLastBootFailed(dir)
    assert.equal(rolled.rolledBack, true)
    assert.equal(bundleUpdates.activeWebDir(dir, '/packaged'), '/packaged')

    // Pressing "check for update" again must not reinstall the bundle that just failed, or the unit loops.
    const retry = await bundleUpdates.checkAndDownload({ userDataDir: dir, updateBase: host.base, packagedVersion: null })
    assert.equal(retry.kind, 'failed')
    assert.match(retry.detail, /failed to start/)
  } finally {
    await host.close()
  }
})

test('a manifest pointing off the update origin is refused before any download', async () => {
  const dir = scratch()
  const archive = makeBundleZip(dir)
  const host = await startHost({
    archive,
    manifest: () => ({ version: 'v9', sha256: 'c'.repeat(64), url: 'https://evil.test/payload.zip' }),
  })
  try {
    const outcome = await bundleUpdates.checkAndDownload({ userDataDir: dir, updateBase: host.base, packagedVersion: null })
    assert.equal(outcome.kind, 'failed')
    assert.match(outcome.detail, /update origin/)
  } finally {
    await host.close()
  }
})
