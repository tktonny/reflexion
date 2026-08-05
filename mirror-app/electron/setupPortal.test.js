// Tests for the phone-driven Wi-Fi setup portal. Run with: npm run test:network
//
// This portal binds to ALL interfaces so a phone can reach it over the mirror's hotspot, which makes its
// access control load-bearing rather than cosmetic. The PIN is the only thing standing between "within radio
// range of the mirror" and "can reconfigure the mirror's network", so the gate and its rate limit are pinned
// here, along with the response-before-apply ordering that the whole flow depends on.

const assert = require('node:assert/strict')
const test = require('node:test')

const { generatePin, renderPortalPage, startSetupPortal } = require('./setupPortal')

const PIN = '135790'

function harness(overrides = {}) {
  const applied = []
  const portal = startSetupPortal({
    pin: PIN,
    port: 0, // ephemeral: tests must not fight over a fixed port
    scanWifi: overrides.scanWifi || (async () => ({
      ok: true,
      networks: [{ ssid: 'HomeWifi', signal: 70, secured: true, saved: false, band: '2.4 GHz' }],
    })),
    status: overrides.status || (async () => ({ online: false, connectivity: 'none', activeConnection: null })),
    applyWifi: overrides.applyWifi || (async (options) => { applied.push(options); return { ok: true } }),
  })
  return { portal, applied }
}

/** startSetupPortal listens asynchronously; wait for the bound port before issuing requests. */
async function baseUrl(portal) {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const address = portal.address?.() ?? null
    if (address?.port) return `http://127.0.0.1:${address.port}`
    await new Promise((resolve) => setTimeout(resolve, 10))
  }
  throw new Error('portal never bound')
}

test('generatePin returns six digits and is not trivially constant', () => {
  const pins = new Set()
  for (let attempt = 0; attempt < 200; attempt += 1) {
    const pin = generatePin()
    assert.match(pin, /^\d{6}$/)
    pins.add(pin)
  }
  // 200 draws from 10^6 colliding into <5 distinct values would mean the generator is broken.
  assert.ok(pins.size > 100, `expected varied pins, got ${pins.size} distinct`)
})

test('the portal page is self-contained — no external resource can be required', () => {
  const html = renderPortalPage()
  // The phone has NO internet while joined to the mirror's hotspot, so any remote reference is a dead link.
  assert.doesNotMatch(html, /src="https?:\/\//)
  assert.doesNotMatch(html, /href="https?:\/\//)
  assert.doesNotMatch(html, /@import/)
  assert.match(html, /<title>Connect the Reflexion mirror<\/title>/)
})

test('GET / serves the page, and captive-portal probe paths serve it too', async () => {
  const { portal } = harness()
  const base = await baseUrl(portal)
  try {
    for (const path of ['/', '/hotspot-detect.html', '/generate_204', '/ncsi.txt']) {
      const response = await fetch(`${base}${path}`)
      assert.equal(response.status, 200, `${path} should serve the page`)
      assert.match(response.headers.get('content-type') || '', /text\/html/)
      // Answering the probe is what makes a phone auto-open the setup sheet instead of silently
      // declaring the hotspot useless and switching back to cellular.
      assert.match(await response.text(), /Connect the mirror to Wi-Fi/)
    }
  } finally {
    await portal.close()
  }
})

test('GET /api/networks lists SSIDs and never leaks a stored secret', async () => {
  const { portal } = harness({
    scanWifi: async () => ({
      ok: true,
      // Whatever the scanner hands over, the portal must project only non-secret fields.
      networks: [{ ssid: 'HomeWifi', signal: 70, secured: true, saved: true, band: '5 GHz', psk: 'must-not-appear' }],
    }),
  })
  const base = await baseUrl(portal)
  try {
    const response = await fetch(`${base}/api/networks`)
    const body = await response.json()
    assert.equal(response.status, 200)
    assert.equal(body.networks[0].ssid, 'HomeWifi')
    assert.equal(body.networks[0].psk, undefined)
    assert.doesNotMatch(JSON.stringify(body), /must-not-appear/)
  } finally {
    await portal.close()
  }
})

test('a wrong PIN is refused and does NOT apply anything', async () => {
  const { portal, applied } = harness()
  const base = await baseUrl(portal)
  try {
    const response = await fetch(`${base}/api/connect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ssid: 'HomeWifi', password: 'hunter2hunter2', pin: '000000' }),
    })
    assert.equal(response.status, 403)
    const body = await response.json()
    assert.equal(body.ok, false)
    assert.match(body.error, /not right/)
    await new Promise((resolve) => setTimeout(resolve, 1500)) // past the apply delay
    assert.deepEqual(applied, [], 'a wrong PIN must never reach the network layer')
  } finally {
    await portal.close()
  }
})

test('repeated wrong PINs are rate-limited so six digits cannot be brute-forced', async () => {
  const { portal, applied } = harness()
  const base = await baseUrl(portal)
  try {
    let sawLockout = false
    for (let attempt = 0; attempt < 15; attempt += 1) {
      const response = await fetch(`${base}/api/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ssid: 'HomeWifi', password: 'x', pin: '999999' }),
      })
      if (response.status === 429) { sawLockout = true; break }
    }
    assert.ok(sawLockout, 'the portal must stop accepting attempts')
    assert.deepEqual(applied, [])
  } finally {
    await portal.close()
  }
})

test('the correct PIN applies the network, and responds BEFORE applying', async () => {
  let resolveApply
  const applyGate = new Promise((resolve) => { resolveApply = resolve })
  const applied = []
  const portal = startSetupPortal({
    pin: PIN,
    port: 0,
    scanWifi: async () => ({ ok: true, networks: [] }),
    status: async () => ({ online: false, connectivity: 'none' }),
    applyWifi: async (options) => { applied.push(options); resolveApply(); return { ok: true } },
  })
  const base = await baseUrl(portal)
  try {
    const response = await fetch(`${base}/api/connect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ssid: 'HomeWifi', password: 'correct-horse', pin: PIN }),
    })
    // 202, not 200: applying takes the radio this very connection runs on, so the answer must be sent
    // first and the outcome reported on the mirror's screen instead.
    assert.equal(response.status, 202)
    const body = await response.json()
    assert.equal(body.ok, true)
    assert.equal(body.applying, true)
    assert.deepEqual(applied, [], 'apply must not have started before the response was sent')

    await applyGate
    assert.deepEqual(applied, [{ ssid: 'HomeWifi', password: 'correct-horse' }])
  } finally {
    await portal.close()
  }
})

test('a correct PIN with no network chosen is rejected', async () => {
  const { portal, applied } = harness()
  const base = await baseUrl(portal)
  try {
    const response = await fetch(`${base}/api/connect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ssid: '   ', password: '', pin: PIN }),
    })
    assert.equal(response.status, 400)
    await new Promise((resolve) => setTimeout(resolve, 1500))
    assert.deepEqual(applied, [])
  } finally {
    await portal.close()
  }
})

test('/api/status reports the last apply result so a failure survives the hotspot restart', async () => {
  const { portal } = harness({ applyWifi: async () => ({ ok: false, error: 'That password was not accepted.' }) })
  const base = await baseUrl(portal)
  try {
    await fetch(`${base}/api/connect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ssid: 'HomeWifi', password: 'wrong', pin: PIN }),
    })
    await new Promise((resolve) => setTimeout(resolve, 1600))
    const body = await (await fetch(`${base}/api/status`)).json()
    assert.equal(body.lastResult.ok, false)
    assert.match(body.lastResult.error, /not accepted/)
  } finally {
    await portal.close()
  }
})

test('unknown routes 404 — the portal exposes nothing beyond its three endpoints', async () => {
  const { portal } = harness()
  const base = await baseUrl(portal)
  try {
    // Specifically: it must NOT serve the mirror SPA or proxy the backend, which would hand a LAN
    // neighbour the device's credentials.
    for (const path of ['/api/v1/device-pairings', '/_expo/static/js/web/index.js', '/health', '/conversation']) {
      const response = await fetch(`${base}${path}`)
      assert.equal(response.status, 404, `${path} must not be served`)
    }
  } finally {
    await portal.close()
  }
})

test('an oversized body is rejected rather than buffered', async () => {
  const { portal } = harness()
  const base = await baseUrl(portal)
  try {
    const response = await fetch(`${base}/api/connect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ssid: 'x'.repeat(200_000), password: 'y', pin: PIN }),
    }).catch(() => null)
    // Either a 4xx/5xx or a destroyed connection is acceptable; silently accepting 200 KB is not.
    if (response) assert.ok(response.status >= 400, `expected rejection, got ${response.status}`)
  } finally {
    await portal.close()
  }
})
