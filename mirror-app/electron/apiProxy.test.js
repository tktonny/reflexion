// Tests for the Linux mirror's backend proxy. Run with:
//   npm run test:network
//
// The proxy is what fixed a shipped AppImage that could not reach the server at all: the renderer's
// loopback origin was not in production's CORS allowlist, so Chromium blocked every request before it left
// the device. These tests pin the two properties that make that impossible to regress:
//
//   1. the request that arrives upstream carries NO Origin/Referer and the upstream's own Host, so there is
//      nothing for a CORS allowlist to reject;
//   2. an unreachable backend produces the backend's own {error:{code,message,retryable}} envelope rather
//      than an HTML error page the renderer would choke on.
//
// A local stub server stands in for the backend so the suite is hermetic and offline-safe.

const assert = require('node:assert/strict')
const http = require('node:http')
const test = require('node:test')

const { createBackendProxy, isBackendPath, normalizeBase } = require('./apiProxy')

function listen(server) {
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve(server.address().port)))
}

function close(server) {
  return new Promise((resolve) => server.close(resolve))
}

/** Stand in for reflexion-server; records what the proxy actually forwarded. */
async function startUpstream(handler) {
  const received = []
  const server = http.createServer((req, res) => {
    const chunks = []
    req.on('data', (chunk) => chunks.push(chunk))
    req.on('end', () => {
      received.push({ method: req.method, url: req.url, headers: req.headers, body: Buffer.concat(chunks).toString() })
      handler(req, res)
    })
  })
  const port = await listen(server)
  return { received, port, base: `http://127.0.0.1:${port}`, stop: () => close(server) }
}

/** Stand in for the mirror's local SPA server, wired exactly as electron/main.js wires it. */
async function startMirrorServer(getApiBase) {
  const proxy = createBackendProxy(getApiBase)
  const server = http.createServer((req, res) => {
    const pathname = new URL(req.url, 'http://localhost').pathname
    if (isBackendPath(pathname)) { proxy(req, res, pathname); return }
    res.statusCode = 200
    res.end('spa')
  })
  const port = await listen(server)
  return { origin: `http://127.0.0.1:${port}`, stop: () => close(server) }
}

test('isBackendPath forwards API and health paths, and only those', () => {
  assert.ok(isBackendPath('/api/v1/device-pairings'))
  assert.ok(isBackendPath('/health'))
  assert.ok(isBackendPath('/healthcheck'))
  // Everything else must keep being served from dist/ — otherwise the SPA itself gets proxied away.
  assert.equal(isBackendPath('/'), false)
  assert.equal(isBackendPath('/conversation'), false)
  assert.equal(isBackendPath('/_expo/static/js/web/index.js'), false)
  assert.equal(isBackendPath('/assets/wakeword/wakeword.onnx'), false)
})

test('normalizeBase trims whitespace and a trailing slash', () => {
  assert.equal(normalizeBase('  https://example.com/  '), 'https://example.com')
  assert.equal(normalizeBase('https://example.com'), 'https://example.com')
  assert.equal(normalizeBase(undefined), '')
})

test('the proxy strips Origin and Referer and sets the upstream Host', async () => {
  const upstream = await startUpstream((_req, res) => { res.setHeader('Content-Type', 'application/json'); res.end('{"ok":true}') })
  const mirror = await startMirrorServer(() => upstream.base)
  try {
    const response = await fetch(`${mirror.origin}/api/v1/device-pairings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Idempotency-Key': 'mirror_test_1',
        'X-Device-Bootstrap': 'bootstrap-token',
        // These are exactly the headers a browser adds and that triggered the blocked preflight.
        Origin: mirror.origin,
        Referer: `${mirror.origin}/`,
      },
      body: JSON.stringify({ hello: 'world' }),
    })
    assert.equal(response.status, 200)
    assert.deepEqual(await response.json(), { ok: true })

    const forwarded = upstream.received.at(-1)
    assert.equal(forwarded.method, 'POST')
    assert.equal(forwarded.url, '/api/v1/device-pairings')
    assert.equal(forwarded.body, JSON.stringify({ hello: 'world' }))
    // The whole point: upstream sees no browser origin, so its CORS allowlist is never consulted.
    assert.equal(forwarded.headers.origin, undefined)
    assert.equal(forwarded.headers.referer, undefined)
    assert.equal(forwarded.headers.host, new URL(upstream.base).host)
    // The device's own auth headers must survive the hop.
    assert.equal(forwarded.headers['x-device-bootstrap'], 'bootstrap-token')
    assert.equal(forwarded.headers['idempotency-key'], 'mirror_test_1')
  } finally {
    await mirror.stop()
    await upstream.stop()
  }
})

test('the proxy preserves the query string and the upstream status code', async () => {
  const upstream = await startUpstream((_req, res) => { res.statusCode = 409; res.end('{"error":{"code":"IDEMPOTENCY_CONFLICT"}}') })
  const mirror = await startMirrorServer(() => upstream.base)
  try {
    const response = await fetch(`${mirror.origin}/api/v1/sessions?limit=5&cursor=abc`)
    assert.equal(response.status, 409)
    assert.equal(upstream.received.at(-1).url, '/api/v1/sessions?limit=5&cursor=abc')
  } finally {
    await mirror.stop()
    await upstream.stop()
  }
})

test('a base with a path prefix is honoured', async () => {
  const upstream = await startUpstream((_req, res) => res.end('{}'))
  const mirror = await startMirrorServer(() => `${upstream.base}/backend`)
  try {
    await fetch(`${mirror.origin}/health`)
    assert.equal(upstream.received.at(-1).url, '/backend/health')
  } finally {
    await mirror.stop()
    await upstream.stop()
  }
})

test('an unreachable backend returns the API error envelope, not an HTML error page', async () => {
  // Port 9 is the discard port: reliably closed, so this exercises the connection-refused path.
  const mirror = await startMirrorServer(() => 'http://127.0.0.1:9')
  try {
    const response = await fetch(`${mirror.origin}/health`)
    assert.equal(response.status, 502)
    const body = await response.json()
    assert.equal(body.error.code, 'MIRROR_API_UNREACHABLE')
    // `retryable` is what makes the mirror queue and retry instead of treating this as fatal.
    assert.equal(body.error.retryable, true)
  } finally {
    await mirror.stop()
  }
})

test('a malformed API base fails loudly instead of proxying somewhere unexpected', async () => {
  const mirror = await startMirrorServer(() => 'not-a-url')
  try {
    const response = await fetch(`${mirror.origin}/health`)
    assert.equal(response.status, 500)
    assert.equal((await response.json()).error.code, 'MIRROR_API_BASE_INVALID')
  } finally {
    await mirror.stop()
  }
})

test('non-backend paths are still served by the SPA server', async () => {
  const mirror = await startMirrorServer(() => 'http://127.0.0.1:9')
  try {
    const response = await fetch(`${mirror.origin}/conversation`)
    assert.equal(response.status, 200)
    assert.equal(await response.text(), 'spa')
  } finally {
    await mirror.stop()
  }
})
