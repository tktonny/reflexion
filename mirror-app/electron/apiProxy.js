// Backend proxy for the Linux (Ubuntu) mirror's local web server.
//
// WHY THIS EXISTS — the bug it fixes:
//
// The Electron build serves the exported SPA from a loopback HTTP server (file:// breaks Expo Router), so
// the renderer is a page on `http://127.0.0.1:8899`. Calling `https://reflexion.production...` from there
// is a CROSS-ORIGIN request, and because the mirror sends `Authorization`, `Idempotency-Key` and
// `X-Device-Bootstrap`, it is a *preflighted* one. Production's `CORS_ALLOWED_ORIGINS` listed only the
// admin SPA, so every preflight came back without `Access-Control-Allow-Origin` and Chromium blocked the
// request before it left the device. The mirror showed "unable to reach the Reflexion service" while the
// API was perfectly healthy — and nothing was logged server-side, because the requests never arrived.
//
// Forwarding /api and /health through the SAME loopback origin the page is served from removes the
// cross-origin condition entirely: no preflight, no allowlist to be missing from. It also moves the API
// origin from a compile-time constant baked into the bundle to runtime device config.
//
// This lives in its own module (rather than inside main.js) so it can be tested without the Electron
// binary — see electron/apiProxy.test.js.

const http = require('http')
const https = require('https')

const DEFAULT_API_BASE = 'https://reflexion.production.tktonny.top'

/** Paths the local server forwards to the backend instead of serving from dist/. */
function isBackendPath(pathname) {
  return pathname.startsWith('/api/') || pathname === '/health' || pathname === '/healthcheck'
}

function normalizeBase(base) {
  return String(base || '').trim().replace(/\/$/, '')
}

/**
 * Build the request handler that forwards one backend call upstream.
 *
 * @param {() => string} getApiBase resolved lazily so a config reload can re-point the device without a
 *   restart, and so the base is not captured before Electron is ready.
 */
function createBackendProxy(getApiBase) {
  return function proxyToBackend(req, res, pathname) {
    const apiBase = normalizeBase(getApiBase())
    let target
    try {
      target = new URL(apiBase)
      if (target.protocol !== 'http:' && target.protocol !== 'https:') throw new Error('unsupported protocol')
    } catch {
      res.statusCode = 500
      res.setHeader('Content-Type', 'application/json')
      res.end(JSON.stringify({ error: { code: 'MIRROR_API_BASE_INVALID', message: `Invalid API base: ${apiBase}`, retryable: false } }))
      return
    }

    const search = req.url.includes('?') ? req.url.slice(req.url.indexOf('?')) : ''
    const transport = target.protocol === 'http:' ? http : https
    const headers = { ...req.headers }
    // The upstream must see its OWN host (virtual hosting + TLS SNI), and must NOT see the loopback
    // Origin/Referer — forwarding those would put the request straight back into the CORS check this
    // proxy exists to avoid.
    headers.host = target.host
    delete headers.origin
    delete headers.referer
    delete headers.connection
    // Length/encoding are re-derived by the upstream request from what we actually pipe.
    delete headers['content-length']

    const fail = (code, message) => {
      console.warn(`[electron] api proxy ${req.method} ${pathname} failed: ${message}`)
      if (res.headersSent) { res.destroy(); return }
      res.statusCode = 502
      res.setHeader('Content-Type', 'application/json')
      // Shaped like the backend's own {error:{code,message,retryable}} envelope so the renderer's existing
      // error handling keeps working instead of choking on an HTML error page.
      res.end(JSON.stringify({ error: { code, message, retryable: true } }))
    }

    const upstream = transport.request({
      protocol: target.protocol,
      hostname: target.hostname,
      port: target.port || (target.protocol === 'http:' ? 80 : 443),
      method: req.method,
      path: `${target.pathname.replace(/\/$/, '')}${pathname}${search}`,
      headers,
      timeout: 60_000,
    }, (upstreamResponse) => {
      res.writeHead(upstreamResponse.statusCode || 502, upstreamResponse.headers)
      upstreamResponse.pipe(res)
    })

    upstream.on('timeout', () => { upstream.destroy(); fail('MIRROR_API_TIMEOUT', 'The Reflexion service did not respond in time.') })
    upstream.on('error', (error) => fail('MIRROR_API_UNREACHABLE', error.message))
    req.pipe(upstream)
  }
}

module.exports = { DEFAULT_API_BASE, createBackendProxy, isBackendPath, normalizeBase }
