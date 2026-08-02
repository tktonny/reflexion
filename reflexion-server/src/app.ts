import cors from 'cors'
import express, { type ErrorRequestHandler } from 'express'
import { router } from './routes/router.js'
import { v1Router } from './v1/router.js'
import { apiErrorHandler, requestContext, v1NotFound } from './v1/platform/http.js'
import { rateLimit } from './v1/platform/rateLimit.js'
import { auditAccess } from './v1/platform/audit.js'
import { maybeMountLocalObjectStore } from './v1/platform/objectStoreLocal.js'

export function createApp() {
  const app = express()

  app.disable('x-powered-by')
  app.use(requestContext)
  app.use(cors(corsOptions()))
  app.use((_request, response, next) => {
    response.setHeader('X-Content-Type-Options', 'nosniff')
    response.setHeader('Referrer-Policy', 'no-referrer')
    response.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()')
    next()
  })
  // Local object-store upload target (if enabled) must be registered BEFORE express.json so the raw
  // binary artifact body is not JSON-parsed. It carries its own HMAC-signed URL token.
  maybeMountLocalObjectStore(app)

  app.use(express.json({ limit: '1mb' }))

  // `ok` stays the top-level field every existing probe (nginx, pm2, smoke:deployment) reads.
  // `readiness` is additive: it tells a mirror in ONE unauthenticated request whether the pieces its
  // conversation depends on are actually configured server-side. The device cannot check these itself —
  // it holds no provider key by design — and each of these has already broken a real check-in:
  // an empty OBJECT_STORE_DRIVER made every artifact upload 503 ("Aria needs a moment"), and a missing
  // regional Qwen key silently falls back to another region.
  // SECURITY: booleans and region names only. Never a key, a URI, or any fragment of one.
  const readiness = () => ({
    qwen: {
      cn: Boolean(process.env.QWEN_CN_API_KEY || process.env.QWEN_API_KEY),
      sg: Boolean(process.env.QWEN_SG_API_KEY || process.env.QWEN_API_KEY_SINGAPORE),
      jp: Boolean(process.env.QWEN_JP_API_KEY),
      defaultRegion: process.env.QWEN_DEFAULT_REGION || 'cn',
    },
    objectStore: Boolean(process.env.OBJECT_STORE_DRIVER),
    database: Boolean(process.env.MONGODB_URI),
  })
  const health = (_request: express.Request, response: express.Response) => {
    // Date lets the device measure clock skew — the wake-bounded "first conversation of the day" rule
    // silently breaks when the mirror's clock drifts, and express sets Date on every response anyway.
    response.json({ ok: true, serverTime: new Date().toISOString(), readiness: readiness() })
  }
  app.get('/health', health)
  app.get('/healthcheck', health)

  app.use('/api/v1/auth', rateLimit({ namespace: 'auth', maximum: Number(process.env.AUTH_RATE_LIMIT_PER_MINUTE || 20) }))
  app.use('/api/v1', rateLimit({ namespace: 'api', maximum: Number(process.env.API_RATE_LIMIT_PER_MINUTE || 300) }), auditAccess, v1Router)
  if (process.env.ENABLE_LEGACY_API === 'true') {
    app.use((_request, response, next) => {
      response.setHeader('Deprecation', 'true')
      response.setHeader('Sunset', process.env.LEGACY_API_SUNSET || 'Thu, 31 Dec 2026 23:59:59 GMT')
      next()
    })
    app.use(router)
  }
  app.use(v1NotFound)
  app.use(apiErrorHandler)
  app.use(notFoundHandler)
  app.use(errorHandler)

  return app
}

// The Linux (Electron) mirror is a Chromium renderer served from a LOOPBACK http server inside the
// appliance itself, so every backend call it makes is cross-origin from `http://127.0.0.1:<web port>`.
// A production `CORS_ALLOWED_ORIGINS` that lists only the admin SPA therefore blocks the whole mirror at
// the browser's preflight — the device looks "unable to reach the server" while the API is perfectly
// healthy, which is exactly how a shipped AppImage failed in the field. These origins are always allowed:
// loopback is not a site an attacker can host from, and the device still has to present a real device
// credential. Newer builds proxy through their own loopback origin and never rely on this, but AppImages
// already installed on Ubuntu units do.
export const MIRROR_LOOPBACK_ORIGINS = ['8899', '8081', '19006', '3000'].flatMap((port) => [
  `http://127.0.0.1:${port}`,
  `http://localhost:${port}`,
])

export function corsOptions(): cors.CorsOptions {
  const configured = process.env.CORS_ALLOWED_ORIGINS?.split(',').map((origin) => origin.trim()).filter(Boolean)
  const allowed = configured?.length
    ? [...new Set([...configured, ...MIRROR_LOOPBACK_ORIGINS])]
    : process.env.NODE_ENV === 'production' ? MIRROR_LOOPBACK_ORIGINS : true
  return {
    origin: allowed,
    credentials: true,
    allowedHeaders: ['Authorization', 'Content-Type', 'Idempotency-Key', 'If-Match', 'X-Request-Id', 'X-Device-Bootstrap'],
    exposedHeaders: ['X-Request-Id', 'Deprecation', 'Sunset'],
  }
}

function notFoundHandler(_request: express.Request, response: express.Response) {
  response.status(404).json({ error: 'Not found' })
}

const errorHandler: ErrorRequestHandler = (error, _request, response, _next) => {
  const status = typeof error?.status === 'number' ? error.status : 500
  response.status(status).json({
    error: error instanceof Error ? error.message : 'Internal server error',
  })
}

const app = createApp()

export default app
