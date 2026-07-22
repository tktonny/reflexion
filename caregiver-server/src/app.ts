import cors from 'cors'
import express, { type ErrorRequestHandler } from 'express'
import { router } from './routes/router.js'
import { v1Router } from './v1/router.js'
import { apiErrorHandler, requestContext, v1NotFound } from './v1/platform/http.js'
import { rateLimit } from './v1/platform/rateLimit.js'
import { auditAccess } from './v1/platform/audit.js'

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
  app.use(express.json({ limit: '1mb' }))

  app.get('/health', (_request, response) => {
    response.json({ ok: true })
  })
  app.get('/healthcheck', (_request, response) => {
    response.json({ ok: true })
  })

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

function corsOptions(): cors.CorsOptions {
  const configured = process.env.CORS_ALLOWED_ORIGINS?.split(',').map((origin) => origin.trim()).filter(Boolean)
  return {
    origin: configured?.length ? configured : process.env.NODE_ENV === 'production' ? false : true,
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
