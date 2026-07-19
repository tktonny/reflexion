import cors from 'cors'
import express, { type ErrorRequestHandler } from 'express'
import { router } from './routes/router.js'

export function createApp() {
  const app = express()

  app.use(cors())
  app.use(express.json({ limit: '1mb' }))

  app.get('/health', (_request, response) => {
    response.json({ ok: true })
  })
  app.get('/healthcheck', (_request, response) => {
    response.json({ ok: true })
  })

  app.use(router)
  app.use(notFoundHandler)
  app.use(errorHandler)

  return app
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
