import assert from 'node:assert/strict'
import test from 'node:test'
import cors from 'cors'
import express from 'express'
import request from 'supertest'

import { MIRROR_LOOPBACK_ORIGINS, corsOptions } from './app.js'

/**
 * A shipped Linux mirror AppImage could not talk to production at all: its renderer is a page on
 * http://127.0.0.1:8899, production's CORS_ALLOWED_ORIGINS named only the admin SPA, so Chromium blocked
 * every request at the preflight and the device reported "unable to reach the Reflexion service" against
 * a healthy API. These tests pin the fix — the mirror's loopback origin is allowed even when an operator's
 * allowlist does not mention it — and pin that the allowlist still rejects arbitrary origins.
 *
 * The middleware is exercised on a bare express app: CORS runs before any route, so this needs no database.
 */
function appWithCors() {
  const app = express()
  app.use(cors(corsOptions()))
  app.post('/api/v1/device-pairings', (_request, response) => response.json({ ok: true }))
  return app
}

function preflight(app: express.Express, origin: string) {
  return request(app)
    .options('/api/v1/device-pairings')
    .set('Origin', origin)
    .set('Access-Control-Request-Method', 'POST')
    .set('Access-Control-Request-Headers', 'x-device-bootstrap,idempotency-key')
}

async function withEnv(env: Record<string, string | undefined>, run: () => Promise<void>) {
  const previous = { ...process.env }
  Object.assign(process.env, env)
  for (const [key, value] of Object.entries(env)) if (value === undefined) delete process.env[key]
  try {
    await run()
  } finally {
    process.env = previous
  }
}

test('mirror loopback origin is allowed even when the operator allowlist omits it', async () => {
  await withEnv({
    NODE_ENV: 'production',
    CORS_ALLOWED_ORIGINS: 'https://admin.reflexion.production.tktonny.top',
  }, async () => {
    const app = appWithCors()
    const response = await preflight(app, 'http://127.0.0.1:8899')
    assert.equal(response.headers['access-control-allow-origin'], 'http://127.0.0.1:8899')
    assert.match(response.headers['access-control-allow-headers'] ?? '', /X-Device-Bootstrap/)
  })
})

test('the configured operator origin keeps working', async () => {
  await withEnv({
    NODE_ENV: 'production',
    CORS_ALLOWED_ORIGINS: 'https://admin.reflexion.production.tktonny.top',
  }, async () => {
    const app = appWithCors()
    const response = await preflight(app, 'https://admin.reflexion.production.tktonny.top')
    assert.equal(response.headers['access-control-allow-origin'], 'https://admin.reflexion.production.tktonny.top')
  })
})

test('an unrelated origin is still refused in production', async () => {
  await withEnv({
    NODE_ENV: 'production',
    CORS_ALLOWED_ORIGINS: 'https://admin.reflexion.production.tktonny.top',
  }, async () => {
    const app = appWithCors()
    const response = await preflight(app, 'https://evil.example.com')
    assert.equal(response.headers['access-control-allow-origin'], undefined)
  })
})

test('production with no allowlist configured still admits the mirror and nothing else', async () => {
  await withEnv({ NODE_ENV: 'production', CORS_ALLOWED_ORIGINS: undefined }, async () => {
    const app = appWithCors()
    const allowed = await preflight(app, MIRROR_LOOPBACK_ORIGINS[0])
    assert.equal(allowed.headers['access-control-allow-origin'], MIRROR_LOOPBACK_ORIGINS[0])
    const refused = await preflight(app, 'https://evil.example.com')
    assert.equal(refused.headers['access-control-allow-origin'], undefined)
  })
})

test('localhost and 127.0.0.1 are both covered — Electron may serve either', () => {
  assert.ok(MIRROR_LOOPBACK_ORIGINS.includes('http://127.0.0.1:8899'))
  assert.ok(MIRROR_LOOPBACK_ORIGINS.includes('http://localhost:8899'))
})
