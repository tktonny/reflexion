import { Router } from 'express'
import { asyncHandler } from '../lib/asyncHandler.js'
import { DB_NAME, PAIRING_COLLECTION } from '../lib/constants.js'
import { withMongo } from '../lib/mongo.js'

type TokenRequestBody = {
  deviceId?: string
  authToken?: string
}

type DashScopeTokenResponse = {
  token?: string
  expires_at?: string
}

const DEFAULT_TOKEN_LIFETIME_SECONDS = 1800
const MAX_TOKEN_LIFETIME_SECONDS = 3600

export const qwenTokenRouter = Router()

qwenTokenRouter.post('/', asyncHandler(async (request, response) => {
  response.setHeader('Cache-Control', 'no-store')

  const body = request.body as TokenRequestBody | null
  const deviceId = body?.deviceId?.trim()
  const authToken = body?.authToken?.trim()
  if (!deviceId || !authToken) {
    response.status(401).json({ success: false, reason: 'unauthorized_device' })
    return
  }

  const paired = await withMongo(async (client) => {
    const session = await client.db(DB_NAME).collection(PAIRING_COLLECTION).findOne({
      deviceId,
      authToken,
      status: 'paired',
    })
    return Boolean(session)
  })

  if (!paired) {
    response.status(401).json({ success: false, reason: 'unauthorized_device' })
    return
  }

  const apiKey = process.env.QWEN_API_KEY || process.env.DASHSCOPE_API_KEY
  if (!apiKey) {
    response.status(503).json({ success: false, reason: 'qwen_not_configured' })
    return
  }

  const configuredLifetime = Number(process.env.QWEN_TOKEN_EXPIRE_SECONDS)
  const lifetimeSeconds = Number.isFinite(configuredLifetime)
    ? Math.min(Math.max(Math.trunc(configuredLifetime), 60), MAX_TOKEN_LIFETIME_SECONDS)
    : DEFAULT_TOKEN_LIFETIME_SECONDS
  const baseUrl = (process.env.QWEN_BASE || 'https://dashscope.aliyuncs.com').replace(/\/$/, '')

  const upstream = await fetch(`${baseUrl}/api/v1/tokens?expire_in_seconds=${lifetimeSeconds}`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: '{}',
    signal: AbortSignal.timeout(10_000),
  })
  const tokenBody = await upstream.json().catch(() => null) as DashScopeTokenResponse | null
  if (!upstream.ok || !tokenBody?.token) {
    response.status(502).json({ success: false, reason: 'token_mint_failed' })
    return
  }

  response.json({
    success: true,
    token: tokenBody.token,
    expiresAt: tokenBody.expires_at,
  })
}))
