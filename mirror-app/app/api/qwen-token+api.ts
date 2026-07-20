import type { RequestHandler } from 'expo-router/server'

import { verifyDevice } from '../../src/server/deviceAuth'

// Short-lived DashScope token mint. Keeps the long-term key server-side; the device/kiosk
// opens the realtime WS (v3) with the returned ephemeral token instead of a baked-in key.
// Verified live: POST /api/v1/tokens?expire_in_seconds=1800 -> { token, expires_at }.

declare const process: { env: Record<string, string | undefined> }

export const OPTIONS: RequestHandler = async () => Response.json({ ok: true })

export const POST: RequestHandler = async (request) => {
  const key = process.env.QWEN_API_KEY || process.env.DASHSCOPE_API_KEY
  if (!key) {
    return Response.json({ success: false, reason: 'missing_qwen_api_key' }, { status: 500 })
  }
  // Device auth (production): only mint for a paired device. Enable via REFLEXION_ENFORCE_DEVICE_AUTH=true.
  if (process.env.REFLEXION_ENFORCE_DEVICE_AUTH === 'true') {
    const uri = process.env.MONGODB_URI
    const cred = (await request.json().catch(() => null)) as { deviceId?: string; authToken?: string } | null
    const verified = uri ? await verifyDevice(uri, cred?.deviceId, cred?.authToken) : { ok: false }
    if (!verified.ok) return Response.json({ success: false, reason: 'unauthorized_device' }, { status: 401 })
  }
  const base = process.env.QWEN_BASE || 'https://dashscope.aliyuncs.com'
  const expire = process.env.QWEN_TOKEN_EXPIRE_SECONDS || '1800'
  try {
    const res = await fetch(`${base}/api/v1/tokens?expire_in_seconds=${expire}`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
      body: '{}',
    })
    const body = (await res.json()) as { token?: string; expires_at?: string }
    if (!res.ok || !body?.token) {
      return Response.json({ success: false, reason: 'mint_failed' }, { status: 502 })
    }
    return Response.json({ success: true, token: body.token, expiresAt: body.expires_at })
  } catch (error) {
    return Response.json(
      { success: false, reason: error instanceof Error ? error.message : 'unknown_token_error' },
      { status: 500 },
    )
  }
}
