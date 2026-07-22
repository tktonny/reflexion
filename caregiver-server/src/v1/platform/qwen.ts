import { ApiError } from './errors.js'

type QwenTokenResponse = { token?: string; expires_at?: string | number }

export async function createQwenRealtimeTicket(language?: string) {
  const apiKey = (process.env.QWEN_API_KEY || process.env.DASHSCOPE_API_KEY || '').trim()
  if (!apiKey) throw new ApiError(503, 'QWEN_NOT_CONFIGURED', 'Qwen credentials are not configured on this server.', true)
  const base = (process.env.QWEN_BASE || 'https://dashscope.aliyuncs.com').replace(/\/$/, '')
  const lifetime = Math.min(Math.max(Number(process.env.QWEN_TOKEN_EXPIRE_SECONDS || 900), 60), 3600)
  const upstream = await fetch(`${base}/api/v1/tokens?expire_in_seconds=${lifetime}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: '{}',
    signal: AbortSignal.timeout(10_000),
  })
  const body = await upstream.json().catch(() => null) as QwenTokenResponse | null
  if (!upstream.ok || !body?.token) {
    throw new ApiError(502, 'QWEN_TICKET_FAILED', 'Unable to create a Qwen session ticket.', true)
  }
  const expiresAt = normalizeExpiry(body.expires_at, lifetime)
  return {
    provider: 'qwen' as const,
    endpoint: process.env.QWEN_REALTIME_ENDPOINT || 'wss://dashscope.aliyuncs.com/api-ws/v1/realtime',
    ticket: body.token,
    expiresAt: expiresAt.toISOString(),
    sessionPolicy: {
      model: process.env.QWEN_REALTIME_MODEL || 'qwen3.5-omni-flash-realtime',
      language: language || 'zh-CN',
      modalities: ['audio', 'text', 'video'],
      clientMaySelectModel: false,
      clinicalDiagnosisAllowed: false,
    },
  }
}

function normalizeExpiry(value: string | number | undefined, fallbackSeconds: number) {
  if (typeof value === 'number') return new Date(value > 10_000_000_000 ? value : value * 1000)
  if (typeof value === 'string') {
    const numeric = Number(value)
    if (Number.isFinite(numeric)) return new Date(numeric > 10_000_000_000 ? numeric : numeric * 1000)
    const parsed = new Date(value)
    if (!Number.isNaN(parsed.getTime())) return parsed
  }
  return new Date(Date.now() + fallbackSeconds * 1000)
}
