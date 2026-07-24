import { ApiError } from './errors.js'

/** Which Qwen region a device's traffic is routed to. 'cn' = mainland China (dashscope.aliyuncs.com),
 *  'sg' = Singapore / ap-southeast-1 workspace host. Chosen per device at pairing and stored on the
 *  device record; the device only ever receives a short-lived ticket + endpoints, never a key. */
export type QwenRegion = 'cn' | 'sg'

type QwenTokenResponse = { token?: string; expires_at?: string | number }

type QwenRegionConfig = {
  apiKey: string
  /** Host root, e.g. https://dashscope.aliyuncs.com. Token mint AND the device's HTTP
   *  TTS/ASR/chat/vision calls all hang off this ({host}/api/v1/tokens, {host}/compatible-mode/v1/...). */
  httpBase: string
  /** Realtime WS endpoint the device connects to. */
  realtimeEndpoint: string
}

const DEFAULT_CN_HOST = 'https://dashscope.aliyuncs.com'
const DEFAULT_SG_HOST = 'https://ws-s37sbnnxivio0l58.ap-southeast-1.maas.aliyuncs.com'

const trimSlash = (value: string) => value.trim().replace(/\/+$/, '')
const wsRealtimeFrom = (host: string) => `${trimSlash(host).replace(/^http/, 'ws')}/api-ws/v1/realtime`

/** First non-empty env var among the given names (new canonical name first, legacy fallbacks after). */
function firstEnv(...names: string[]): string {
  for (const name of names) {
    const value = (process.env[name] || '').trim()
    if (value) return value
  }
  return ''
}

/** Reduce any host spelling — bare host, host root, or a base ending in /api/v1 or /compatible-mode/v1 —
 *  to the scheme+host root the token-mint and HTTP clients build on. */
function normalizeHost(raw: string, fallback: string): string {
  const value = (raw || '').trim()
  if (!value) return fallback
  const withScheme = /^https?:\/\//.test(value) ? value : `https://${value}`
  return trimSlash(withScheme).replace(/\/(api\/v1|compatible-mode\/v1)$/, '')
}

/** Coerce an arbitrary value (device field, env) to a valid region. Anything that looks like Singapore /
 *  Southeast Asia maps to 'sg'; everything else is 'cn' — the backward-compatible default, so a legacy
 *  device with no region field keeps its current China routing. */
export function normalizeRegion(value: unknown): QwenRegion {
  const v = String(value ?? '').trim().toLowerCase()
  return v === 'sg' || v === 'sea' || v === 'singapore' || v === 'ap-southeast-1' ? 'sg' : 'cn'
}

function regionConfig(region: QwenRegion): QwenRegionConfig {
  if (region === 'sg') {
    // New canonical names: QWEN_SG_API_KEY / QWEN_SG_HTTP_BASE / QWEN_SG_REALTIME_ENDPOINT.
    // Legacy fallbacks: QWEN_API_KEY_SINGAPORE, QWEN_BASE_SINGAPORE | QWEN_API_HOST | DASHSCOPE_BASE.
    const host = normalizeHost(
      firstEnv('QWEN_SG_HTTP_BASE', 'QWEN_BASE_SINGAPORE', 'QWEN_API_HOST', 'DASHSCOPE_BASE'),
      DEFAULT_SG_HOST,
    )
    return {
      apiKey: firstEnv('QWEN_SG_API_KEY', 'QWEN_API_KEY_SINGAPORE'),
      httpBase: host,
      realtimeEndpoint: firstEnv('QWEN_SG_REALTIME_ENDPOINT', 'QWEN_REALTIME_ENDPOINT_SINGAPORE') || wsRealtimeFrom(host),
    }
  }
  // China. New canonical: QWEN_CN_*. Legacy fallbacks: QWEN_API_KEY / DASHSCOPE_API_KEY / QWEN_BASE /
  // QWEN_REALTIME_ENDPOINT (the names the current prod .env already uses).
  const host = normalizeHost(firstEnv('QWEN_CN_HTTP_BASE', 'QWEN_BASE'), DEFAULT_CN_HOST)
  return {
    apiKey: firstEnv('QWEN_CN_API_KEY', 'QWEN_API_KEY', 'DASHSCOPE_API_KEY'),
    httpBase: host,
    realtimeEndpoint: firstEnv('QWEN_CN_REALTIME_ENDPOINT', 'QWEN_REALTIME_ENDPOINT') || wsRealtimeFrom(host),
  }
}

/**
 * Mint a short-lived DashScope token for the given region and return it with the region-appropriate
 * realtime endpoint + HTTP base, so the device connects to the right host without ever holding a key.
 * The token-exchange is identical across regions (POST {host}/api/v1/tokens — verified on ap-southeast-1).
 */
export async function createQwenRealtimeTicket(language?: string, region: QwenRegion = 'cn') {
  const cfg = regionConfig(region)
  if (!cfg.apiKey) {
    throw new ApiError(503, 'QWEN_NOT_CONFIGURED', `Qwen credentials are not configured for region '${region}'.`, true)
  }
  const lifetime = Math.min(Math.max(Number(process.env.QWEN_TOKEN_EXPIRE_SECONDS || 900), 60), 3600)
  const upstream = await fetch(`${cfg.httpBase}/api/v1/tokens?expire_in_seconds=${lifetime}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${cfg.apiKey}`, 'Content-Type': 'application/json' },
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
    region,
    // Realtime WS endpoint (the device MUST use this rather than a build-time URL to be region-correct).
    endpoint: cfg.realtimeEndpoint,
    // HTTP host root for the device's turn-based TTS/ASR/chat/vision calls (same region as the token).
    httpBase: cfg.httpBase,
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
