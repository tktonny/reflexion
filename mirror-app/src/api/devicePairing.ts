import Constants from 'expo-constants'
import { Platform } from 'react-native'

import { getApiUrl } from '../config/apiUrl'
import {
  clearPendingCredentialExchange,
  getDevicePairingIdentity,
  getPairingIdempotencyKey,
  getPendingCredentialExchange,
  persistDeviceCredential,
  randomIdempotencyKey,
  savePendingCredentialExchange,
  type StoredDeviceCredential,
} from '../storage/deviceCredentials'

type Envelope<T> = {
  data?: T
  error?: { code?: string; message?: string; retryable?: boolean }
  meta?: { requestId?: string }
}

export class ApiRequestError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly code: string,
    public readonly retryable: boolean,
    public readonly requestId?: string,
  ) {
    super(message)
    this.name = 'ApiRequestError'
  }
}

export type V1Pairing = { deviceId: string; pairingId: string; displayCode: string; pairingToken: string; state: 'pending'; expiresAt: string; pollAfterSeconds?: number }
export type V1PairingStatus = { pairingId: string; state: 'pending' | 'paired' | 'expired' | 'cancelled'; expiresAt: string; patientDisplayName?: string; exchangeTicket?: string; exchangeTicketExpiresAt?: string }
export type DeviceConfiguration = {
  deviceId: string
  configVersion: number
  desired?: Record<string, unknown>
  effectiveAt?: string
  consent?: {
    product: 'pending' | 'accepted' | 'declined' | 'withdrawn'
    control: 'active' | 'paused'
    research?: 'not_invited' | 'invitation_pending' | 'consented' | 'declined' | 'withdrawn' | 'study_closed'
  }
  away?: {
    active: boolean
    startsOn?: string
    endsOn?: string
    timezone?: string
  }
  patient?: {
    patientId: string
    displayName: string
    preferredLanguage: string
    timezone: string
    version: number
    carePlan?: {
      version?: number
      communicationPreferences?: Record<string, unknown>
      dailyRoutine?: Record<string, unknown>
    } | null
  } | null
}

// Host ROOTS (overridable). We probe each region's token-mint path, which returns 401 quickly WITHOUT
// auth from the region's real origin — a fair, comparable RTT. (A plain GET to the SG realtime host,
// which is WebSocket-only, can hang until timeout and make a reachable region look unreachable.)
const CN_PROBE_URL = (process.env.EXPO_PUBLIC_QWEN_CN_PROBE_URL || 'https://dashscope.aliyuncs.com').replace(/\/+$/, '')
const SG_PROBE_URL = (process.env.EXPO_PUBLIC_QWEN_SG_PROBE_URL || 'https://ws-s37sbnnxivio0l58.ap-southeast-1.maas.aliyuncs.com').replace(/\/+$/, '')
const PROBE_TIMEOUT_MS = 1500
const PROBE_MARGIN_MS = 120

/**
 * One-time probe (at pairing) of the CN vs SG hosts. Returns a region ONLY when the signal is decisive —
 * exactly one host reachable, or a clear latency margin — otherwise undefined so the backend decides by
 * IP-geo (avoids a CDN-fronted CN edge beating the single-region SG origin on a device that is in SEA).
 * Best-effort and bounded (~1.5s). The backend cross-validates this against IP.
 */
async function probeRegion(): Promise<'cn' | 'sg' | undefined> {
  const measure = async (host: string): Promise<number> => {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS)
    const start = Date.now()
    try {
      // Any HTTP response (401 here) proves reachability; fetch only rejects on a network error/abort.
      await fetch(`${host}/api/v1/tokens?expire_in_seconds=60`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}', signal: controller.signal,
      })
      return Date.now() - start
    } catch {
      return Number.POSITIVE_INFINITY
    } finally {
      clearTimeout(timer)
    }
  }
  const [cn, sg] = await Promise.all([measure(CN_PROBE_URL), measure(SG_PROBE_URL)])
  const cnUp = Number.isFinite(cn)
  const sgUp = Number.isFinite(sg)
  if (!cnUp && !sgUp) return undefined              // neither reachable → backend uses IP/timezone
  if (cnUp !== sgUp) return cnUp ? 'cn' : 'sg'      // exactly one reachable → decisive
  if (Math.abs(cn - sg) < PROBE_MARGIN_MS) return undefined  // too close to call → defer to IP-geo
  return sg < cn ? 'sg' : 'cn'
}

export async function createDevicePairing(idempotencyKey?: string) {
  const identity = await getDevicePairingIdentity()
  const key = idempotencyKey || await getPairingIdempotencyKey()
  const probedRegion = await probeRegion()
  const requestId = newPairingRequestId()
  const response = await fetch(getApiUrl('/api/v1/device-pairings'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...identity.headers, 'Idempotency-Key': key, 'X-Request-Id': requestId },
    body: JSON.stringify({
      hardwareRevision: `${Platform.OS}-${String(Platform.Version)}`,
      softwareVersion: Constants.expoConfig?.version || 'unknown',
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC',
      ...(probedRegion ? { probedRegion } : {}),
    }),
  })
  return dataOrThrow<V1Pairing>(response)
}

export async function getDevicePairing(pairingId: string) {
  const identity = await getDevicePairingIdentity()
  const response = await fetch(getApiUrl(`/api/v1/device-pairings/${encodeURIComponent(pairingId)}`), {
    headers: { ...identity.headers, 'X-Request-Id': newPairingRequestId() },
  })
  return dataOrThrow<V1PairingStatus>(response)
}

export async function exchangeDeviceCredential(pairing: V1PairingStatus) {
  const stored = await getPendingCredentialExchange()
  const ticket = stored?.pairingId === pairing.pairingId ? stored.exchangeTicket : pairing.exchangeTicket
  if (!ticket) throw new Error('pairing_exchange_ticket_missing')
  const identity = await getDevicePairingIdentity()
  const idempotencyKey = stored?.pairingId === pairing.pairingId ? stored.idempotencyKey : randomIdempotencyKey()
  if (!stored || stored.pairingId !== pairing.pairingId) {
    await savePendingCredentialExchange({ pairingId: pairing.pairingId, exchangeTicket: ticket, idempotencyKey })
  }

  try {
    const requestId = newPairingRequestId()
    const response = await fetch(getApiUrl('/api/v1/device-credentials/exchange'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...identity.headers, 'Idempotency-Key': idempotencyKey, 'X-Request-Id': requestId },
      body: JSON.stringify({ pairingId: pairing.pairingId, exchangeTicket: ticket }),
    })
    const credential = await dataOrThrow<StoredDeviceCredential>(response)
    await persistDeviceCredential(credential)
    await clearPendingCredentialExchange()
    return credential
  } catch (error) {
    // A status poll may have rotated the short-lived delivery ticket after the first response was lost.
    // Replace only that pending delivery attempt; the permanent credential remains server-side and the
    // exchange route is idempotent for the original key.
    if (pairing.exchangeTicket && ticket !== pairing.exchangeTicket && error instanceof Error && error.message === 'EXCHANGE_TICKET_INVALID') {
      const nextKey = randomIdempotencyKey()
      await savePendingCredentialExchange({ pairingId: pairing.pairingId, exchangeTicket: pairing.exchangeTicket, idempotencyKey: nextKey })
      return exchangeDeviceCredential({ ...pairing, exchangeTicket: pairing.exchangeTicket })
    }
    throw error
  }
}

export async function dataOrThrow<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null) as Envelope<T> | null
  if (!response.ok || !payload?.data) {
    throw new ApiRequestError(
      payload?.error?.message || payload?.error?.code || `api_${response.status}`,
      response.status,
      payload?.error?.code || `HTTP_${response.status}`,
      Boolean(payload?.error?.retryable),
      payload?.meta?.requestId || response.headers.get('x-request-id') || undefined,
    )
  }
  return payload.data
}

function newPairingRequestId() {
  return `req_pair_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`
}
