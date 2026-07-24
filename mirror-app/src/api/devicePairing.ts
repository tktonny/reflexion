import Constants from 'expo-constants'
import { Platform } from 'react-native'

import { getApiUrl } from '../config/apiUrl'
import { getBootstrapCredential, persistDeviceCredential, randomIdempotencyKey, type StoredDeviceCredential } from '../storage/deviceCredentials'

type Envelope<T> = { data?: T; error?: { code?: string; message?: string } }

export type V1Pairing = { pairingId: string; displayCode: string; state: 'pending'; expiresAt: string; pollAfterSeconds?: number }
export type V1PairingStatus = { pairingId: string; state: 'pending' | 'paired' | 'expired' | 'cancelled'; expiresAt: string; patientDisplayName?: string; exchangeTicket?: string; exchangeTicketExpiresAt?: string }
export type DeviceConfiguration = {
  deviceId: string
  configVersion: number
  desired?: Record<string, unknown>
  effectiveAt?: string
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

const CN_PROBE_URL = process.env.EXPO_PUBLIC_QWEN_CN_PROBE_URL || 'https://dashscope.aliyuncs.com'
const SG_PROBE_URL = process.env.EXPO_PUBLIC_QWEN_SG_PROBE_URL || 'https://ws-s37sbnnxivio0l58.ap-southeast-1.maas.aliyuncs.com'

/**
 * One-time reachability/latency probe of the CN vs SG Qwen hosts, run at pairing. Returns the faster
 * reachable region, or undefined if neither answered (backend then falls back to IP-geo / timezone).
 * This is the "probe wins, IP validates" signal. Best-effort and bounded — never blocks pairing.
 */
async function probeRegion(): Promise<'cn' | 'sg' | undefined> {
  const measure = async (url: string): Promise<number> => {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 3000)
    const start = Date.now()
    try {
      // Any HTTP response (even 401/404) proves reachability; fetch only rejects on a network error/abort.
      await fetch(url, { method: 'GET', signal: controller.signal })
      return Date.now() - start
    } catch {
      return Number.POSITIVE_INFINITY
    } finally {
      clearTimeout(timer)
    }
  }
  const [cn, sg] = await Promise.all([measure(CN_PROBE_URL), measure(SG_PROBE_URL)])
  if (!Number.isFinite(cn) && !Number.isFinite(sg)) return undefined
  return sg < cn ? 'sg' : 'cn'
}

export async function createDevicePairing() {
  const bootstrap = await getBootstrapCredential()
  if (!bootstrap) throw new Error('device_not_provisioned')
  const probedRegion = await probeRegion()
  const response = await fetch(getApiUrl('/api/v1/device-pairings'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Device-Bootstrap': bootstrap.token, 'Idempotency-Key': randomIdempotencyKey() },
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
  const bootstrap = await getBootstrapCredential()
  if (!bootstrap) throw new Error('device_not_provisioned')
  const response = await fetch(getApiUrl(`/api/v1/device-pairings/${encodeURIComponent(pairingId)}`), {
    headers: { 'X-Device-Bootstrap': bootstrap.token },
  })
  return dataOrThrow<V1PairingStatus>(response)
}

export async function exchangeDeviceCredential(pairing: V1PairingStatus) {
  if (!pairing.exchangeTicket) throw new Error('pairing_exchange_ticket_missing')
  const bootstrap = await getBootstrapCredential()
  if (!bootstrap) throw new Error('device_not_provisioned')
  const response = await fetch(getApiUrl('/api/v1/device-credentials/exchange'), {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Device-Bootstrap': bootstrap.token },
    body: JSON.stringify({ pairingId: pairing.pairingId, exchangeTicket: pairing.exchangeTicket }),
  })
  const credential = await dataOrThrow<StoredDeviceCredential>(response)
  await persistDeviceCredential(credential)
  return credential
}

export async function dataOrThrow<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null) as Envelope<T> | null
  if (!response.ok || !payload?.data) throw new Error(payload?.error?.code || `api_${response.status}`)
  return payload.data
}
