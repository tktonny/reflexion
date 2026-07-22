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

export async function createDevicePairing() {
  const bootstrap = await getBootstrapCredential()
  if (!bootstrap) throw new Error('device_not_provisioned')
  const response = await fetch(getApiUrl('/api/v1/device-pairings'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Device-Bootstrap': bootstrap.token, 'Idempotency-Key': randomIdempotencyKey() },
    body: JSON.stringify({
      hardwareRevision: `${Platform.OS}-${String(Platform.Version)}`,
      softwareVersion: Constants.expoConfig?.version || 'unknown',
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC',
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
