import { dataOrThrow } from './devicePairing'
import { deviceFetch, getDeviceCredential, randomIdempotencyKey } from '../storage/deviceCredentials'

export type MirrorConsentStatus = 'granted' | 'declined' | 'withdrawn'

/** Records the older adult's choice. The server rejects caregiver-shaped or unrelated consent writes. */
export async function recordMirrorConsent(status: MirrorConsentStatus) {
  const credential = await getDeviceCredential()
  if (!credential) throw new Error('device_not_paired')
  const response = await deviceFetch(`/api/v1/devices/${encodeURIComponent(credential.deviceId)}/consent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Idempotency-Key': randomIdempotencyKey() },
    body: JSON.stringify({ status, documentVersion: 'checkin-consent-2026-07' }),
  })
  return dataOrThrow<{ status: MirrorConsentStatus }>(response)
}
