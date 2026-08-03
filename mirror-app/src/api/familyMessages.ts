import { dataOrThrow } from './devicePairing'
import { deviceFetch } from '../storage/deviceCredentials'

/**
 * A deliberately small one-way inbox. The Mirror polls while awake, then shows a notification without
 * exposing its contents until the older adult chooses to open it.
 */
export type FamilyMessage = {
  messageId: string
  patientId: string
  body: string
  type: 'text'
  state: 'delivered' | 'opened'
  scheduledFor: string
  createdAt: string
  deliveredAt: string | null
  openedAt: string | null
}

export async function fetchFamilyMessages(deviceId: string): Promise<FamilyMessage[]> {
  const response = await deviceFetch(`/api/v1/devices/${encodeURIComponent(deviceId)}/family-messages`)
  const data = await dataOrThrow<{ messages: FamilyMessage[] }>(response)
  return Array.isArray(data.messages) ? data.messages : []
}

export async function markFamilyMessageOpened(deviceId: string, messageId: string): Promise<FamilyMessage> {
  const response = await deviceFetch(`/api/v1/devices/${encodeURIComponent(deviceId)}/family-messages/${encodeURIComponent(messageId)}/opened`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
  })
  return dataOrThrow<FamilyMessage>(response)
}
