import { queuePendingConversation } from '../storage/conversationQueue'
import { getStoredMirrorSessionMetadata } from '../storage/mirrorStorage'
import { getDeviceCredential } from '../storage/deviceCredentials'
import type { ScreeningAssessment } from './assess'
import type { ChatMessage } from '../hooks/conversationTypes'
import { buildPendingSessionCompletion, clearActiveMirrorSession, uploadPendingSessionCompletion } from './sessionSync'

export type CheckinArgs = {
  messages: ChatMessage[]
  startedAt: Date
  endedAt: Date
  nurseId?: string
  patientId?: string
  deviceId?: string
  authToken?: string
  language?: string
  assessment?: ScreeningAssessment | null
  cameraFrames?: string[]
}

/** Builds a durable v1 session completion. Client-side assessment is deliberately not clinical evidence. */
export async function buildCheckinPayload(args: CheckinArgs) {
  return buildPendingSessionCompletion(args.messages, args.startedAt, args.endedAt,
    (args.cameraFrames || []).map((frame) => ({
      kind: 'image' as const,
      contentType: 'image/jpeg' as const,
      dataBase64: frame.replace(/^data:image\/jpeg;base64,/, ''),
    })))
}

export async function resolveOwnerIds(): Promise<{ nurseId: string; patientId: string; deviceId?: string; language?: string }> {
  const [credential, metadata] = await Promise.all([getDeviceCredential(), getStoredMirrorSessionMetadata()])
  if (!credential) throw new Error('device_not_paired')
  return { nurseId: '', patientId: credential.patientId, deviceId: credential.deviceId, language: metadata.language ?? undefined }
}

/** Uploads transcript events and completes the server session; failures enter the durable local outbox. */
export async function saveCheckin(args: CheckinArgs): Promise<{ saved: boolean; sessionId: string; operationId?: string; reason?: string }> {
  const pending = await buildCheckinPayload(args)
  try {
    const result = await uploadPendingSessionCompletion(pending)
    return { saved: true, ...result }
  } catch (error) {
    await queuePendingConversation(pending)
    // The queued payload is self-contained. Releasing only this active-session pointer prevents a
    // later conversation from being mistaken for the failed one while preserving retryability.
    await clearActiveMirrorSession(pending.sessionId)
    return { saved: false, sessionId: pending.sessionId, reason: error instanceof Error ? error.message : 'network_error' }
  }
}
