import { queueAbandon, queuePendingConversation } from '../storage/conversationQueue'
import { getStoredMirrorSessionMetadata } from '../storage/mirrorStorage'
import { getDeviceCredential } from '../storage/deviceCredentials'
import type { ScreeningAssessment } from './assess'
import type { ChatMessage } from '../hooks/conversationTypes'
import type { SessionTelemetry } from '../orchestration/sessionTelemetry'
import { abandonMirrorSessionById, buildPendingSessionCompletion, buildSessionWavBase64, clearActiveMirrorSession, uploadPendingSessionCompletion, type SessionArtifactInput } from './sessionSync'

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
  /** Per-turn + per-session telemetry from the conversation hook (baseline §3). */
  telemetry?: SessionTelemetry | null
  /** Raw patient PCM16 frames for the session audio artifact (transcription + Phase-6 acoustic). */
  sessionAudio?: { base64Frames: string[]; sampleRate: number } | null
}

/** Builds a durable v1 session completion. Client-side assessment is deliberately not clinical evidence. */
export async function buildCheckinPayload(args: CheckinArgs) {
  const artifacts: SessionArtifactInput[] = (args.cameraFrames || []).map((frame) => ({
    kind: 'image' as const,
    contentType: 'image/jpeg' as const,
    dataBase64: frame.replace(/^data:image\/jpeg;base64,/, ''),
  }))
  if (args.sessionAudio?.base64Frames?.length) {
    const wav = buildSessionWavBase64(args.sessionAudio.base64Frames, args.sessionAudio.sampleRate)
    if (wav) artifacts.push({ kind: 'audio', contentType: 'audio/wav', dataBase64: wav })
  }
  return buildPendingSessionCompletion(args.messages, args.startedAt, args.endedAt, args.telemetry ?? null, artifacts)
}

export async function resolveOwnerIds(): Promise<{ nurseId: string; patientId: string; deviceId?: string; language?: string }> {
  const [credential, metadata] = await Promise.all([getDeviceCredential(), getStoredMirrorSessionMetadata()])
  if (!credential) throw new Error('device_not_paired')
  return { nurseId: '', patientId: credential.patientId, deviceId: credential.deviceId, language: metadata.language ?? undefined }
}

/** Uploads transcript events and completes the server session; failures enter the durable local outbox. */
export async function saveCheckin(args: CheckinArgs): Promise<{ saved: boolean; sessionId: string; operationId?: string; reason?: string }> {
  const pending = await buildCheckinPayload(args)
  // A session with no transcript turns cannot be completed (no events); represent it as an abandon so
  // it never becomes a poison payload that retries forever in the offline queue (baseline §7 Phase 7).
  if (pending.events.length === 0) {
    const idempotencyKey = `abandon_${pending.sessionId}`
    try {
      await abandonMirrorSessionById(pending.sessionId, 'user_cancelled', idempotencyKey)
      return { saved: true, sessionId: pending.sessionId }
    } catch (error) {
      await queueAbandon(pending.sessionId, 'user_cancelled', idempotencyKey)
      await clearActiveMirrorSession(pending.sessionId)
      return { saved: false, sessionId: pending.sessionId, reason: error instanceof Error ? error.message : 'network_error' }
    }
  }
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
