import AsyncStorage from '@react-native-async-storage/async-storage'
import { CryptoDigestAlgorithm, digest } from 'expo-crypto'

import { getApiUrl } from '../config/apiUrl'
import { ACTIVE_SESSION_STORAGE_KEY } from '../constants/nursePatientConfig'
import type { ChatMessage } from '../hooks/conversationTypes'
import {
  dailyConversationMetadataForPatientTurn,
  type DailyConversationPlan,
} from '../orchestration/deterministicSpeech'
import { deviceFetch, randomIdempotencyKey } from '../storage/deviceCredentials'
import { dataOrThrow } from './devicePairing'

export type MirrorSessionType = 'companion' | 'daily_checkin' | 'device_test'
export type ActiveMirrorSession = {
  sessionId: string
  stateVersion: number
  type: MirrorSessionType
  language: string
  startedAt: string
  dailyPlan?: DailyConversationPlan
  ticket?: string
  ticketExpiresAt?: string
}

type TranscriptPayload = {
  turnId: string
  role: 'patient' | 'assistant'
  text: string
  protocolStage?: string
  cognitiveSignals?: string[]
  protocolVersion?: string
}

export type PendingSessionCompletion = {
  sessionId: string
  events: Array<{ eventId: string; sequence: number; occurredAt: string; kind: 'transcript_turn'; payload: TranscriptPayload }>
  localCompletedAt: string
  acquisitionSummary: { durationMs: number; patientTurns: number }
  artifacts: LocalSessionArtifact[]
  eventBatchKeys: string[]
  completeKey: string
}

export type SessionArtifactInput = { kind: 'image'; contentType: 'image/jpeg'; dataBase64: string }
export type LocalSessionArtifact = SessionArtifactInput & { clientArtifactId: string }
export type SessionProcessingStatus = {
  sessionId: string
  operationId: string | null
  state: 'accepted' | 'queued' | 'processing' | 'completed' | 'failed'
  stage: string
  retryable: boolean
  result: Record<string, unknown> | null
  updatedAt: string
}

let memorySession: ActiveMirrorSession | null = null

export async function beginMirrorSession(type: MirrorSessionType, language: string, dailyPlan?: DailyConversationPlan) {
  const previous = await getActiveMirrorSession()
  if (previous) {
    const response = await deviceFetch(`/api/v1/sessions/${encodeURIComponent(previous.sessionId)}/abandon`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Idempotency-Key': `abandon_${previous.sessionId}` },
      body: JSON.stringify({ reason: 'client_restart' }),
    })
    if (!response.ok && response.status !== 409) throw new Error('previous_session_abandon_failed')
    await clearActiveMirrorSession(previous.sessionId)
  }
  const response = await deviceFetch('/api/v1/sessions', {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'Idempotency-Key': randomIdempotencyKey() },
    body: JSON.stringify({ type, clientSessionId: randomIdempotencyKey(), requestedLanguage: language,
      clientContext: {
        app: 'mirror',
        transport: process.env.EXPO_PUBLIC_CONVERSATION_MODE || 'relay',
        ...(dailyPlan ? {
          protocolVersion: dailyPlan.protocolVersion,
          reminiscenceEnabled: dailyPlan.includeReminiscence,
          medicationOccurrenceId: dailyPlan.medicationReminder?.occurrenceId ?? null,
        } : {}),
      } }),
  })
  const data = await dataOrThrow<{ sessionId: string; stateVersion: number }>(response)
  memorySession = { sessionId: data.sessionId, stateVersion: data.stateVersion, type, language, startedAt: new Date().toISOString(), dailyPlan }
  await AsyncStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, JSON.stringify(memorySession))
  return memorySession
}

export async function getActiveMirrorSession() {
  if (memorySession) return memorySession
  const raw = await AsyncStorage.getItem(ACTIVE_SESSION_STORAGE_KEY)
  if (!raw) return null
  try {
    memorySession = JSON.parse(raw) as ActiveMirrorSession
    return memorySession
  } catch {
    memorySession = null
    await AsyncStorage.removeItem(ACTIVE_SESSION_STORAGE_KEY)
    return null
  }
}

export async function getActiveQwenTicket() {
  let session = await getActiveMirrorSession()
  if (!session) session = await beginMirrorSession('companion', 'mandarin')
  if (session.ticket && session.ticketExpiresAt && Date.parse(session.ticketExpiresAt) > Date.now() + 60_000) return session.ticket
  const response = await deviceFetch(`/api/v1/sessions/${encodeURIComponent(session.sessionId)}/realtime-tickets`, {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'Idempotency-Key': randomIdempotencyKey() }, body: '{}',
  })
  const ticket = await dataOrThrow<{ ticket: string; expiresAt: string }>(response)
  session = { ...session, ticket: ticket.ticket, ticketExpiresAt: ticket.expiresAt }
  memorySession = session
  await AsyncStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, JSON.stringify(session))
  return ticket.ticket
}

export async function buildPendingSessionCompletion(
  messages: ChatMessage[],
  startedAt: Date,
  endedAt: Date,
  artifacts: SessionArtifactInput[] = [],
): Promise<PendingSessionCompletion> {
  const session = await getActiveMirrorSession()
  if (!session) throw new Error('backend_session_missing')
  const turns = messages.filter((message) => (message.role === 'user' || message.role === 'assistant') && message.text.trim())
  let patientTurn = 0
  const events = turns.map((message, sequence) => {
    const role = message.role === 'user' ? 'patient' as const : 'assistant' as const
    const payload: TranscriptPayload = {
      turnId: message.id || `turn_${sequence}`,
      role,
      text: message.text.trim(),
    }
    if (role === 'patient' && session.type === 'daily_checkin' && session.dailyPlan) {
      patientTurn += 1
      const metadata = dailyConversationMetadataForPatientTurn(patientTurn, session.dailyPlan)
      payload.protocolStage = metadata.protocolStage
      payload.cognitiveSignals = metadata.cognitiveSignals
      payload.protocolVersion = session.dailyPlan.protocolVersion
    }
    return {
      eventId: `turn_${session.sessionId}_${message.id || sequence}`,
      sequence,
      occurredAt: endedAt.toISOString(),
      kind: 'transcript_turn' as const,
      payload,
    }
  })
  const batches = Math.ceil(events.length / 100)
  return {
    sessionId: session.sessionId,
    events,
    localCompletedAt: endedAt.toISOString(),
    acquisitionSummary: { durationMs: Math.max(0, endedAt.getTime() - startedAt.getTime()), patientTurns: events.filter((event) => event.payload.role === 'patient').length },
    artifacts: artifacts.map((artifact, index) => ({ ...artifact, clientArtifactId: `capture_${session.sessionId}_${index}` })),
    eventBatchKeys: Array.from({ length: batches }, () => randomIdempotencyKey()),
    completeKey: randomIdempotencyKey(),
  }
}

export async function uploadPendingSessionCompletion(pending: PendingSessionCompletion) {
  if (!pending.events.length) throw new Error('session_has_no_transcript_events')
  for (let offset = 0, batchIndex = 0; offset < pending.events.length; offset += 100, batchIndex++) {
    const response = await deviceFetch(`/api/v1/sessions/${encodeURIComponent(pending.sessionId)}/event-batches`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'Idempotency-Key': pending.eventBatchKeys[batchIndex] },
      body: JSON.stringify({ events: pending.events.slice(offset, offset + 100) }),
    })
    await dataOrThrow(response)
  }
  const artifactIds = await uploadSessionArtifacts(pending.sessionId, pending.artifacts || [])
  const stateResponse = await deviceFetch(`/api/v1/sessions/${encodeURIComponent(pending.sessionId)}`)
  const state = await dataOrThrow<{ stateVersion: number }>(stateResponse)
  const completeResponse = await deviceFetch(`/api/v1/sessions/${encodeURIComponent(pending.sessionId)}/complete`, {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'Idempotency-Key': pending.completeKey, 'If-Match': String(state.stateVersion) },
    body: JSON.stringify({
      localCompletedAt: pending.localCompletedAt,
      finalSequence: pending.events.length - 1,
      artifactIds,
      acquisitionSummary: pending.acquisitionSummary,
    }),
  })
  const completion = await dataOrThrow<{ operationId: string; state: 'queued' }>(completeResponse)
  await clearActiveMirrorSession(pending.sessionId)
  return { sessionId: pending.sessionId, operationId: completion.operationId }
}

export async function getSessionProcessingStatus(sessionId: string) {
  const response = await deviceFetch(`/api/v1/sessions/${encodeURIComponent(sessionId)}/processing-status`)
  return dataOrThrow<SessionProcessingStatus>(response)
}

async function uploadSessionArtifacts(sessionId: string, artifacts: LocalSessionArtifact[]) {
  if (!artifacts.length) return []
  const prepared = await Promise.all(artifacts.map(async (artifact) => {
    const bytes = decodeBase64(artifact.dataBase64)
    const hash = bytesToHex(await digest(CryptoDigestAlgorithm.SHA256, bytes))
    return { artifact, bytes, hash }
  }))
  const planResponse = await deviceFetch(`/api/v1/sessions/${encodeURIComponent(sessionId)}/artifact-upload-plans`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Idempotency-Key': randomIdempotencyKey() },
    body: JSON.stringify({ artifacts: prepared.map(({ artifact, bytes, hash }) => ({
      clientArtifactId: artifact.clientArtifactId,
      kind: artifact.kind,
      contentType: artifact.contentType,
      sizeBytes: bytes.byteLength,
      hash,
    })) }),
  })
  const plans = await dataOrThrow<Array<{
    artifactId: string
    uploadUrl?: string
    requiredHeaders?: Record<string, string>
    alreadyUploaded?: boolean
  }>>(planResponse)
  if (plans.length !== prepared.length) throw new Error('artifact_upload_plan_incomplete')
  const commitItems: Array<{ artifactId: string; hash: string; sizeBytes: number }> = []
  for (let index = 0; index < plans.length; index += 1) {
    const plan = plans[index]
    const item = prepared[index]
    if (!plan.alreadyUploaded) {
      if (!plan.uploadUrl || !isTrustedUploadUrl(plan.uploadUrl)) throw new Error('artifact_upload_url_invalid')
      const uploadResponse = await fetch(plan.uploadUrl, {
        method: 'PUT',
        headers: plan.requiredHeaders,
        body: item.bytes.buffer as ArrayBuffer,
      })
      if (!uploadResponse.ok) throw new Error(`artifact_upload_failed_${uploadResponse.status}`)
      commitItems.push({ artifactId: plan.artifactId, hash: item.hash, sizeBytes: item.bytes.byteLength })
    }
  }
  if (commitItems.length) {
    const commitResponse = await deviceFetch(`/api/v1/sessions/${encodeURIComponent(sessionId)}/artifacts/commit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Idempotency-Key': randomIdempotencyKey() },
      body: JSON.stringify({ artifacts: commitItems }),
    })
    await dataOrThrow(commitResponse)
  }
  return plans.map((plan) => plan.artifactId)
}

function decodeBase64(value: string) {
  const binary = globalThis.atob(value)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index)
  return bytes
}

function bytesToHex(value: ArrayBuffer) {
  return Array.from(new Uint8Array(value), (byte) => byte.toString(16).padStart(2, '0')).join('')
}

function isTrustedUploadUrl(value: string) {
  try {
    const parsed = new URL(value)
    return parsed.protocol === 'https:' || (__DEV__ && parsed.protocol === 'http:')
  } catch {
    return false
  }
}

export async function clearActiveMirrorSession(expectedSessionId?: string) {
  const current = await getActiveMirrorSession()
  if (expectedSessionId && current?.sessionId !== expectedSessionId) return
  memorySession = null
  await AsyncStorage.removeItem(ACTIVE_SESSION_STORAGE_KEY)
}
