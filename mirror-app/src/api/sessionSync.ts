import AsyncStorage from '@react-native-async-storage/async-storage'
import Constants from 'expo-constants'
import { CryptoDigestAlgorithm, digest } from 'expo-crypto'

import { getApiUrl } from '../config/apiUrl'
import { ACTIVE_SESSION_STORAGE_KEY } from '../constants/nursePatientConfig'
import type { ChatMessage } from '../hooks/conversationTypes'
import {
  dailyConversationMetadataForPatientTurn,
  type DailyConversationPlan,
} from '../orchestration/deterministicSpeech'
import type { SessionTelemetry, TurnTelemetry } from '../orchestration/sessionTelemetry'
import { deviceFetch, randomIdempotencyKey } from '../storage/deviceCredentials'
import { dataOrThrow } from './devicePairing'

const APP_VERSION = String((Constants.expoConfig as { version?: string } | null)?.version || process.env.EXPO_PUBLIC_APP_VERSION || 'unknown')

// CJK-aware word count: whitespace splitting massively undercounts Mandarin (the default language).
// Prefer Intl.Segmenter word granularity where the runtime provides it; otherwise count CJK codepoints
// individually and whitespace-split the rest.
function countWords(text: string): number {
  const trimmed = text.trim()
  if (!trimmed) return 0
  const Segmenter = (Intl as unknown as { Segmenter?: typeof Intl.Segmenter }).Segmenter
  if (Segmenter) {
    try {
      const segmenter = new Segmenter(undefined, { granularity: 'word' })
      return Array.from(segmenter.segment(trimmed)).filter((part) => (part as { isWordLike?: boolean }).isWordLike).length
    } catch {
      // fall through to the manual estimate
    }
  }
  const cjk = (trimmed.match(/[㐀-鿿぀-ヿ가-힯]/g) || []).length
  const nonCjk = trimmed.replace(/[㐀-鿿぀-ヿ가-힯]/g, ' ').trim()
  const words = nonCjk ? nonCjk.split(/\s+/).filter(Boolean).length : 0
  return cjk + words
}

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
  startedAt?: string
  endedAt?: string
  protocolStage?: string
  cognitiveSignals?: string[]
  protocolVersion?: string
}

type CaptureMetricPayload = {
  metric: 'turn_timing'
  turnId: string
  questionId?: string
  ariaPromptEndAt?: string
  userSpeechStartAt?: string
  responseLatencyMs?: number | null
  userSpeechMs?: number
  repromptCount: number
}

type SessionEvent =
  | { eventId: string; sequence: number; occurredAt: string; kind: 'transcript_turn'; payload: TranscriptPayload }
  | { eventId: string; sequence: number; occurredAt: string; kind: 'capture_metric'; payload: CaptureMetricPayload }

export type AcquisitionSummary = {
  durationMs: number
  patientTurns: number
  patientSpeechMs?: number
  ariaSpeechMs?: number
  ariaTurns?: number
  repromptCount?: number
  wordCount?: number
  transcriptAvailable?: boolean
  medianResponseLatencyMs?: number | null
  sessionStartMinuteOfDay?: number
  sessionStatus?: 'completed' | 'incomplete' | 'technical_error'
  technicalError?: boolean
  technicalErrorType?: string | null
  timezone?: string
  appVersion?: string
}

export type PendingSessionCompletion = {
  sessionId: string
  events: SessionEvent[]
  localCompletedAt: string
  acquisitionSummary: AcquisitionSummary
  artifacts: LocalSessionArtifact[]
  eventBatchKeys: string[]
  completeKey: string
}

export type SessionArtifactInput =
  | { kind: 'image'; contentType: 'image/jpeg'; dataBase64: string }
  | { kind: 'audio'; contentType: 'audio/wav'; dataBase64: string }
export type LocalSessionArtifact = SessionArtifactInput & { clientArtifactId: string }

// Assemble captured PCM16 mono frames into a single WAV (16-bit) for the session audio artifact.
export function buildSessionWavBase64(base64Frames: string[], sampleRate: number): string | null {
  if (!base64Frames.length) return null
  const parts = base64Frames.map((frame) => decodeBase64(frame))
  const dataLength = parts.reduce((total, part) => total + part.byteLength, 0)
  if (dataLength === 0) return null
  const out = new Uint8Array(44 + dataLength)
  const view = new DataView(out.buffer)
  const writeAscii = (offset: number, text: string) => { for (let i = 0; i < text.length; i += 1) out[offset + i] = text.charCodeAt(i) }
  writeAscii(0, 'RIFF'); view.setUint32(4, 36 + dataLength, true); writeAscii(8, 'WAVE')
  writeAscii(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true) // byteRate
  view.setUint16(32, 2, true); view.setUint16(34, 16, true) // blockAlign, bitsPerSample
  writeAscii(36, 'data'); view.setUint32(40, dataLength, true)
  let offset = 44
  for (const part of parts) { out.set(part, offset); offset += part.byteLength }
  return encodeBase64(out)
}

function encodeBase64(bytes: Uint8Array): string {
  const encoder = (globalThis as unknown as { btoa?: (value: string) => string }).btoa
  if (!encoder) throw new Error('base64 encoding unavailable')
  let binary = ''
  const CHUNK = 0x8000 // avoid String.fromCharCode call-stack limits on large buffers
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK))
  }
  return encoder(binary)
}
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

function resolveTimezone(): string {
  try { return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC' } catch { return 'UTC' }
}

export async function buildPendingSessionCompletion(
  messages: ChatMessage[],
  startedAt: Date,
  endedAt: Date,
  telemetry: SessionTelemetry | null = null,
  artifacts: SessionArtifactInput[] = [],
): Promise<PendingSessionCompletion> {
  const session = await getActiveMirrorSession()
  if (!session) throw new Error('backend_session_missing')
  const turns = messages.filter((message) => (message.role === 'user' || message.role === 'assistant') && message.text.trim())

  const patientTelemetry = telemetry?.turns.filter((turn) => turn.role === 'patient') ?? []
  const assistantTelemetry = telemetry?.turns.filter((turn) => turn.role === 'assistant') ?? []

  let patientIndex = 0
  let assistantIndex = 0
  let userWordText = ''
  const events: SessionEvent[] = []

  // Transcript turns carry REAL per-utterance start/end (baseline §3.2); occurredAt is the turn's own
  // end, no longer collapsed to session end, so offline-queued sessions land in the correct local day.
  turns.forEach((message, sequence) => {
    const role = message.role === 'user' ? 'patient' as const : 'assistant' as const
    const timing: TurnTelemetry | undefined = role === 'patient' ? patientTelemetry[patientIndex] : assistantTelemetry[assistantIndex]
    const payload: TranscriptPayload = {
      turnId: message.id || `turn_${sequence}`,
      role,
      text: message.text.trim(),
      startedAt: timing?.startedAt,
      endedAt: timing?.endedAt,
    }
    if (role === 'patient') {
      userWordText += (userWordText ? ' ' : '') + message.text.trim()
      patientIndex += 1
      if (session.type === 'daily_checkin' && session.dailyPlan) {
        const metadata = dailyConversationMetadataForPatientTurn(patientIndex, session.dailyPlan)
        payload.protocolStage = metadata.protocolStage
        payload.cognitiveSignals = metadata.cognitiveSignals
        payload.protocolVersion = session.dailyPlan.protocolVersion
      }
    } else {
      assistantIndex += 1
    }
    events.push({
      eventId: `turn_${session.sessionId}_${message.id || sequence}`,
      sequence,
      occurredAt: timing?.endedAt || endedAt.toISOString(),
      kind: 'transcript_turn',
      payload,
    })
  })

  // Per-turn timing metrics (M7/M13 latency signal) as capture_metric events, sequenced after transcripts.
  patientTelemetry.forEach((turn, index) => {
    const metadata = session.type === 'daily_checkin' && session.dailyPlan
      ? dailyConversationMetadataForPatientTurn(index + 1, session.dailyPlan) : null
    events.push({
      eventId: `metric_${session.sessionId}_${turn.turnId}`,
      sequence: events.length,
      occurredAt: turn.endedAt,
      kind: 'capture_metric',
      payload: {
        metric: 'turn_timing',
        turnId: turn.turnId,
        questionId: metadata?.protocolStage ?? turn.questionId,
        ariaPromptEndAt: turn.ariaPromptEndAt,
        userSpeechStartAt: turn.userSpeechStartAt,
        responseLatencyMs: turn.responseLatencyMs,
        userSpeechMs: turn.speechMs,
        repromptCount: telemetry?.repromptCount ?? 0,
      },
    })
  })

  const patientTurns = turns.filter((message) => message.role === 'user').length
  const durationMs = Math.max(0, endedAt.getTime() - startedAt.getTime())
  // Backend decides M1 completion; sessionStatus is advisory context only (baseline §3.1).
  const completed = patientTurns >= 3 && (telemetry?.patientSpeechMs ?? 0) >= 30_000
  const acquisitionSummary: AcquisitionSummary = {
    durationMs,
    patientTurns,
    patientSpeechMs: telemetry?.patientSpeechMs,
    ariaSpeechMs: telemetry?.ariaSpeechMs,
    ariaTurns: telemetry?.ariaTurns ?? turns.filter((message) => message.role === 'assistant').length,
    repromptCount: telemetry?.repromptCount,
    wordCount: countWords(userWordText),
    transcriptAvailable: userWordText.length > 0,
    medianResponseLatencyMs: telemetry?.medianResponseLatencyMs ?? null,
    sessionStartMinuteOfDay: startedAt.getHours() * 60 + startedAt.getMinutes(),
    sessionStatus: completed ? 'completed' : 'incomplete',
    technicalError: false,
    technicalErrorType: null,
    timezone: resolveTimezone(),
    appVersion: APP_VERSION,
  }

  const batches = Math.max(1, Math.ceil(events.length / 100))
  return {
    sessionId: session.sessionId,
    events,
    localCompletedAt: endedAt.toISOString(),
    acquisitionSummary,
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

// Abandon a session that produced no usable transcript (or that repeatedly failed to upload) instead
// of queuing an uncompletable completion. Idempotent; a 409 (already abandoned/completed) is fine.
export async function abandonMirrorSessionById(sessionId: string, reason: string, idempotencyKey?: string) {
  const response = await deviceFetch(`/api/v1/sessions/${encodeURIComponent(sessionId)}/abandon`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Idempotency-Key': idempotencyKey || `abandon_${sessionId}` },
    body: JSON.stringify({ reason }),
  })
  if (!response.ok && response.status !== 409) throw new Error(`session_abandon_failed_${response.status}`)
  await clearActiveMirrorSession(sessionId)
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
