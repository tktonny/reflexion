import { Router, type Request } from 'express'
import { MongoServerError } from 'mongodb'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { getDb, inTransaction } from '../../lib/mongo.js'
import { authorizePatient, getPrincipal, requireActor } from '../platform/auth.js'
import { collections } from '../platform/collections.js'
import { openSecret, sealSecret } from '../platform/crypto.js'
import { ApiError, badRequest, conflict, forbidden, notFound } from '../platform/errors.js'
import { sendData } from '../platform/http.js'
import { newId } from '../platform/ids.js'
import { executeIdempotent, type IdempotencyCodec } from '../platform/idempotency.js'
import { getObjectStore } from '../platform/objectStore.js'
import { appendOutbox } from '../platform/outbox.js'
import { createQwenRealtimeTicket } from '../platform/qwen.js'
import { enumValue, isoDate, objectBody, optionalString, positiveInteger, requiredString, stringArray } from '../platform/validation.js'

const SESSION_TYPES = ['companion', 'daily_checkin', 'clinic_assessment', 'device_test'] as const
const EVENT_KINDS = ['transcript_turn', 'tool_call', 'tool_result', 'capture_metric', 'lifecycle', 'user_correction'] as const
const ARTIFACT_KINDS = ['audio', 'video', 'transcript', 'image', 'sidecar'] as const
const DAILY_PROTOCOL_STAGES = new Set(['warm_up', 'yesterday_recall', 'present_planning', 'medication_reminder', 'reminiscence'])
const DAILY_COGNITIVE_SIGNALS = new Set([
  'mood', 'speech_initiation', 'response_latency', 'episodic_memory', 'temporal_orientation',
  'narrative_coherence', 'executive_function', 'prospective_memory', 'social_connectedness',
  'memory', 'caregiver_adjunct', 'semantic_memory', 'language_richness', 'lexical_diversity', 'speech_fluency',
])

export const sessionsRouter = Router()
const requireSessionActor = requireActor('human', 'device')

sessionsRouter.post('/sessions', requireSessionActor, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/sessions', async () => {
    const body = objectBody(request.body)
    const principal = getPrincipal(request)
    const type = enumValue(body.type, 'type', SESSION_TYPES)
    const patientId = principal.kind === 'device'
      ? principal.patientId
      : requiredString(body, 'patientId', 100)
    const patient = await authorizePatient(request, patientId, principal.kind === 'device' ? 'session:write' : 'patient:write')
    const consent = await consentForSession(principal.tenantId, patientId, type)
    const clientSessionId = optionalString(body, 'clientSessionId', 150)
    const requestedLanguage = optionalString(body, 'requestedLanguage', 40) || String(patient.preferredLanguage || 'zh-CN')
    const sessionId = newId('ses')
    const now = new Date()
    const sessionDocument = {
      _id: sessionId, tenantId: principal.tenantId, patientId,
      deviceId: principal.kind === 'device' ? principal.deviceId : null,
      clientSessionId, type, state: 'created', stateVersion: 1,
      protocolContext: await resolveProtocolContext(type),
      consentRef: consent ? { consentId: consent._id, purpose: consent.purpose, version: consent.documentVersion } : null,
      acquisition: { language: requestedLanguage, timezone: patient.timezone, clientContext: body.clientContext || {} },
      artifactCounts: {}, latestProcessingRevision: 0,
      createdBy: { type: principal.kind, id: principal.subjectId }, createdAt: now, updatedAt: now,
    }
    await inTransaction(async (db, transaction) => {
      await db.collection<any>(collections.sessions).insertOne(sessionDocument, { session: transaction })
      await appendOutbox(db, { eventType: 'session.created', tenantId: principal.tenantId, patientId,
        aggregateType: 'session', aggregateId: sessionId, correlationId: request.requestId,
        payload: { type, deviceId: sessionDocument.deviceId } }, transaction)
    })
    return { status: 201, data: serializeSession(sessionDocument) }
  })
  sendData(response, result.data, result.status)
}))

sessionsRouter.get('/sessions/:sessionId', requireSessionActor, asyncHandler(async (request, response) => {
  const session = await authorizedSession(request, request.params.sessionId, 'session:read')
  sendData(response, serializeSession(session))
}))

sessionsRouter.get('/sessions/:sessionId/processing-status', requireSessionActor, asyncHandler(async (request, response) => {
  const session = await authorizedSession(request, request.params.sessionId, 'session:read')
  response.setHeader('Cache-Control', 'no-store')
  sendData(response, serializeProcessingStatus(session))
}))

sessionsRouter.post('/sessions/:sessionId/realtime-tickets', requireSessionActor, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/sessions/:sessionId/realtime-tickets', async () => {
    const principal = getPrincipal(request)
    if (principal.kind !== 'device') throw forbidden('Only an assigned mirror may request a realtime ticket.')
    const session = await authorizedSession(request, request.params.sessionId, 'session:write')
    if (!['created', 'active'].includes(String(session.state))) throw conflict('SESSION_NOT_ACTIVE', 'Realtime access is not available for this session.')
    const ticket = await createQwenRealtimeTicket(String(session.acquisition?.language || 'zh-CN'))
    const db = await getDb()
    if (session.state === 'created') {
      await db.collection<any>(collections.sessions).updateOne({ _id: session._id, state: 'created' }, {
        $set: { state: 'active', activatedAt: new Date(), updatedAt: new Date() }, $inc: { stateVersion: 1 },
      })
    }
    return { status: 201, data: ticket }
  }, undefined, sealedJsonCodec)
  response.setHeader('Cache-Control', 'no-store')
  sendData(response, result.data, result.status)
}))

sessionsRouter.post('/sessions/:sessionId/event-batches', requireSessionActor, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/sessions/:sessionId/event-batches', async () => {
    const session = await authorizedSession(request, request.params.sessionId, 'session:write')
    if (!['created', 'active'].includes(String(session.state))) throw conflict('SESSION_NOT_WRITABLE', 'This session no longer accepts events.')
    const body = objectBody(request.body)
    if (!Array.isArray(body.events) || body.events.length < 1 || body.events.length > 100) {
      throw badRequest('VALIDATION_FAILED', 'events must contain between 1 and 100 items.')
    }
    const db = await getDb()
    let accepted = 0
    let duplicates = 0
    for (const raw of body.events) {
      const event = validateEvent(raw)
      const existing = await db.collection<any>(collections.sessionEvents).findOne({
        sessionId: session._id, $or: [{ eventId: event.eventId }, { sequence: event.sequence }],
      })
      if (existing) {
        if (existing.eventId === event.eventId && existing.sequence === event.sequence) { duplicates++; continue }
        throw conflict('EVENT_SEQUENCE_CONFLICT', `Sequence ${event.sequence} is already assigned to another event.`)
      }
      try {
        await db.collection<any>(collections.sessionEvents).insertOne({
          ...event, tenantId: session.tenantId, patientId: session.patientId, sessionId: session._id, receivedAt: new Date(),
        })
        accepted++
        if (event.kind === 'transcript_turn') await materializeTranscriptTurn(session, event)
      } catch (error) {
        if (!(error instanceof MongoServerError) || error.code !== 11000) throw error
        duplicates++
      }
    }
    await db.collection<any>(collections.sessions).updateOne({ _id: session._id, state: 'created' }, {
      $set: { state: 'active', activatedAt: new Date(), updatedAt: new Date() }, $inc: { stateVersion: 1 },
    })
    return { status: 202, data: { accepted, duplicates, nextExpectedSequence: await nextExpectedSequence(String(session._id)) } }
  })
  sendData(response, result.data, result.status)
}))

sessionsRouter.post('/sessions/:sessionId/artifact-upload-plans', requireSessionActor, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/sessions/:sessionId/artifact-upload-plans', async () => {
    const session = await authorizedSession(request, request.params.sessionId, 'session:write')
    if (!['created', 'active'].includes(String(session.state))) throw conflict('SESSION_NOT_WRITABLE', 'This session no longer accepts artifacts.')
    const body = objectBody(request.body)
    if (!Array.isArray(body.artifacts) || body.artifacts.length < 1 || body.artifacts.length > 20) {
      throw badRequest('VALIDATION_FAILED', 'artifacts must contain between 1 and 20 items.')
    }
    const store = getObjectStore()
    const db = await getDb()
    const prepared = []
    for (const raw of body.artifacts) {
      const intent = validateArtifactIntent(raw)
      const existing = await db.collection<any>(collections.artifacts).findOne({
        sessionId: session._id, clientArtifactId: intent.clientArtifactId,
      })
      if (existing && ['verification_pending', 'verified'].includes(String(existing.state))) {
        if (existing.kind !== intent.kind || existing.contentType !== intent.contentType ||
          existing.hash !== intent.hash || Number(existing.sizeBytes) !== intent.sizeBytes) {
          throw conflict('CLIENT_ARTIFACT_ID_REUSED', `Artifact ${intent.clientArtifactId} was already used with different content.`)
        }
        prepared.push({ artifactId: String(existing._id), alreadyUploaded: true })
        continue
      }
      const artifactId = String(existing?._id || newId('art'))
      const objectKey = `${session.tenantId}/${session.patientId}/${session._id}/${artifactId}-${intent.kind}`
      const plan = await store.prepareUpload({ objectKey, contentType: intent.contentType, hash: intent.hash })
      await db.collection<any>(collections.artifacts).updateOne({ _id: artifactId }, { $set: {
        tenantId: session.tenantId, patientId: session.patientId, sessionId: session._id, clientArtifactId: intent.clientArtifactId,
        kind: intent.kind, contentType: intent.contentType, objectKey, hash: intent.hash, sizeBytes: intent.sizeBytes,
        encryption: 'object_store_managed', retentionClass: retentionClass(intent.kind), state: 'upload_pending',
        uploadExpiresAt: plan.expiresAt, updatedAt: new Date(),
      }, $setOnInsert: { createdAt: new Date() } }, { upsert: true })
      prepared.push({ artifactId, uploadUrl: plan.uploadUrl, expiresAt: plan.expiresAt.toISOString(), requiredHeaders: plan.requiredHeaders })
    }
    return { status: 200, data: prepared }
  }, undefined, sealedJsonCodec)
  response.setHeader('Cache-Control', 'no-store')
  sendData(response, result.data, result.status)
}))

sessionsRouter.post('/sessions/:sessionId/artifacts/commit', requireSessionActor, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/sessions/:sessionId/artifacts/commit', async () => {
    const session = await authorizedSession(request, request.params.sessionId, 'session:write')
    const body = objectBody(request.body)
    if (!Array.isArray(body.artifacts) || !body.artifacts.length) throw badRequest('VALIDATION_FAILED', 'artifacts is required.')
    const operationId = newId('op')
    const db = await getDb()
    for (const raw of body.artifacts) {
      const item = objectBody(raw)
      const artifactId = requiredString(item, 'artifactId', 100)
      const hash = requiredString(item, 'hash', 200)
      const sizeBytes = positiveInteger(item.sizeBytes, 'sizeBytes')
      const artifact = await db.collection<any>(collections.artifacts).findOne({ _id: artifactId, sessionId: session._id })
      if (!artifact || artifact.hash !== hash || artifact.sizeBytes !== sizeBytes) {
        throw conflict('ARTIFACT_MISMATCH', `Artifact ${artifactId} does not match its upload intent.`)
      }
      await db.collection<any>(collections.artifacts).updateOne({ _id: artifactId, state: 'upload_pending' }, { $set: {
        state: 'verification_pending', committedAt: new Date(), operationId, updatedAt: new Date(),
      } })
      await appendOutbox(db, { eventType: 'artifact.committed', tenantId: String(session.tenantId), patientId: String(session.patientId),
        aggregateType: 'artifact', aggregateId: artifactId, correlationId: request.requestId,
        payload: { sessionId: session._id } })
    }
    return { status: 202, data: { operationId, state: 'queued' } }
  })
  sendData(response, result.data, result.status)
}))

sessionsRouter.post('/sessions/:sessionId/complete', requireSessionActor, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/sessions/:sessionId/complete', async () => {
    const session = await authorizedSession(request, request.params.sessionId, 'session:write')
    const expectedVersion = parseIfMatch(request.header('If-Match'))
    if (Number(session.stateVersion) !== expectedVersion) throw conflict('VERSION_CONFLICT', 'The session state changed. Refresh and retry.')
    if (!['created', 'active'].includes(String(session.state))) throw conflict('SESSION_ALREADY_COMPLETED', 'This session is already complete or processing.')
    const body = objectBody(request.body)
    const localCompletedAt = isoDate(body.localCompletedAt, 'localCompletedAt')
    const finalSequence = positiveInteger(body.finalSequence, 'finalSequence', true)
    const artifactIds = stringArray(body.artifactIds, 'artifactIds', 50)
    const db = await getDb()
    const eventCount = await db.collection<any>(collections.sessionEvents).countDocuments({
      sessionId: session._id, sequence: { $gte: 0, $lte: finalSequence },
    })
    if (eventCount !== finalSequence + 1) throw conflict('EVENT_GAP', 'Session events are incomplete. Upload missing events before completion.', true)
    if (artifactIds.length) {
      const artifactCount = await db.collection<any>(collections.artifacts).countDocuments({
        _id: { $in: artifactIds }, sessionId: session._id, state: { $in: ['verification_pending', 'verified'] },
      })
      if (artifactCount !== artifactIds.length) throw conflict('ARTIFACTS_NOT_COMMITTED', 'All declared artifacts must be committed first.', true)
    }
    const acquisitionSummary = validateAcquisitionSummary(body.acquisitionSummary)
    const operationId = newId('op')
    const acquisitionUpdate = Object.fromEntries(Object.entries(acquisitionSummary).map(([key, value]) => [`acquisition.${key}`, value]))
    await inTransaction(async (transactionDb, transaction) => {
      const changed = await transactionDb.collection<any>(collections.sessions).updateOne({
        _id: session._id, stateVersion: expectedVersion, state: { $in: ['created', 'active'] },
      }, { $set: {
        state: 'ingesting', localCompletedAt, finalSequence, artifactIds, operationId,
        ...acquisitionUpdate,
        updatedAt: new Date(),
      }, $inc: { stateVersion: 1 } }, { session: transaction })
      if (!changed.modifiedCount) throw conflict('VERSION_CONFLICT', 'The session state changed. Refresh and retry.')
      await appendOutbox(transactionDb, { eventType: 'session.completed', tenantId: String(session.tenantId), patientId: String(session.patientId),
        aggregateType: 'session', aggregateId: String(session._id), correlationId: request.requestId,
        payload: { operationId, finalSequence, artifactIds } }, transaction)
    })
    return { status: 202, data: { operationId, state: 'queued' } }
  })
  sendData(response, result.data, result.status)
}))

sessionsRouter.post('/sessions/:sessionId/upload-retries', requireSessionActor, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/sessions/:sessionId/upload-retries', async () => {
    const session = await authorizedSession(request, request.params.sessionId, 'session:write')
    const body = objectBody(request.body)
    const artifactIds = stringArray(body.artifactIds, 'artifactIds', 20)
    const db = await getDb()
    const artifacts = await db.collection<any>(collections.artifacts).find({
      _id: { $in: artifactIds }, sessionId: session._id, state: { $in: ['failed', 'upload_pending'] },
    }).toArray()
    if (artifacts.length !== artifactIds.length) throw conflict('ARTIFACT_RETRY_INVALID', 'One or more artifacts cannot be retried.')
    const store = getObjectStore()
    const plans = []
    for (const artifact of artifacts) {
      const plan = await store.prepareUpload({ objectKey: String(artifact.objectKey), contentType: String(artifact.contentType), hash: String(artifact.hash) })
      await db.collection<any>(collections.artifacts).updateOne({ _id: artifact._id }, { $set: { state: 'upload_pending', uploadExpiresAt: plan.expiresAt, updatedAt: new Date() } })
      plans.push({ artifactId: artifact._id, uploadUrl: plan.uploadUrl, expiresAt: plan.expiresAt.toISOString(), requiredHeaders: plan.requiredHeaders })
    }
    return { status: 200, data: plans }
  }, undefined, sealedJsonCodec)
  response.setHeader('Cache-Control', 'no-store')
  sendData(response, result.data, result.status)
}))

sessionsRouter.post('/sessions/:sessionId/abandon', requireSessionActor, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/sessions/:sessionId/abandon', async () => {
    const session = await authorizedSession(request, request.params.sessionId, 'session:write')
    if (session.state === 'abandoned') return { status: 200, data: serializeSession(session) }
    if (!['created', 'active'].includes(String(session.state))) {
      throw conflict('SESSION_NOT_ABANDONABLE', 'Only a created or active session can be abandoned.')
    }
    const body = objectBody(request.body)
    const reason = enumValue(body.reason, 'reason', ['client_restart', 'user_cancelled', 'transport_failed', 'superseded'] as const)
    const db = await getDb()
    const now = new Date()
    const changed = await db.collection<any>(collections.sessions).findOneAndUpdate({
      _id: session._id, tenantId: session.tenantId, state: { $in: ['created', 'active'] },
    }, { $set: { state: 'abandoned', abandonedAt: now, abandonReason: reason, updatedAt: now }, $inc: { stateVersion: 1 } }, { returnDocument: 'after' })
    if (!changed) throw conflict('SESSION_STATE_CHANGED', 'The session state changed before it could be abandoned.')
    await appendOutbox(db, { eventType: 'session.abandoned', tenantId: String(session.tenantId), patientId: String(session.patientId),
      aggregateType: 'session', aggregateId: String(session._id), correlationId: request.requestId, payload: { reason } })
    return { status: 200, data: serializeSession(changed) }
  })
  sendData(response, result.data, result.status)
}))

export async function authorizedSession(request: Request, sessionId: string, scope: string) {
  const principal = getPrincipal(request)
  const db = await getDb()
  const session = await db.collection<any>(collections.sessions).findOne({ _id: sessionId, tenantId: principal.tenantId })
  if (!session) throw notFound('Session')
  if (principal.kind === 'device') {
    if (session.deviceId !== principal.deviceId || session.patientId !== principal.patientId || !principal.scopes.includes(scope)) throw forbidden()
  } else {
    await authorizePatient(request, String(session.patientId), scope === 'session:read' ? 'patient:read' : 'patient:write')
  }
  return session
}

export function serializeSession(session: Record<string, any>) {
  return {
    sessionId: session._id, patientId: session.patientId, deviceId: session.deviceId || null, type: session.type,
    state: session.state, stateVersion: Number(session.stateVersion), protocolContext: session.protocolContext,
    createdAt: new Date(session.createdAt).toISOString(), operationId: session.operationId,
    processingSummary: session.processingSummary,
  }
}

export function serializeProcessingStatus(session: Record<string, any>) {
  const state = String(session.state)
  const summary = session.processingSummary && typeof session.processingSummary === 'object'
    ? session.processingSummary as Record<string, unknown>
    : {}
  const publicState = ['completed', 'excluded', 'review_pending'].includes(state) ? 'completed'
    : state === 'processing_failed' || state === 'abandoned' ? 'failed'
      : state === 'processing' ? 'processing'
        : state === 'ingesting' ? 'queued'
          : 'accepted'
  return {
    sessionId: session._id,
    operationId: session.operationId || null,
    state: publicState,
    stage: typeof summary.stage === 'string' ? summary.stage
      : publicState === 'queued' ? 'queued'
        : publicState === 'accepted' ? 'session_open'
          : publicState === 'failed' ? 'failed'
            : 'complete',
    retryable: typeof summary.retryable === 'boolean' ? summary.retryable : state === 'processing_failed',
    result: publicState === 'completed' ? {
      outcome: state,
      inclusion: summary.inclusion,
      monitoringUse: summary.monitoringUse,
      anomalyState: summary.anomalyState,
    } : null,
    updatedAt: new Date(session.updatedAt || session.createdAt).toISOString(),
  }
}

async function consentForSession(tenantId: string, patientId: string, type: typeof SESSION_TYPES[number]) {
  if (type === 'companion' || type === 'device_test') return null
  const db = await getDb()
  const consent = await db.collection<any>(collections.consents).findOne({
    tenantId, patientId, purpose: 'home_cognitive_monitoring', status: 'granted', withdrawnAt: null,
  }, { sort: { signedAt: -1 } })
  if (!consent) throw new ApiError(403, 'CONSENT_REQUIRED', 'Active home monitoring consent is required for this session type.')
  return consent
}

async function resolveProtocolContext(type: typeof SESSION_TYPES[number]) {
  const db = await getDb()
  const registered = await db.collection<any>(collections.protocolRegistry).findOne({ sessionType: type, status: 'active' }, { sort: { activatedAt: -1 } })
  return registered ? {
    protocolId: registered.protocolId || type,
    protocolVersion: registered.protocolVersion,
    promptVersion: registered.promptVersion,
    toolPolicyVersion: registered.toolPolicyVersion,
    featureSchemaVersion: registered.featureSchemaVersion,
  } : {
    protocolId: type,
    protocolVersion: type === 'daily_checkin' ? 'daily-conversation-v2' : process.env.PROTOCOL_VERSION || 'mirror-v1',
    promptVersion: type === 'daily_checkin' ? 'aria-daily-v2' : process.env.PROMPT_VERSION || 'mirror-zh-v1',
    toolPolicyVersion: process.env.TOOL_POLICY_VERSION || 'home-safe-v1',
    featureSchemaVersion: process.env.FEATURE_SCHEMA_VERSION || 'home-features-v1',
  }
}

function validateEvent(value: unknown) {
  const body = objectBody(value)
  const eventId = requiredString(body, 'eventId', 150)
  const sequence = positiveInteger(body.sequence, 'sequence', true)
  const occurredAt = isoDate(body.occurredAt, 'occurredAt')
  const kind = enumValue(body.kind, 'kind', EVENT_KINDS)
  const payload = body.payload && typeof body.payload === 'object' && !Array.isArray(body.payload) ? body.payload as Record<string, unknown> : {}
  return { eventId, sequence, occurredAt, kind, payload }
}

async function materializeTranscriptTurn(session: Record<string, any>, event: ReturnType<typeof validateEvent>) {
  const payload = event.payload
  if (typeof payload.text !== 'string' || !payload.text.trim()) return
  const turnId = typeof payload.turnId === 'string' ? payload.turnId : event.eventId
  const role = ['patient', 'assistant', 'caregiver', 'system'].includes(String(payload.role)) ? String(payload.role) : 'patient'
  const startedAt = typeof payload.startedAt === 'string' && !Number.isNaN(Date.parse(payload.startedAt)) ? new Date(payload.startedAt) : event.occurredAt
  const endedAt = typeof payload.endedAt === 'string' && !Number.isNaN(Date.parse(payload.endedAt)) ? new Date(payload.endedAt) : undefined
  const db = await getDb()
  await db.collection<any>(collections.transcriptTurns).updateOne({ sessionId: session._id, turnId }, { $setOnInsert: {
    tenantId: session.tenantId, patientId: session.patientId, sessionId: session._id, turnId, sequence: event.sequence,
    role, startedAt, endedAt, text: payload.text.trim(), asr: payload.asr || null, redaction: payload.redaction || null,
    protocolStage: typeof payload.protocolStage === 'string' && DAILY_PROTOCOL_STAGES.has(payload.protocolStage)
      ? payload.protocolStage
      : null,
    protocolVersion: payload.protocolVersion === 'daily-conversation-v2' ? payload.protocolVersion : null,
    cognitiveSignals: Array.isArray(payload.cognitiveSignals)
      ? payload.cognitiveSignals.filter((item): item is string => typeof item === 'string' && DAILY_COGNITIVE_SIGNALS.has(item)).slice(0, 12)
      : [],
  } }, { upsert: true })
}

async function nextExpectedSequence(sessionId: string) {
  const db = await getDb()
  const events = await db.collection<any>(collections.sessionEvents).find({ sessionId }).project({ sequence: 1 }).sort({ sequence: 1 }).toArray()
  let expected = 0
  for (const event of events) {
    if (event.sequence !== expected) break
    expected++
  }
  return expected
}

function validateArtifactIntent(value: unknown) {
  const body = objectBody(value)
  const clientArtifactId = requiredString(body, 'clientArtifactId', 150)
  const kind = enumValue(body.kind, 'kind', ARTIFACT_KINDS)
  const contentType = requiredString(body, 'contentType', 150)
  const sizeBytes = positiveInteger(body.sizeBytes, 'sizeBytes')
  const hash = requiredString(body, 'hash', 200).toLowerCase()
  if (!/^[a-f0-9]{64}$/.test(hash)) throw badRequest('INVALID_ARTIFACT_HASH', 'Artifact hash must be a lowercase SHA-256 hex digest.')
  return { clientArtifactId, kind, contentType, sizeBytes, hash }
}

function retentionClass(kind: typeof ARTIFACT_KINDS[number]) {
  return kind === 'audio' || kind === 'video' || kind === 'image' ? 'sensitive_media' : 'session_evidence'
}

function validateAcquisitionSummary(value: unknown) {
  const body = value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
  const result: Record<string, number> = {}
  if (body.durationMs !== undefined) result.durationMs = positiveInteger(body.durationMs, 'acquisitionSummary.durationMs', true)
  if (body.patientSpeechMs !== undefined) result.patientSpeechMs = positiveInteger(body.patientSpeechMs, 'acquisitionSummary.patientSpeechMs', true)
  if (body.patientTurns !== undefined) result.patientTurns = positiveInteger(body.patientTurns, 'acquisitionSummary.patientTurns', true)
  return result
}

function parseIfMatch(value?: string) {
  const version = Number(value?.replace(/^W\//, '').replaceAll('"', ''))
  if (!Number.isInteger(version) || version < 1) throw badRequest('IF_MATCH_REQUIRED', 'If-Match must contain the current integer version.')
  return version
}

const sealedJsonCodec: IdempotencyCodec<any> = {
  encode: (value) => ({ sealed: sealSecret(JSON.stringify(value)) }),
  decode: (value) => JSON.parse(openSecret(String((value as Record<string, unknown>).sealed))),
}
