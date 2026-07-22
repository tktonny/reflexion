import assert from 'node:assert/strict'
import test from 'node:test'
import { MongoMemoryReplSet } from 'mongodb-memory-server'
import request from 'supertest'

import { createApp } from '../../app.js'
import { closeMongo, getDb } from '../../lib/mongo.js'
import { hashPassword } from '../../lib/password.js'
import { processNextOutboxEvent } from '../workers/outboxWorker.js'
import { collections } from '../platform/collections.js'
import { openSecret, sha256 } from '../platform/crypto.js'
import { ensureV1Indexes } from '../platform/indexes.js'
import { issueAccessToken } from '../platform/tokens.js'

const TEST_PASSWORD = 'correct-horse-battery-staple'
const NEXT_PASSWORD = 'new-correct-horse-battery-staple'
const TENANT_ID = 'ten_phase3_integration'
const USER_ID = 'usr_phase3_caregiver'
const PATIENT_ID = 'pat_phase3_primary'
const NO_CONSENT_PATIENT_ID = 'pat_phase3_no_consent'
const DEVICE_ID = 'dev_phase3_mirror'
const SERIAL_HASH = sha256('RF-MIRROR-PHASE3-0001')
const ARTIFACT_HASH = sha256('phase3-image-bytes')
const ARTIFACT_SIZE = 128

function idempotencyKey(label: string) {
  return `${label}_phase3_00000001`
}

function bearer(token: string) {
  return { Authorization: `Bearer ${token}` }
}

test('Phase 3 API works end-to-end across auth, pairing, session ingestion and the durable worker', async (t) => {
  const originalFetch = globalThis.fetch
  const originalEnvironment = { ...process.env }
  const replicaSet = await MongoMemoryReplSet.create({
    replSet: { count: 1, storageEngine: 'wiredTiger' },
  })

  process.env.NODE_ENV = 'test'
  process.env.MONGODB_URI = replicaSet.getUri()
  process.env.MONGODB_DB = 'reflexion_phase3_integration'
  process.env.JWT_SECRET = 'phase3-jwt-secret-with-at-least-32-characters'
  process.env.PAIRING_PEPPER = 'phase3-pairing-pepper-with-at-least-32-characters'
  process.env.CREDENTIAL_ENCRYPTION_KEY = 'phase3-encryption-key-with-at-least-32-characters'
  process.env.AUTH_RATE_LIMIT_PER_MINUTE = '1000'
  process.env.API_RATE_LIMIT_PER_MINUTE = '5000'
  process.env.QWEN_API_KEY = 'server-only-qwen-integration-key'
  process.env.QWEN_BASE = 'https://qwen.integration.invalid'
  process.env.QWEN_REALTIME_ENDPOINT = 'wss://qwen.integration.invalid/api-ws/v1/realtime'
  process.env.QWEN_REALTIME_MODEL = 'qwen3.5-omni-flash-realtime'
  process.env.OBJECT_STORE_DRIVER = 's3'
  process.env.OBJECT_STORE_ENDPOINT = 'https://objects.integration.invalid'
  process.env.OBJECT_STORE_BUCKET = 'reflexion-integration'
  process.env.OBJECT_STORE_ACCESS_KEY = 'integration-access-key'
  process.env.OBJECT_STORE_SECRET_KEY = 'integration-secret-key'
  process.env.OBJECT_STORE_REGION = 'ap-southeast-1'
  delete process.env.EMBEDDING_PROVIDER

  let objectVerificationSucceeds = true
  globalThis.fetch = async (input, init) => {
    const url = String(input)
    if (url.startsWith('https://qwen.integration.invalid/api/v1/tokens')) {
      assert.equal(init?.method, 'POST')
      assert.equal(new Headers(init?.headers).get('authorization'), `Bearer ${process.env.QWEN_API_KEY}`)
      return Response.json({ token: 'st-phase3-short-lived', expires_at: Math.floor(Date.now() / 1000) + 900 })
    }
    if (url.startsWith('https://objects.integration.invalid/') && init?.method === 'HEAD') {
      return new Response(null, {
        status: objectVerificationSucceeds ? 200 : 503,
        headers: objectVerificationSucceeds ? {
          'content-length': String(ARTIFACT_SIZE),
          'x-amz-meta-sha256': ARTIFACT_HASH,
        } : {},
      })
    }
    throw new Error(`Unexpected integration fetch: ${init?.method || 'GET'} ${url}`)
  }

  try {
    const db: any = await getDb()
    await ensureV1Indexes(db)
    await seedPlatform(db)
    const app = request(createApp())

    await t.test('health and v1 errors expose the frozen envelope and production headers', async () => {
      const health = await app.get('/health').expect(200)
      assert.deepEqual(health.body, { ok: true })
      assert.equal(health.headers['x-powered-by'], undefined)
      assert.equal(health.headers['x-content-type-options'], 'nosniff')

      const unknown = await app.get('/api/v1/does-not-exist').expect(404)
      assert.equal(unknown.body.error.code, 'ROUTE_NOT_FOUND')
      assert.match(unknown.body.meta.requestId, /^req_/)
      await app.post('/api/v1/device-pairings').send({}).expect(401)
    })

    let humanAccessToken = ''
    let humanRefreshToken = ''
    await t.test('human sessions rotate credentials and relationship authorization fails closed', async () => {
      await app.post('/api/v1/auth/sessions')
        .send({ email: 'caregiver@example.com', password: 'wrong-password' })
        .expect(401)

      const login = await app.post('/api/v1/auth/sessions')
        .send({ email: 'caregiver@example.com', password: TEST_PASSWORD })
        .expect(201)
      humanAccessToken = login.body.data.accessToken
      humanRefreshToken = login.body.data.refreshToken
      assert.equal(login.body.data.actor.userId, USER_ID)

      const me = await app.get('/api/v1/me').set(bearer(humanAccessToken)).expect(200)
      assert.equal(me.body.data.tenantId, TENANT_ID)

      const refresh = await app.post('/api/v1/auth/session-refreshes')
        .send({ refreshToken: humanRefreshToken })
        .expect(201)
      humanAccessToken = refresh.body.data.accessToken
      const rotatedRefreshToken = refresh.body.data.refreshToken
      assert.notEqual(rotatedRefreshToken, humanRefreshToken)
      await app.post('/api/v1/auth/session-refreshes')
        .send({ refreshToken: humanRefreshToken })
        .expect(401)
      humanRefreshToken = rotatedRefreshToken

      const noConsent = await app.post('/api/v1/sessions')
        .set({ ...bearer(humanAccessToken), 'Idempotency-Key': idempotencyKey('no-consent') })
        .send({ type: 'daily_checkin', patientId: NO_CONSENT_PATIENT_ID })
        .expect(403)
      assert.equal(noConsent.body.error.code, 'CONSENT_REQUIRED')

      const strangerLogin = await app.post('/api/v1/auth/sessions')
        .send({ email: 'stranger@example.com', password: TEST_PASSWORD })
        .expect(201)
      const forbiddenSession = await app.post('/api/v1/sessions')
        .set({ ...bearer(strangerLogin.body.data.accessToken), 'Idempotency-Key': idempotencyKey('stranger') })
        .send({ type: 'companion', patientId: PATIENT_ID })
        .expect(403)
      assert.equal(forbiddenSession.body.error.code, 'FORBIDDEN')
    })

    const bootstrapToken = issueAccessToken({
      sub: DEVICE_ID,
      kind: 'bootstrap',
      did: DEVICE_ID,
      serialHash: SERIAL_HASH,
    }, 3600)
    const bootstrapHeader = { 'X-Device-Bootstrap': bootstrapToken }
    let deviceAccessToken = ''
    let deviceRefreshCredential = ''
    let deviceCredentialId = ''
    await t.test('pairing v2 is idempotent, transactional and exchanges a ticket only once', async () => {
      const pairingBody = {
        hardwareRevision: 'mirror-v1',
        softwareVersion: '1.0.0',
        timezone: 'Asia/Shanghai',
        deviceNonce: 'device-install-nonce',
      }
      const pairingKey = idempotencyKey('pairing')
      const pairing = await app.post('/api/v1/device-pairings')
        .set({ ...bootstrapHeader, 'Idempotency-Key': pairingKey })
        .send(pairingBody)
        .expect(201)
      assert.match(pairing.body.data.displayCode, /^\d{6}$/)
      const pairingId = pairing.body.data.pairingId as string

      const replay = await app.post('/api/v1/device-pairings')
        .set({ ...bootstrapHeader, 'Idempotency-Key': pairingKey })
        .send(pairingBody)
        .expect(201)
      assert.equal(replay.body.data.pairingId, pairingId)
      const reused = await app.post('/api/v1/device-pairings')
        .set({ ...bootstrapHeader, 'Idempotency-Key': pairingKey })
        .send({ ...pairingBody, softwareVersion: 'changed' })
        .expect(409)
      assert.equal(reused.body.error.code, 'IDEMPOTENCY_KEY_REUSED')

      const invalidClaim = await app.post('/api/v1/device-pairing-claims')
        .set({ ...bearer(humanAccessToken), 'Idempotency-Key': idempotencyKey('invalid-claim') })
        .send({ pairingCode: '000000', patientId: PATIENT_ID })
        .expect(400)
      assert.equal(invalidClaim.body.error.code, 'PAIRING_CODE_INVALID')
      assert.equal(await db.collection(collections.assignments).countDocuments({ deviceId: DEVICE_ID }), 0)

      const claim = await app.post('/api/v1/device-pairing-claims')
        .set({ ...bearer(humanAccessToken), 'Idempotency-Key': idempotencyKey('valid-claim') })
        .send({ pairingCode: pairing.body.data.displayCode, patientId: PATIENT_ID, mirrorName: 'Bedroom Mirror' })
        .expect(200)
      assert.equal(claim.body.data.patientId, PATIENT_ID)
      assert.equal(await db.collection(collections.assignments).countDocuments({ deviceId: DEVICE_ID, status: 'active' }), 1)

      const paired = await app.get(`/api/v1/device-pairings/${pairingId}`)
        .set(bootstrapHeader)
        .expect(200)
      assert.equal(paired.body.data.state, 'paired')
      assert.ok(paired.body.data.exchangeTicket)

      const exchange = await app.post('/api/v1/device-credentials/exchange')
        .set(bootstrapHeader)
        .send({ pairingId, exchangeTicket: paired.body.data.exchangeTicket })
        .expect(200)
      deviceAccessToken = exchange.body.data.accessToken
      deviceRefreshCredential = exchange.body.data.refreshCredential
      deviceCredentialId = exchange.body.data.credentialId
      assert.equal(exchange.body.data.patientId, PATIENT_ID)

      const consumed = await app.post('/api/v1/device-credentials/exchange')
        .set(bootstrapHeader)
        .send({ pairingId, exchangeTicket: paired.body.data.exchangeTicket })
        .expect(400)
      assert.equal(consumed.body.error.code, 'EXCHANGE_TICKET_INVALID')
      const polledAfterExchange = await app.get(`/api/v1/device-pairings/${pairingId}`)
        .set(bootstrapHeader)
        .expect(200)
      assert.equal(polledAfterExchange.body.data.exchangeTicket, undefined)
    })

    await t.test('device authentication rotates and revokes the previous access token', async () => {
      const device = await app.get(`/api/v1/devices/${DEVICE_ID}`)
        .set(bearer(deviceAccessToken))
        .expect(200)
      assert.equal(device.body.data.assignment.patientId, PATIENT_ID)

      const configuration = await app.get(`/api/v1/devices/${DEVICE_ID}/configuration`)
        .set(bearer(deviceAccessToken))
        .expect(200)
      assert.equal(configuration.body.data.desired.heartbeatIntervalSeconds, 60)
      assert.equal(configuration.body.data.patient.displayName, 'Margaret')

      const previousAccess = deviceAccessToken
      const rotated = await app.post(`/api/v1/devices/${DEVICE_ID}/credential-rotations`)
        .set({ 'Idempotency-Key': idempotencyKey('rotate-device') })
        .send({ credentialId: deviceCredentialId, refreshCredential: deviceRefreshCredential })
        .expect(201)
      deviceAccessToken = rotated.body.data.accessToken
      deviceRefreshCredential = rotated.body.data.refreshCredential
      deviceCredentialId = rotated.body.data.credentialId
      await app.get(`/api/v1/devices/${DEVICE_ID}`).set(bearer(previousAccess)).expect(401)
      await app.get(`/api/v1/devices/${DEVICE_ID}`).set(bearer(deviceAccessToken)).expect(200)
    })

    await t.test('heartbeat is authorized and idempotently persisted', async () => {
      const heartbeatBody = {
        heartbeatId: 'heartbeat-phase3-0001',
        recordedAt: new Date().toISOString(),
        appVersion: '1.0.0',
        networkStatus: 'online',
        micStatus: 'ok',
        speakerStatus: 'ok',
        backendReachable: true,
        diagnostics: { queueDepth: 0 },
      }
      const heartbeatKey = idempotencyKey('heartbeat')
      await app.post(`/api/v1/devices/${DEVICE_ID}/heartbeats`)
        .set({ ...bearer(deviceAccessToken), 'Idempotency-Key': heartbeatKey })
        .send(heartbeatBody)
        .expect(202)
      await app.post(`/api/v1/devices/${DEVICE_ID}/heartbeats`)
        .set({ ...bearer(deviceAccessToken), 'Idempotency-Key': heartbeatKey })
        .send(heartbeatBody)
        .expect(202)
      assert.equal(await db.collection(collections.deviceTelemetry).countDocuments({ 'meta.deviceId': DEVICE_ID }), 1)

      const wrongDevice = await app.post('/api/v1/devices/dev_other/heartbeats')
        .set({ ...bearer(deviceAccessToken), 'Idempotency-Key': idempotencyKey('wrong-heartbeat') })
        .send(heartbeatBody)
        .expect(403)
      assert.equal(wrongDevice.body.error.code, 'FORBIDDEN')
    })

    let sessionId = ''
    let sessionVersion = 0
    let artifactId = ''
    let completionOperationId = ''
    await t.test('session create and realtime ticket are scoped, sealed and idempotent', async () => {
      const createBody = {
        type: 'daily_checkin',
        clientSessionId: 'mirror-local-phase3-session-0001',
        requestedLanguage: 'zh-CN',
        clientContext: { app: 'mirror', protocolVersion: 'daily-conversation-v2' },
      }
      const createKey = idempotencyKey('create-session')
      const created = await app.post('/api/v1/sessions')
        .set({ ...bearer(deviceAccessToken), 'Idempotency-Key': createKey })
        .send(createBody)
        .expect(201)
      sessionId = created.body.data.sessionId
      sessionVersion = created.body.data.stateVersion
      assert.equal(created.body.data.protocolContext.protocolVersion, 'daily-conversation-v2')

      const replay = await app.post('/api/v1/sessions')
        .set({ ...bearer(deviceAccessToken), 'Idempotency-Key': createKey })
        .send(createBody)
        .expect(201)
      assert.equal(replay.body.data.sessionId, sessionId)

      const humanTicket = await app.post(`/api/v1/sessions/${sessionId}/realtime-tickets`)
        .set({ ...bearer(humanAccessToken), 'Idempotency-Key': idempotencyKey('human-ticket') })
        .send({})
        .expect(403)
      assert.equal(humanTicket.body.error.code, 'FORBIDDEN')

      const ticketKey = idempotencyKey('realtime-ticket')
      const ticket = await app.post(`/api/v1/sessions/${sessionId}/realtime-tickets`)
        .set({ ...bearer(deviceAccessToken), 'Idempotency-Key': ticketKey })
        .send({})
        .expect(201)
      assert.equal(ticket.body.data.ticket, 'st-phase3-short-lived')
      assert.equal(ticket.body.data.sessionPolicy.model, 'qwen3.5-omni-flash-realtime')
      const ticketReplay = await app.post(`/api/v1/sessions/${sessionId}/realtime-tickets`)
        .set({ ...bearer(deviceAccessToken), 'Idempotency-Key': ticketKey })
        .send({})
        .expect(201)
      assert.equal(ticketReplay.body.data.ticket, ticket.body.data.ticket)

      const storedIdempotency = await db.collection(collections.idempotencyRecords).findOne({
        routeKey: 'POST:/api/v1/sessions/:sessionId/realtime-tickets', key: ticketKey,
      })
      assert.ok(storedIdempotency?.responseData?.sealed)
      assert.doesNotMatch(JSON.stringify(storedIdempotency?.responseData), /st-phase3-short-lived/)

      const current = await app.get(`/api/v1/sessions/${sessionId}`).set(bearer(deviceAccessToken)).expect(200)
      sessionVersion = current.body.data.stateVersion
      assert.equal(current.body.data.state, 'active')
    })

    const conversationEvents = buildConversationEvents()
    await t.test('ordered event batches report duplicates and preserve only allowed clinical metadata', async () => {
      const batch = await app.post(`/api/v1/sessions/${sessionId}/event-batches`)
        .set({ ...bearer(deviceAccessToken), 'Idempotency-Key': idempotencyKey('event-batch') })
        .send({ events: conversationEvents })
        .expect(202)
      assert.deepEqual(batch.body.data, { accepted: conversationEvents.length, duplicates: 0, nextExpectedSequence: conversationEvents.length })

      const duplicates = await app.post(`/api/v1/sessions/${sessionId}/event-batches`)
        .set({ ...bearer(deviceAccessToken), 'Idempotency-Key': idempotencyKey('event-duplicates') })
        .send({ events: conversationEvents })
        .expect(202)
      assert.deepEqual(duplicates.body.data, { accepted: 0, duplicates: conversationEvents.length, nextExpectedSequence: conversationEvents.length })

      const conflict = await app.post(`/api/v1/sessions/${sessionId}/event-batches`)
        .set({ ...bearer(deviceAccessToken), 'Idempotency-Key': idempotencyKey('event-conflict') })
        .send({ events: [{ ...conversationEvents[0], eventId: 'different-event-at-sequence-zero' }] })
        .expect(409)
      assert.equal(conflict.body.error.code, 'EVENT_SEQUENCE_CONFLICT')

      const turn = await db.collection(collections.transcriptTurns).findOne({ sessionId, turnId: 'turn-patient-warmup' })
      assert.equal(turn?.protocolStage, 'warm_up')
      assert.deepEqual(turn?.cognitiveSignals, ['mood', 'speech_initiation', 'response_latency'])
      assert.equal(turn?.protocolVersion, 'daily-conversation-v2')
    })

    await t.test('artifact plan, commit and retry use signed object-store URLs without proxying media', async () => {
      const plan = await app.post(`/api/v1/sessions/${sessionId}/artifact-upload-plans`)
        .set({ ...bearer(deviceAccessToken), 'Idempotency-Key': idempotencyKey('artifact-plan') })
        .send({ artifacts: [{
          clientArtifactId: 'capture-phase3-0001',
          kind: 'image',
          contentType: 'image/jpeg',
          sizeBytes: ARTIFACT_SIZE,
          hash: ARTIFACT_HASH,
        }] })
        .expect(200)
      artifactId = plan.body.data[0].artifactId
      assert.match(plan.body.data[0].uploadUrl, /^https:\/\/objects\.integration\.invalid\//)
      assert.equal(plan.body.data[0].requiredHeaders['x-amz-meta-sha256'], ARTIFACT_HASH)

      const commit = await app.post(`/api/v1/sessions/${sessionId}/artifacts/commit`)
        .set({ ...bearer(deviceAccessToken), 'Idempotency-Key': idempotencyKey('artifact-commit') })
        .send({ artifacts: [{ artifactId, hash: ARTIFACT_HASH, sizeBytes: ARTIFACT_SIZE }] })
        .expect(202)
      assert.equal(commit.body.data.state, 'queued')

      await db.collection(collections.artifacts).updateOne({ _id: artifactId }, { $set: { state: 'failed' } })
      const retry = await app.post(`/api/v1/sessions/${sessionId}/upload-retries`)
        .set({ ...bearer(deviceAccessToken), 'Idempotency-Key': idempotencyKey('artifact-retry') })
        .send({ artifactIds: [artifactId] })
        .expect(200)
      assert.match(retry.body.data[0].uploadUrl, /^https:\/\/objects\.integration\.invalid\//)
      await app.post(`/api/v1/sessions/${sessionId}/artifacts/commit`)
        .set({ ...bearer(deviceAccessToken), 'Idempotency-Key': idempotencyKey('artifact-recommit') })
        .send({ artifacts: [{ artifactId, hash: ARTIFACT_HASH, sizeBytes: ARTIFACT_SIZE }] })
        .expect(202)
    })

    await t.test('session complete uses optimistic concurrency and atomically queues processing', async () => {
      const completionBody = {
        localCompletedAt: new Date().toISOString(),
        finalSequence: conversationEvents.length - 1,
        artifactIds: [artifactId],
        acquisitionSummary: { durationMs: 210_000, patientSpeechMs: 70_000, patientTurns: 5 },
      }
      const wrongVersion = await app.post(`/api/v1/sessions/${sessionId}/complete`)
        .set({
          ...bearer(deviceAccessToken),
          'Idempotency-Key': idempotencyKey('wrong-version'),
          'If-Match': String(sessionVersion + 100),
        })
        .send(completionBody)
        .expect(409)
      assert.equal(wrongVersion.body.error.code, 'VERSION_CONFLICT')
      const unchanged = await db.collection(collections.sessions).findOne({ _id: sessionId })
      assert.equal(unchanged?.state, 'active')
      assert.equal(unchanged?.stateVersion, sessionVersion)

      const completionKey = idempotencyKey('complete-session')
      const complete = await app.post(`/api/v1/sessions/${sessionId}/complete`)
        .set({
          ...bearer(deviceAccessToken),
          'Idempotency-Key': completionKey,
          'If-Match': String(sessionVersion),
        })
        .send(completionBody)
        .expect(202)
      completionOperationId = complete.body.data.operationId
      assert.equal(complete.body.data.state, 'queued')
      const replay = await app.post(`/api/v1/sessions/${sessionId}/complete`)
        .set({
          ...bearer(deviceAccessToken),
          'Idempotency-Key': completionKey,
          'If-Match': String(sessionVersion),
        })
        .send(completionBody)
        .expect(202)
      assert.equal(replay.body.data.operationId, completionOperationId)

      const queued = await app.get(`/api/v1/sessions/${sessionId}/processing-status`)
        .set(bearer(deviceAccessToken))
        .expect(200)
      assert.equal(queued.body.data.state, 'queued')
      assert.equal(queued.body.data.operationId, completionOperationId)
    })

    await t.test('worker verifies artifacts, processes the session and publishes safe status', async () => {
      await drainOutbox(db)
      const artifact = await db.collection(collections.artifacts).findOne({ _id: artifactId })
      assert.equal(artifact?.state, 'verified')

      const completed = await app.get(`/api/v1/sessions/${sessionId}/processing-status`)
        .set(bearer(deviceAccessToken))
        .expect(200)
      assert.equal(completed.body.data.state, 'completed')
      assert.equal(completed.body.data.result.outcome, 'completed')
      assert.equal(completed.body.data.result.inclusion, 'include')
      assert.equal(completed.body.data.result.anomalyState, 'baseline_building')
      assert.equal(completed.body.data.result.overallScore, undefined)

      assert.equal(await db.collection(collections.featureSnapshots).countDocuments({ sessionId }), 1)
      assert.equal(await db.collection(collections.operationalBaselines).countDocuments({ patientId: PATIENT_ID }), 1)
      assert.equal(await db.collection(collections.outboxEvents).countDocuments({ state: { $in: ['pending', 'retry', 'processing'] } }), 0)
      await waitFor(() => db.collection(collections.auditEvents).countDocuments({ 'actor.id': DEVICE_ID }).then((count: number) => count > 0))
    })

    await t.test('worker retries are visible and recover from the same durable event', async () => {
      const retrySessionId = 'ses_phase3_retry'
      const retryArtifactId = 'art_phase3_retry'
      const retryEventId = 'evt_phase3_retry'
      await db.collection(collections.sessions).insertOne({
        _id: retrySessionId,
        tenantId: TENANT_ID,
        patientId: PATIENT_ID,
        deviceId: DEVICE_ID,
        type: 'companion',
        state: 'ingesting',
        stateVersion: 2,
        artifactIds: [retryArtifactId],
        acquisition: { patientSpeechMs: 0 },
        createdAt: new Date(),
        updatedAt: new Date(),
      })
      await db.collection(collections.artifacts).insertOne({
        _id: retryArtifactId,
        tenantId: TENANT_ID,
        patientId: PATIENT_ID,
        sessionId: retrySessionId,
        state: 'verification_pending',
        objectKey: 'retry/object',
        hash: ARTIFACT_HASH,
        sizeBytes: ARTIFACT_SIZE,
      })
      await db.collection(collections.outboxEvents).insertOne({
        _id: retryEventId,
        eventType: 'session.completed',
        eventVersion: 1,
        tenantId: TENANT_ID,
        patientId: PATIENT_ID,
        aggregateType: 'session',
        aggregateId: retrySessionId,
        correlationId: 'req_phase3_retry',
        state: 'pending',
        attempt: 0,
        occurredAt: new Date(),
        nextAttemptAt: new Date(),
        createdAt: new Date(),
      })

      objectVerificationSucceeds = false
      assert.equal(await processNextOutboxEvent(db), true)
      const failedEvent = await db.collection(collections.outboxEvents).findOne({ _id: retryEventId })
      assert.equal(failedEvent?.state, 'retry')
      const failedSession = await db.collection(collections.sessions).findOne({ _id: retrySessionId })
      assert.equal(failedSession?.state, 'processing_failed')
      assert.equal(failedSession?.processingSummary.retryable, true)

      objectVerificationSucceeds = true
      await db.collection(collections.artifacts).updateOne({ _id: retryArtifactId }, { $set: { state: 'verified' } })
      await db.collection(collections.outboxEvents).updateOne({ _id: retryEventId }, { $set: { nextAttemptAt: new Date(0) } })
      assert.equal(await processNextOutboxEvent(db), true)
      const recoveredEvent = await db.collection(collections.outboxEvents).findOne({ _id: retryEventId })
      assert.equal(recoveredEvent?.state, 'completed')
      const recoveredSession = await db.collection(collections.sessions).findOne({ _id: retrySessionId })
      assert.equal(recoveredSession?.state, 'excluded')
    })

    await t.test('abandon is durable and password reset revokes active human sessions', async () => {
      const abandoned = await app.post('/api/v1/sessions')
        .set({ ...bearer(deviceAccessToken), 'Idempotency-Key': idempotencyKey('create-abandon') })
        .send({ type: 'companion', clientSessionId: 'mirror-local-abandon-0001' })
        .expect(201)
      const abandon = await app.post(`/api/v1/sessions/${abandoned.body.data.sessionId}/abandon`)
        .set({ ...bearer(deviceAccessToken), 'Idempotency-Key': idempotencyKey('abandon-session') })
        .send({ reason: 'user_cancelled' })
        .expect(200)
      assert.equal(abandon.body.data.state, 'abandoned')
      const failedStatus = await app.get(`/api/v1/sessions/${abandoned.body.data.sessionId}/processing-status`)
        .set(bearer(deviceAccessToken))
        .expect(200)
      assert.equal(failedStatus.body.data.state, 'failed')

      await app.post('/api/v1/auth/password-reset-requests')
        .send({ email: 'unknown@example.com' })
        .expect(202)
      await app.post('/api/v1/auth/password-reset-requests')
        .send({ email: 'caregiver@example.com' })
        .expect(202)
      const resetEvent = await db.collection(collections.outboxEvents).findOne({
        eventType: 'password_reset.requested', aggregateId: USER_ID,
      }, { sort: { occurredAt: -1 } })
      assert.ok(resetEvent?.payload?.sealedToken)
      const resetToken = openSecret(String(resetEvent?.payload?.sealedToken))
      await app.post('/api/v1/auth/password-resets')
        .send({ token: resetToken, newPassword: NEXT_PASSWORD })
        .expect(200)
      await app.get('/api/v1/me').set(bearer(humanAccessToken)).expect(401)
      const reusedReset = await app.post('/api/v1/auth/password-resets')
        .send({ token: resetToken, newPassword: NEXT_PASSWORD })
        .expect(400)
      assert.equal(reusedReset.body.error.code, 'PASSWORD_RESET_INVALID')
      await app.post('/api/v1/auth/sessions')
        .send({ email: 'caregiver@example.com', password: NEXT_PASSWORD })
        .expect(201)
    })
  } finally {
    await new Promise((resolve) => setTimeout(resolve, 50))
    globalThis.fetch = originalFetch
    await closeMongo()
    await replicaSet.stop()
    for (const key of Object.keys(process.env)) {
      if (!(key in originalEnvironment)) delete process.env[key]
    }
    Object.assign(process.env, originalEnvironment)
  }
})

async function seedPlatform(db: any) {
  await db.collection(collections.tenants).insertOne({
    _id: TENANT_ID,
    name: 'Reflexion Phase 3',
    status: 'active',
    createdAt: new Date(),
  })
  await db.collection(collections.users).insertMany([
    {
      _id: USER_ID,
      tenantId: TENANT_ID,
      authSubject: 'email:caregiver@example.com',
      email: 'caregiver@example.com',
      emailNormalized: 'caregiver@example.com',
      name: 'Primary Caregiver',
      roles: ['caregiver'],
      scopes: [],
      passwordHash: hashPassword(TEST_PASSWORD),
      status: 'active',
      createdAt: new Date(),
    },
    {
      _id: 'usr_phase3_stranger',
      tenantId: TENANT_ID,
      authSubject: 'email:stranger@example.com',
      email: 'stranger@example.com',
      emailNormalized: 'stranger@example.com',
      name: 'Unrelated Caregiver',
      roles: ['caregiver'],
      scopes: [],
      passwordHash: hashPassword(TEST_PASSWORD),
      status: 'active',
      createdAt: new Date(),
    },
  ])
  await db.collection(collections.patients).insertMany([
    {
      _id: PATIENT_ID,
      tenantId: TENANT_ID,
      displayName: 'Margaret',
      preferredLanguage: 'zh-CN',
      timezone: 'Asia/Shanghai',
      status: 'active',
      version: 1,
      createdAt: new Date(),
    },
    {
      _id: NO_CONSENT_PATIENT_ID,
      tenantId: TENANT_ID,
      displayName: 'No Consent Patient',
      preferredLanguage: 'en-SG',
      timezone: 'Asia/Singapore',
      status: 'active',
      version: 1,
      createdAt: new Date(),
    },
  ])
  const relationshipScopes = ['patient:read', 'patient:write', 'device:assign', 'session:read', 'session:write', 'monitoring:read']
  await db.collection(collections.careRelationships).insertMany([
    {
      _id: 'rel_phase3_primary',
      tenantId: TENANT_ID,
      patientId: PATIENT_ID,
      userId: USER_ID,
      status: 'active',
      scopes: relationshipScopes,
      validTo: null,
      createdAt: new Date(),
    },
    {
      _id: 'rel_phase3_no_consent',
      tenantId: TENANT_ID,
      patientId: NO_CONSENT_PATIENT_ID,
      userId: USER_ID,
      status: 'active',
      scopes: relationshipScopes,
      validTo: null,
      createdAt: new Date(),
    },
  ])
  await db.collection(collections.consents).insertOne({
    _id: 'con_phase3_home_monitoring',
    tenantId: TENANT_ID,
    patientId: PATIENT_ID,
    purpose: 'home_cognitive_monitoring',
    status: 'granted',
    documentVersion: '2026-07',
    signedAt: new Date(),
    withdrawnAt: null,
  })
  await db.collection(collections.devices).insertOne({
    _id: DEVICE_ID,
    serialHash: SERIAL_HASH,
    status: 'provisioned',
    createdAt: new Date(),
    updatedAt: new Date(),
  })
}

function buildConversationEvents() {
  const occurredAt = new Date().toISOString()
  return [
    {
      eventId: 'event-patient-warmup', sequence: 0, occurredAt, kind: 'transcript_turn',
      payload: {
        turnId: 'turn-patient-warmup', role: 'patient',
        text: 'I feel well this morning and I am happy to talk with you today.',
        protocolStage: 'warm_up', protocolVersion: 'daily-conversation-v2',
        cognitiveSignals: ['mood', 'speech_initiation', 'response_latency', 'not_allowed'],
      },
    },
    {
      eventId: 'event-assistant-recall', sequence: 1, occurredAt, kind: 'transcript_turn',
      payload: { turnId: 'turn-assistant-recall', role: 'assistant', text: 'What did you have for dinner yesterday?' },
    },
    {
      eventId: 'event-patient-recall', sequence: 2, occurredAt, kind: 'transcript_turn',
      payload: {
        turnId: 'turn-patient-recall', role: 'patient',
        text: 'Yesterday evening I had rice, steamed fish and vegetables with my daughter.',
        protocolStage: 'yesterday_recall', protocolVersion: 'daily-conversation-v2',
        cognitiveSignals: ['episodic_memory', 'temporal_orientation', 'narrative_coherence'],
      },
    },
    {
      eventId: 'event-patient-sleep', sequence: 3, occurredAt, kind: 'transcript_turn',
      payload: {
        turnId: 'turn-patient-sleep', role: 'patient',
        text: 'I slept through the night and woke up shortly after seven this morning.',
        protocolStage: 'yesterday_recall', protocolVersion: 'daily-conversation-v2',
        cognitiveSignals: ['episodic_memory', 'temporal_orientation'],
      },
    },
    {
      eventId: 'event-patient-plan', sequence: 4, occurredAt, kind: 'transcript_turn',
      payload: {
        turnId: 'turn-patient-plan', role: 'patient',
        text: 'Today I plan to water the plants, have lunch, and walk in the garden.',
        protocolStage: 'present_planning', protocolVersion: 'daily-conversation-v2',
        cognitiveSignals: ['executive_function', 'prospective_memory'],
      },
    },
    {
      eventId: 'event-patient-social', sequence: 5, occurredAt, kind: 'transcript_turn',
      payload: {
        turnId: 'turn-patient-social', role: 'patient',
        text: 'My daughter is visiting on Friday afternoon and we will drink tea together.',
        protocolStage: 'present_planning', protocolVersion: 'daily-conversation-v2',
        cognitiveSignals: ['prospective_memory', 'social_connectedness'],
      },
    },
  ]
}

async function drainOutbox(db: any) {
  for (let index = 0; index < 100; index += 1) {
    if (!await processNextOutboxEvent(db)) return
  }
  throw new Error('Outbox did not drain within 100 events.')
}

async function waitFor(predicate: () => Promise<boolean>, timeoutMs = 2000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (await predicate()) return
    await new Promise((resolve) => setTimeout(resolve, 20))
  }
  throw new Error('Condition was not met before timeout.')
}
