import 'dotenv/config'
import { randomBytes } from 'node:crypto'
import { hashPassword } from '../lib/password.js'
import { closeMongo, getDb } from '../lib/mongo.js'
import { collections } from '../v1/platform/collections.js'
import { sha256 } from '../v1/platform/crypto.js'
import { newId } from '../v1/platform/ids.js'
import { issueAccessToken } from '../v1/platform/tokens.js'

type JsonRecord = Record<string, any>

const apiBase = (argument('base') || process.env.SMOKE_API_BASE || '').replace(/\/+$/, '')
if (!apiBase.startsWith('https://')) throw new Error('Provide --base=https://your-production-api.')

const runId = `deployment-smoke-${Date.now()}-${randomBytes(4).toString('hex')}`
const email = `${runId}@reflexion.invalid`
const password = randomBytes(24).toString('base64url')
const userId = newId('usr')
const deviceId = newId('dev')
let patientId = ''
let sessionId = ''
let pairingId = ''
const checks: string[] = []

try {
  const db = await getDb()
  const tenant = await db.collection<any>(collections.tenants).findOne({ status: 'active' }, { sort: { createdAt: 1 } })
  if (!tenant?._id) throw new Error('No active tenant is available for the deployment smoke test.')

  await db.collection<any>(collections.users).insertOne({
    _id: userId,
    tenantId: String(tenant._id),
    authSubject: `local:${email}`,
    email,
    emailNormalized: email,
    name: 'Deployment Smoke Admin',
    passwordHash: hashPassword(password),
    status: 'active',
    roles: ['tenant_admin', 'provider', 'caregiver'],
    scopes: [
      'patient:read', 'patient:write', 'device:assign', 'session:read', 'session:write',
      'care_plan:read', 'care_plan:write', 'monitoring:read', 'review:read', 'review:write',
    ],
    createdAt: new Date(),
    updatedAt: new Date(),
  })

  const login = await request('/api/v1/auth/sessions', {
    method: 'POST',
    body: { email, password },
    expectedStatus: 201,
  })
  const humanToken = required(login.data?.accessToken, 'human access token')
  checks.push('admin_login')

  const patient = await request('/api/v1/patients', {
    method: 'POST',
    token: humanToken,
    idempotencyKey: key('patient'),
    body: {
      displayName: `Deployment Smoke ${runId.slice(-8)}`,
      preferredLanguage: 'zh-CN',
      timezone: 'Asia/Shanghai',
      ageBand: '75-84',
      relationshipType: 'deployment_test',
    },
    expectedStatus: 201,
  })
  patientId = required(patient.data?.patientId, 'patient id')
  checks.push('patient_create')

  await request(`/api/v1/patients/${patientId}/consents`, {
    method: 'POST',
    token: humanToken,
    idempotencyKey: key('consent'),
    body: { purpose: 'home_cognitive_monitoring', documentVersion: 'deployment-smoke-v1', status: 'granted' },
    expectedStatus: 201,
  })
  checks.push('monitoring_consent')

  const serialHash = sha256(`reflexion-${runId}`)
  await db.collection<any>(collections.devices).insertOne({
    _id: deviceId,
    serialHash,
    hardwareRevision: 'deployment-smoke',
    softwareVersion: '1.0.1',
    status: 'provisioned',
    createdAt: new Date(),
    updatedAt: new Date(),
  })
  const bootstrapToken = issueAccessToken({
    sub: deviceId,
    kind: 'bootstrap',
    did: deviceId,
    serialHash,
    roles: ['device_bootstrap'],
    scopes: ['device:pair'],
  }, 60 * 60)
  checks.push('device_provision')

  const pairing = await request('/api/v1/device-pairings', {
    method: 'POST',
    bootstrapToken,
    idempotencyKey: key('pairing'),
    body: {
      hardwareRevision: 'deployment-smoke',
      softwareVersion: '1.0.1',
      timezone: 'Asia/Shanghai',
      deviceNonce: randomBytes(16).toString('hex'),
    },
    expectedStatus: 201,
  })
  pairingId = required(pairing.data?.pairingId, 'pairing id')
  const pairingCode = required(pairing.data?.displayCode, 'pairing code')

  await request('/api/v1/device-pairing-claims', {
    method: 'POST',
    token: humanToken,
    idempotencyKey: key('pairing-claim'),
    body: { pairingCode, patientId, mirrorName: 'Deployment Smoke Mirror' },
    expectedStatus: 200,
  })
  const paired = await request(`/api/v1/device-pairings/${pairingId}`, {
    bootstrapToken,
    expectedStatus: 200,
  })
  const exchangeTicket = required(paired.data?.exchangeTicket, 'exchange ticket')
  const exchange = await request('/api/v1/device-credentials/exchange', {
    method: 'POST',
    bootstrapToken,
    body: { pairingId, exchangeTicket },
    expectedStatus: 200,
  })
  const deviceToken = required(exchange.data?.accessToken, 'device access token')
  checks.push('pairing_v2')

  await request(`/api/v1/devices/${deviceId}/heartbeats`, {
    method: 'POST',
    token: deviceToken,
    idempotencyKey: key('heartbeat'),
    body: {
      heartbeatId: `${runId}-heartbeat`,
      recordedAt: new Date().toISOString(),
      appVersion: '1.0.1',
      networkStatus: 'online',
      micStatus: 'ok',
      speakerStatus: 'ok',
      backendReachable: true,
      diagnostics: { source: 'authenticated_deployment_smoke' },
    },
    expectedStatus: 202,
  })
  checks.push('authenticated_heartbeat')

  const createdSession = await request('/api/v1/sessions', {
    method: 'POST',
    token: deviceToken,
    idempotencyKey: key('session'),
    body: {
      type: 'daily_checkin',
      clientSessionId: runId,
      requestedLanguage: 'zh-CN',
      clientContext: { app: 'mirror', protocolVersion: 'daily-conversation-v2', source: 'deployment_smoke' },
    },
    expectedStatus: 201,
  })
  sessionId = required(createdSession.data?.sessionId, 'session id')

  const realtime = await request(`/api/v1/sessions/${sessionId}/realtime-tickets`, {
    method: 'POST',
    token: deviceToken,
    idempotencyKey: key('realtime-ticket'),
    body: {},
    expectedStatus: 201,
    timeoutMs: 30_000,
  })
  required(realtime.data?.ticket, 'Qwen realtime ticket')
  checks.push('realtime_ticket')

  const events = conversationEvents(runId)
  await request(`/api/v1/sessions/${sessionId}/event-batches`, {
    method: 'POST',
    token: deviceToken,
    idempotencyKey: key('events'),
    body: { events },
    expectedStatus: 202,
  })
  const currentSession = await request(`/api/v1/sessions/${sessionId}`, { token: deviceToken, expectedStatus: 200 })
  const stateVersion = Number(currentSession.data?.stateVersion)
  if (!Number.isInteger(stateVersion)) throw new Error('Session state version is missing.')

  const completed = await request(`/api/v1/sessions/${sessionId}/complete`, {
    method: 'POST',
    token: deviceToken,
    idempotencyKey: key('complete'),
    ifMatch: String(stateVersion),
    body: {
      localCompletedAt: new Date().toISOString(),
      finalSequence: events.length - 1,
      artifactIds: [],
      acquisitionSummary: { durationMs: 210_000, patientSpeechMs: 70_000, patientTurns: 5 },
    },
    expectedStatus: 202,
  })
  required(completed.data?.operationId, 'processing operation id')
  checks.push('session_ingestion')

  const processing = await pollProcessing(sessionId, deviceToken)
  if (processing.state !== 'completed') {
    throw new Error(`Worker ended in unexpected state ${String(processing.state)}.`)
  }
  checks.push('worker_processing')

  console.log(JSON.stringify({
    ok: true,
    apiBase,
    runId,
    checks,
    processing: {
      state: processing.state,
      outcome: processing.result?.outcome,
      inclusion: processing.result?.inclusion,
      anomalyState: processing.result?.anomalyState,
    },
  }, null, 2))
} finally {
  try {
    await cleanup()
  } finally {
    await closeMongo()
  }
}

function argument(name: string) {
  const prefix = `--${name}=`
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length)
}

function key(label: string) {
  return `${runId}-${label}`
}

function required(value: unknown, label: string) {
  if (typeof value !== 'string' || !value) throw new Error(`Missing ${label}.`)
  return value
}

async function request(path: string, options: {
  method?: string
  token?: string
  bootstrapToken?: string
  idempotencyKey?: string
  ifMatch?: string
  body?: JsonRecord
  expectedStatus: number
  timeoutMs?: number
}) {
  const headers: Record<string, string> = { Accept: 'application/json' }
  if (options.token) headers.Authorization = `Bearer ${options.token}`
  if (options.bootstrapToken) headers['X-Device-Bootstrap'] = options.bootstrapToken
  if (options.idempotencyKey) headers['Idempotency-Key'] = options.idempotencyKey
  if (options.ifMatch) headers['If-Match'] = options.ifMatch
  if (options.body !== undefined) headers['Content-Type'] = 'application/json'
  const response = await fetch(`${apiBase}${path}`, {
    method: options.method || 'GET',
    headers,
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    signal: AbortSignal.timeout(options.timeoutMs || 20_000),
  })
  const payload = await response.json().catch(() => ({})) as JsonRecord
  if (response.status !== options.expectedStatus) {
    throw new Error(`${options.method || 'GET'} ${path} returned ${response.status}: ${JSON.stringify(payload)}`)
  }
  return payload
}

async function pollProcessing(id: string, deviceToken: string) {
  const deadline = Date.now() + 180_000
  while (Date.now() < deadline) {
    const payload = await request(`/api/v1/sessions/${id}/processing-status`, {
      token: deviceToken,
      expectedStatus: 200,
    })
    const data = payload.data || {}
    if (['completed', 'processing_failed', 'review_pending', 'excluded'].includes(String(data.state))) return data
    await new Promise((resolve) => setTimeout(resolve, 2_000))
  }
  throw new Error('Worker processing did not reach a terminal state within 180 seconds.')
}

function conversationEvents(prefix: string) {
  const occurredAt = new Date().toISOString()
  return [
    transcript(`${prefix}-warmup`, 0, occurredAt, 'patient', 'I feel well this morning and I am happy to talk with you today.',
      'warm_up', ['mood', 'speech_initiation', 'response_latency']),
    transcript(`${prefix}-recall-question`, 1, occurredAt, 'assistant', 'What did you have for dinner yesterday?'),
    transcript(`${prefix}-recall`, 2, occurredAt, 'patient', 'Yesterday I had rice, steamed fish and vegetables with my daughter.',
      'yesterday_recall', ['episodic_memory', 'temporal_orientation', 'narrative_coherence']),
    transcript(`${prefix}-sleep`, 3, occurredAt, 'patient', 'I slept through the night and woke shortly after seven.',
      'yesterday_recall', ['episodic_memory', 'temporal_orientation']),
    transcript(`${prefix}-plan`, 4, occurredAt, 'patient', 'Today I will water the plants, have lunch, and walk in the garden.',
      'present_planning', ['executive_function', 'prospective_memory']),
    transcript(`${prefix}-social`, 5, occurredAt, 'patient', 'My daughter is visiting on Friday and we will drink tea together.',
      'present_planning', ['prospective_memory', 'social_connectedness']),
  ]
}

function transcript(eventId: string, sequence: number, occurredAt: string, role: string, text: string,
  protocolStage?: string, cognitiveSignals?: string[]) {
  return {
    eventId,
    sequence,
    occurredAt,
    kind: 'transcript_turn',
    payload: {
      turnId: `${eventId}-turn`,
      role,
      text,
      ...(protocolStage ? { protocolStage, protocolVersion: 'daily-conversation-v2', cognitiveSignals } : {}),
    },
  }
}

async function cleanup() {
  const db = await getDb()
  const ids = [userId, deviceId, patientId, sessionId, pairingId].filter(Boolean)
  const idempotencyPrefix = `^${runId.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`
  const references: JsonRecord[] = [
    { _id: { $in: ids } },
    { userId },
    { deviceId },
    { 'meta.deviceId': deviceId },
    { 'actor.id': { $in: [userId, deviceId] } },
    { 'object.id': { $in: ids } },
    { key: { $regex: idempotencyPrefix } },
  ]
  if (patientId) references.push({ patientId })
  if (sessionId) references.push({ sessionId })
  for (const collectionName of Object.values(collections)) {
    await db.collection(collectionName).deleteMany({
      $or: references,
    })
  }
}
