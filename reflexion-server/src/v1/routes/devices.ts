import { Router, type Request } from 'express'
import { MongoServerError } from 'mongodb'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { getDb, inTransaction } from '../../lib/mongo.js'
import { authorizePatient, getPrincipal, requireActor } from '../platform/auth.js'
import { collections } from '../platform/collections.js'
import { hashSecret, hmac, openSecret, sealSecret, sha256, verifySecret } from '../platform/crypto.js'
import { ApiError, badRequest, conflict, forbidden, notFound, unauthorized } from '../platform/errors.js'
import { sendData } from '../platform/http.js'
import { newId, randomPairingCode, randomSecret } from '../platform/ids.js'
import { executeIdempotent } from '../platform/idempotency.js'
import { appendOutbox } from '../platform/outbox.js'
import { issueAccessToken, verifyAccessToken } from '../platform/tokens.js'
import { enumValue, isoDate, objectBody, optionalString, requiredString } from '../platform/validation.js'

const PAIRING_TTL_MS = 10 * 60 * 1000
const EXCHANGE_TTL_MS = 5 * 60 * 1000
const ACCESS_TTL_SECONDS = 15 * 60
const REFRESH_TTL_MS = 90 * 24 * 60 * 60 * 1000
const DEVICE_SCOPES = ['session:write', 'session:read', 'care_plan:read', 'reminder:respond', 'device:heartbeat']

export const devicesRouter = Router()

devicesRouter.post('/device-pairings', asyncHandler(async (request, response) => {
  const bootstrap = await bootstrapClaims(request)
  const result = await executeIdempotent(request, 'POST:/api/v1/device-pairings', async () => {
    const body = objectBody(request.body)
    const hardwareRevision = requiredString(body, 'hardwareRevision', 80)
    const softwareVersion = requiredString(body, 'softwareVersion', 80)
    const timezone = validateTimezone(requiredString(body, 'timezone', 80))
    const deviceNonce = optionalString(body, 'deviceNonce', 200)
    const db = await getDb()
    const device = await db.collection<any>(collections.devices).findOne({
      _id: bootstrap.did, serialHash: bootstrap.serialHash, status: { $ne: 'revoked' },
    })
    if (!device) throw unauthorized('The device bootstrap credential has been revoked.')
    await db.collection<any>(collections.pairings).updateMany({
      deviceId: bootstrap.did, state: 'pending',
    }, { $set: { state: 'cancelled', cancelledAt: new Date() } })

    const expiresAt = new Date(Date.now() + PAIRING_TTL_MS)
    const pairingId = newId('pair')
    const displayCode = await insertPairingWithUniqueCode({
      pairingId, deviceId: bootstrap.did, hardwareRevision, softwareVersion, timezone, deviceNonce, expiresAt,
    })
    await db.collection<any>(collections.devices).updateOne({ _id: bootstrap.did }, { $set: {
      hardwareRevision, softwareVersion, timezone, updatedAt: new Date(),
    } })
    return { status: 201, data: {
      pairingId, displayCode, state: 'pending', expiresAt: expiresAt.toISOString(), pollAfterSeconds: 2,
    } }
  }, `bootstrap:${bootstrap.did}`)
  response.setHeader('Cache-Control', 'no-store')
  sendData(response, result.data, result.status)
}))

devicesRouter.get('/device-pairings/:pairingId', asyncHandler(async (request, response) => {
  const bootstrap = await bootstrapClaims(request)
  const db = await getDb()
  const pairing = await db.collection<any>(collections.pairings).findOne({
    _id: request.params.pairingId, deviceId: bootstrap.did,
  })
  if (!pairing) throw notFound('Pairing session')
  let state = String(pairing.state)
  if (state === 'pending' && new Date(pairing.expiresAt).getTime() <= Date.now()) {
    state = 'expired'
    await db.collection<any>(collections.pairings).updateOne({ _id: pairing._id, state: 'pending' }, { $set: { state } })
  }
  const data: Record<string, unknown> = {
    pairingId: pairing._id, state, expiresAt: new Date(pairing.expiresAt).toISOString(),
  }
  if (state === 'paired') {
    data.patientDisplayName = pairing.patientDisplayName
    if (!pairing.exchangeConsumedAt) {
      const ticketLive = pairing.exchangeTicketCipher && new Date(pairing.exchangeTicketExpiresAt).getTime() > Date.now()
      if (ticketLive) {
        data.exchangeTicket = openSecret(String(pairing.exchangeTicketCipher))
        data.exchangeTicketExpiresAt = new Date(pairing.exchangeTicketExpiresAt).toISOString()
      } else {
        // Re-issue the exchange ticket to the owning device when it is missing/expired but not yet
        // consumed. Without this, a device that was not polling within the initial 5-minute window
        // (app closed, rebooted, or the caregiver claimed before the mirror was watching) is trapped
        // forever on "paired but no ticket" — the mirror only re-pairs on expired/cancelled, never on
        // paired. Safe: this request is authenticated with THIS device's bootstrap token and the
        // pairing is bound to bootstrap.did, so the ticket only ever reaches the legitimate device.
        const exchangeTicket = randomSecret()
        const exchangeTicketExpiresAt = new Date(Date.now() + EXCHANGE_TTL_MS)
        await db.collection<any>(collections.pairings).updateOne(
          { _id: pairing._id, exchangeConsumedAt: null },
          { $set: {
            exchangeTicketHash: hashSecret(exchangeTicket),
            exchangeTicketDigest: sha256(exchangeTicket),
            exchangeTicketCipher: sealSecret(exchangeTicket),
            exchangeTicketExpiresAt,
          } },
        )
        data.exchangeTicket = exchangeTicket
        data.exchangeTicketExpiresAt = exchangeTicketExpiresAt.toISOString()
      }
    }
  }
  response.setHeader('Cache-Control', 'no-store')
  sendData(response, data)
}))

devicesRouter.post('/device-pairing-claims', requireActor('human'), asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/device-pairing-claims', async () => {
    const principal = getPrincipal(request)
    if (principal.kind !== 'human') throw forbidden()
    const body = objectBody(request.body)
    const pairingCode = requiredString(body, 'pairingCode', 6)
    if (!/^\d{6}$/.test(pairingCode)) throw badRequest('INVALID_PAIRING_CODE', 'pairingCode must contain six digits.')
    const patientId = requiredString(body, 'patientId', 100)
    const mirrorName = optionalString(body, 'mirrorName', 80)
    const patient = await authorizePatient(request, patientId, 'device:assign')
    await enforcePairingClaimRate(principal.tenantId, principal.userId)
    const db = await getDb()
    const pairing = await db.collection<any>(collections.pairings).findOne({
      codeHash: hmac(pairingCode), state: 'pending', expiresAt: { $gt: new Date() }, failedAttempts: { $lt: 5 },
    })
    if (!pairing) {
      await recordPairingFailure(principal.tenantId, principal.userId, request.requestId)
      throw new ApiError(400, 'PAIRING_CODE_INVALID', 'Pairing code is invalid or expired.')
    }
    const assignmentId = newId('asg')
    const exchangeTicket = randomSecret()
    const exchangeTicketExpiresAt = new Date(Date.now() + EXCHANGE_TTL_MS)
    const now = new Date()
    await inTransaction(async (transactionDb, session) => {
      await transactionDb.collection<any>(collections.assignments).updateMany({
        tenantId: principal.tenantId,
        status: 'active',
        $or: [{ deviceId: pairing.deviceId }, { patientId, assignmentType: 'primary' }],
      }, { $set: { status: 'replaced', revokedAt: now, revokedBy: principal.userId } }, { session })
      await transactionDb.collection<any>(collections.assignments).insertOne({
        _id: assignmentId, tenantId: principal.tenantId, deviceId: pairing.deviceId, patientId,
        assignmentType: 'primary', mirrorName: mirrorName || 'Reflexion Mirror', status: 'active',
        assignedAt: now, assignedBy: principal.userId, version: 1,
      }, { session })
      const claimed = await transactionDb.collection<any>(collections.pairings).updateOne({
        _id: pairing._id, state: 'pending', expiresAt: { $gt: now },
      }, { $set: {
        state: 'paired', tenantId: principal.tenantId, claimedBy: principal.userId, claimedPatientId: patientId,
        patientDisplayName: patient.displayName, pairedAt: now, exchangeTicketHash: hashSecret(exchangeTicket),
        exchangeTicketDigest: sha256(exchangeTicket), exchangeTicketCipher: sealSecret(exchangeTicket),
        exchangeTicketExpiresAt, exchangeConsumedAt: null,
      } }, { session })
      if (!claimed.modifiedCount) throw conflict('PAIRING_ALREADY_CLAIMED', 'This pairing session was already claimed.')
      await transactionDb.collection<any>(collections.devices).updateOne({ _id: pairing.deviceId }, { $set: {
        tenantId: principal.tenantId, status: 'active', displayName: mirrorName || 'Reflexion Mirror', updatedAt: now,
      } }, { session })
      await appendOutbox(transactionDb, {
        eventType: 'device.paired', tenantId: principal.tenantId, patientId, aggregateType: 'device',
        aggregateId: String(pairing.deviceId), correlationId: request.requestId,
        payload: { assignmentId },
      }, session)
    })
    return { status: 200, data: {
      assignmentId, deviceId: pairing.deviceId, patientId, mirrorName: mirrorName || 'Reflexion Mirror',
      status: 'active', assignedAt: now.toISOString(),
    } }
  })
  sendData(response, result.data, result.status)
}))

devicesRouter.post('/device-credentials/exchange', asyncHandler(async (request, response) => {
  const bootstrap = await bootstrapClaims(request)
  const body = objectBody(request.body)
  const pairingId = requiredString(body, 'pairingId', 100)
  const exchangeTicket = requiredString(body, 'exchangeTicket', 500)
  const credential = await createCredentialFromExchange(bootstrap.did!, pairingId, exchangeTicket)
  response.setHeader('Cache-Control', 'no-store')
  sendData(response, credential)
}))

devicesRouter.get('/devices/:deviceId', requireActor('human', 'device'), asyncHandler(async (request, response) => {
  const { device, assignment } = await authorizedDevice(request, request.params.deviceId)
  sendData(response, serializeDevice(device, assignment))
}))

devicesRouter.post('/devices/:deviceId/credential-rotations', asyncHandler(async (request, response) => {
  const body = objectBody(request.body)
  const credentialId = requiredString(body, 'credentialId', 100)
  const refreshCredential = requiredString(body, 'refreshCredential', 500)
  const db = await getDb()
  const current = await db.collection<any>(collections.credentials).findOne({
    _id: credentialId, deviceId: request.params.deviceId, secretDigest: sha256(refreshCredential),
    status: 'active', refreshExpiresAt: { $gt: new Date() },
  })
  if (!current?.secretHash || !verifySecret(refreshCredential, String(current.secretHash))) throw unauthorized('The device refresh credential is invalid.')
  const assignment = await db.collection<any>(collections.assignments).findOne({
    tenantId: current.tenantId, deviceId: request.params.deviceId, patientId: current.patientId, status: 'active',
  })
  if (!assignment) throw unauthorized('The device assignment is no longer active.')
  const result = await executeIdempotent(request, 'POST:/api/v1/devices/:deviceId/credential-rotations', async () => {
    const next = await rotateDeviceCredential({
      kind: 'device', subjectId: request.params.deviceId, deviceId: request.params.deviceId,
      credentialId, tenantId: String(current.tenantId), patientId: String(current.patientId), roles: ['device'], scopes: DEVICE_SCOPES,
    }, current)
    return { status: 201, data: next }
  }, `device-refresh:${credentialId}`)
  response.setHeader('Cache-Control', 'no-store')
  sendData(response, result.data, result.status)
}))

devicesRouter.post('/devices/:deviceId/revocations', requireActor('human'), asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/devices/:deviceId/revocations', async () => {
    const { device, assignment } = await authorizedDevice(request, request.params.deviceId, 'device:assign')
    const principal = getPrincipal(request)
    if (principal.kind !== 'human') throw forbidden()
    const body = objectBody(request.body)
    const reason = optionalString(body, 'reason', 500) || 'revoked_by_caregiver'
    const now = new Date()
    await inTransaction(async (db, session) => {
      await db.collection<any>(collections.devices).updateOne({ _id: device._id }, { $set: { status: 'revoked', revokedAt: now, revocationReason: reason } }, { session })
      await db.collection<any>(collections.assignments).updateMany({ deviceId: device._id, status: 'active' }, { $set: { status: 'revoked', revokedAt: now } }, { session })
      await db.collection<any>(collections.credentials).updateMany({ deviceId: device._id, status: 'active' }, { $set: { status: 'revoked', revokedAt: now } }, { session })
      await appendOutbox(db, { eventType: 'device.revoked', tenantId: principal.tenantId, patientId: String(assignment.patientId),
        aggregateType: 'device', aggregateId: String(device._id), correlationId: request.requestId, payload: { reason } }, session)
    })
    return { status: 202, data: { operationId: newId('op'), state: 'accepted' } }
  })
  sendData(response, result.data, result.status)
}))

devicesRouter.get('/devices/:deviceId/configuration', requireActor('human', 'device'), asyncHandler(async (request, response) => {
  const { device, assignment } = await authorizedDevice(request, request.params.deviceId)
  const db = await getDb()
  const [configuration, patient, carePlan] = await Promise.all([
    db.collection<any>(collections.deviceConfigurations).findOne({ deviceId: device._id }, { sort: { configVersion: -1 } }),
    db.collection<any>(collections.patients).findOne({ _id: assignment.patientId, tenantId: assignment.tenantId }, { projection: { displayName: 1, preferredLanguage: 1, timezone: 1, version: 1 } }),
    db.collection<any>(collections.carePlans).findOne({ tenantId: assignment.tenantId, patientId: assignment.patientId, status: 'active' }, { sort: { version: -1 }, projection: { version: 1, communicationPreferences: 1, dailyRoutine: 1 } }),
  ])
  const patientConfiguration = patient ? { patientId: patient._id, displayName: patient.displayName,
    preferredLanguage: patient.preferredLanguage, timezone: patient.timezone, version: patient.version,
    carePlan: carePlan ? { version: carePlan.version, communicationPreferences: carePlan.communicationPreferences, dailyRoutine: carePlan.dailyRoutine } : null } : null
  sendData(response, configuration ? {
    deviceId: device._id, configVersion: configuration.configVersion, desired: configuration.desired,
    effectiveAt: configuration.effectiveAt, patient: patientConfiguration,
  } : {
    deviceId: device._id, configVersion: 1, desired: defaultDeviceConfiguration(), effectiveAt: new Date(0).toISOString(), patient: patientConfiguration,
  })
}))

devicesRouter.post('/devices/:deviceId/heartbeats', requireActor('device'), asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/devices/:deviceId/heartbeats', async () => {
    const principal = getPrincipal(request)
    if (principal.kind !== 'device' || principal.deviceId !== request.params.deviceId) throw forbidden()
    const body = objectBody(request.body)
    const heartbeatId = requiredString(body, 'heartbeatId', 100)
    const recordedAt = isoDate(body.recordedAt, 'recordedAt')
    const appVersion = requiredString(body, 'appVersion', 80)
    const networkStatus = enumValue(body.networkStatus, 'networkStatus', ['online', 'degraded', 'offline'] as const)
    const micStatus = enumValue(body.micStatus, 'micStatus', ['ok', 'unavailable', 'permission_denied', 'error'] as const)
    const speakerStatus = body.speakerStatus === undefined ? 'ok' : enumValue(body.speakerStatus, 'speakerStatus', ['ok', 'unavailable', 'error'] as const)
    if (typeof body.backendReachable !== 'boolean') throw badRequest('VALIDATION_FAILED', 'backendReachable must be boolean.')
    const db = await getDb()
    await Promise.all([
      db.collection<any>(collections.deviceTelemetry).updateOne({
        'meta.deviceId': principal.deviceId, 'measurements.heartbeatId': heartbeatId,
      }, { $setOnInsert: {
        recordedAt, meta: { tenantId: principal.tenantId, deviceId: principal.deviceId, kind: 'heartbeat' },
        measurements: { heartbeatId, appVersion, networkStatus, micStatus, speakerStatus,
          backendReachable: body.backendReachable, diagnostics: body.diagnostics || {} },
      } }, { upsert: true }),
      db.collection<any>(collections.devices).updateOne({ _id: principal.deviceId }, { $set: {
        lastSeenAt: recordedAt, softwareVersion: appVersion, technicalState: networkStatus === 'online' && micStatus === 'ok' ? 'ok' : 'possible_issue',
      } }),
      appendOutbox(db, { eventType: 'device.heartbeat_received', tenantId: principal.tenantId, patientId: principal.patientId,
        aggregateType: 'device', aggregateId: principal.deviceId, correlationId: request.requestId,
        payload: { heartbeatId, recordedAt: recordedAt.toISOString() } }),
    ])
    return { status: 202, data: { operationId: newId('op'), state: 'accepted' } }
  })
  sendData(response, result.data, result.status)
}))

async function bootstrapClaims(request: Request) {
  const token = request.header('X-Device-Bootstrap')?.trim()
  if (!token) throw unauthorized('X-Device-Bootstrap is required.')
  const claims = verifyAccessToken(token, ['bootstrap'])
  if (!claims.did || !claims.serialHash) throw unauthorized('The device bootstrap credential is incomplete.')
  const db = await getDb()
  const device = await db.collection<any>(collections.devices).findOne({ _id: claims.did, serialHash: claims.serialHash, status: { $ne: 'revoked' } }, { projection: { _id: 1 } })
  if (!device) throw unauthorized('The device bootstrap credential has been revoked.')
  return claims
}

async function insertPairingWithUniqueCode(input: Record<string, unknown>) {
  const db = await getDb()
  for (let attempt = 0; attempt < 5; attempt++) {
    const displayCode = randomPairingCode()
    try {
      await db.collection<any>(collections.pairings).insertOne({
        _id: input.pairingId, deviceId: input.deviceId, codeHash: hmac(displayCode), codeHint: displayCode.slice(-2),
        state: 'pending', expiresAt: input.expiresAt, failedAttempts: 0, hardwareRevision: input.hardwareRevision,
        softwareVersion: input.softwareVersion, timezone: input.timezone, deviceNonce: input.deviceNonce,
        createdAt: new Date(),
      })
      return displayCode
    } catch (error) {
      if (!(error instanceof MongoServerError) || error.code !== 11000) throw error
    }
  }
  throw new ApiError(503, 'PAIRING_CODE_UNAVAILABLE', 'Unable to allocate a pairing code. Retry shortly.', true)
}

async function createCredentialFromExchange(deviceId: string, pairingId: string, exchangeTicket: string) {
  const pairingDb = await getDb()
  const pairing = await pairingDb.collection<any>(collections.pairings).findOne({ _id: pairingId, deviceId, state: 'paired' })
  if (!pairing?.exchangeTicketHash || !pairing.tenantId || !pairing.claimedPatientId
    || pairing.exchangeConsumedAt || new Date(pairing.exchangeTicketExpiresAt).getTime() <= Date.now()
    || !verifySecret(exchangeTicket, String(pairing.exchangeTicketHash))) {
    throw new ApiError(400, 'EXCHANGE_TICKET_INVALID', 'The exchange ticket is invalid, expired, or already used.')
  }
  const assignment = await pairingDb.collection<any>(collections.assignments).findOne({
    tenantId: pairing.tenantId, deviceId, patientId: pairing.claimedPatientId, status: 'active',
  })
  if (!assignment) throw conflict('ASSIGNMENT_NOT_ACTIVE', 'The device assignment is no longer active.')
  const credentialId = newId('cred')
  const refreshCredential = randomSecret()
  const refreshExpiresAt = new Date(Date.now() + REFRESH_TTL_MS)
  await inTransaction(async (db, session) => {
    const consumed = await db.collection<any>(collections.pairings).updateOne({
      _id: pairingId, deviceId, exchangeConsumedAt: null, exchangeTicketDigest: sha256(exchangeTicket),
    }, { $set: { exchangeConsumedAt: new Date() }, $unset: { exchangeTicketCipher: '' } }, { session })
    if (!consumed.modifiedCount) throw conflict('EXCHANGE_TICKET_USED', 'The exchange ticket has already been used.')
    await db.collection<any>(collections.credentials).updateMany({ deviceId, status: 'active' }, { $set: { status: 'rotated', rotatedAt: new Date() } }, { session })
    await db.collection<any>(collections.credentials).insertOne({
      _id: credentialId, deviceId, tenantId: pairing.tenantId, patientId: pairing.claimedPatientId,
      secretHash: hashSecret(refreshCredential), secretDigest: sha256(refreshCredential), version: 1, status: 'active', issuedAt: new Date(), refreshExpiresAt,
    }, { session })
  })
  return deviceCredentialResponse(deviceId, String(pairing.tenantId), String(pairing.claimedPatientId), credentialId, refreshCredential, refreshExpiresAt)
}

async function rotateDeviceCredential(principal: Extract<ReturnType<typeof getPrincipal>, { kind: 'device' }>, current: Record<string, unknown>) {
  const credentialId = newId('cred')
  const refreshCredential = randomSecret()
  const refreshExpiresAt = new Date(Date.now() + REFRESH_TTL_MS)
  await inTransaction(async (db, session) => {
    const rotated = await db.collection<any>(collections.credentials).updateOne({
      _id: current._id, status: 'active',
    }, { $set: { status: 'rotated', rotatedAt: new Date() } }, { session })
    if (!rotated.modifiedCount) throw conflict('CREDENTIAL_ALREADY_ROTATED', 'The credential was already rotated.')
    await db.collection<any>(collections.credentials).insertOne({
      _id: credentialId, deviceId: principal.deviceId, tenantId: principal.tenantId, patientId: principal.patientId,
      secretHash: hashSecret(refreshCredential), secretDigest: sha256(refreshCredential), version: Number(current.version || 0) + 1,
      status: 'active', issuedAt: new Date(), refreshExpiresAt,
    }, { session })
  })
  return deviceCredentialResponse(principal.deviceId, principal.tenantId, principal.patientId, credentialId, refreshCredential, refreshExpiresAt)
}

function deviceCredentialResponse(deviceId: string, tenantId: string, patientId: string, credentialId: string, refreshCredential: string, refreshExpiresAt: Date) {
  return {
    deviceId,
    credentialId,
    patientId,
    accessToken: issueAccessToken({ sub: deviceId, kind: 'device', did: deviceId, tid: tenantId, pid: patientId,
      cid: credentialId, roles: ['device'], scopes: DEVICE_SCOPES }, ACCESS_TTL_SECONDS),
    accessTokenExpiresAt: new Date(Date.now() + ACCESS_TTL_SECONDS * 1000).toISOString(),
    refreshCredential,
    refreshCredentialExpiresAt: refreshExpiresAt.toISOString(),
  }
}

async function authorizedDevice(request: Request, deviceId: string, scope = 'patient:read') {
  const principal = getPrincipal(request)
  const db = await getDb()
  const device = await db.collection<any>(collections.devices).findOne({ _id: deviceId, status: { $ne: 'revoked' } })
  if (!device) throw notFound('Device')
  const assignment = await db.collection<any>(collections.assignments).findOne({ deviceId, status: 'active' })
  if (!assignment) throw forbidden()
  if (principal.kind === 'device') {
    if (principal.deviceId !== deviceId || principal.patientId !== assignment.patientId) throw forbidden()
  } else {
    await authorizePatient(request, String(assignment.patientId), scope)
  }
  return { device, assignment }
}

function serializeDevice(device: Record<string, unknown>, assignment: Record<string, unknown>) {
  return {
    deviceId: device._id, displayName: device.displayName || 'Reflexion Mirror', hardwareRevision: device.hardwareRevision,
    softwareVersion: device.softwareVersion, status: device.status, lastSeenAt: device.lastSeenAt || null,
    assignment: { assignmentId: assignment._id, patientId: assignment.patientId, status: assignment.status, assignedAt: assignment.assignedAt },
  }
}

async function enforcePairingClaimRate(tenantId: string, userId: string) {
  const db = await getDb()
  const count = await db.collection<any>(collections.auditEvents).countDocuments({
    tenantId, 'actor.id': userId, action: 'pairing.claim_failed', occurredAt: { $gt: new Date(Date.now() - PAIRING_TTL_MS) },
  })
  if (count >= 5) throw new ApiError(429, 'PAIRING_ATTEMPTS_EXCEEDED', 'Too many failed pairing attempts. Try again later.', true)
}

async function recordPairingFailure(tenantId: string, userId: string, correlationId: string) {
  const db = await getDb()
  await db.collection<any>(collections.auditEvents).insertOne({
    _id: newId('audit'), tenantId, actor: { type: 'user', id: userId }, action: 'pairing.claim_failed',
    object: { type: 'pairing_code', id: 'redacted' }, outcome: 'failure', correlationId, occurredAt: new Date(),
  })
}

function validateTimezone(timezone: string) {
  try { new Intl.DateTimeFormat('en', { timeZone: timezone }).format() } catch { throw badRequest('INVALID_TIMEZONE', 'timezone must be a valid IANA timezone.') }
  return timezone
}

function defaultDeviceConfiguration() {
  return {
    heartbeatIntervalSeconds: 60, pairingPollSeconds: 2, sessionUploadBatchSize: 50,
    capture: { microphoneRequired: true, cameraRequiredForAssessment: false },
    realtime: { provider: 'qwen', credentialMode: 'session_ticket' },
  }
}
