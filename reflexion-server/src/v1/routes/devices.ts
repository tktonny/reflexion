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
import { getClientIp, resolveDeviceRegion } from '../platform/region.js'
import { DAILY_CHECKIN_CONSENT_PURPOSE } from '../platform/consent.js'

const PAIRING_TTL_MS = 10 * 60 * 1000
const EXCHANGE_TTL_MS = 5 * 60 * 1000
const CREDENTIAL_RECOVERY_TTL_MS = 24 * 60 * 60 * 1000
const ACCESS_TTL_SECONDS = 15 * 60
const REFRESH_TTL_MS = 90 * 24 * 60 * 60 * 1000
const DEVICE_SCOPES = ['session:write', 'session:read', 'care_plan:read', 'reminder:respond', 'device:heartbeat', 'consent:read', 'consent:write']
const LOCAL_DEVICE_ID_PATTERN = /^dev_[A-Za-z0-9_-]{16,100}$/
const LOCAL_DEVICE_SECRET_MIN_LENGTH = 32

type DevicePairingClaims = { did: string; serialHash?: string; identity: 'bootstrap' | 'local' }

const credentialCodec = {
  encode: (value: Record<string, unknown>) => sealSecret(JSON.stringify(value)),
  decode: (value: unknown) => JSON.parse(openSecret(String(value))) as Record<string, unknown>,
}

export const devicesRouter = Router()

devicesRouter.post('/device-pairings', asyncHandler(async (request, response) => {
  let deviceClaims: DevicePairingClaims
  try {
    deviceClaims = await devicePairingClaims(request, { createIfMissing: true })
  } catch (error) {
    console.warn(`[pairing] request=${request.requestId} stage=PAIRING_SESSION_FAILED code=${safePairingErrorCode(error)}`)
    throw error
  }
  // Correlate a physical attempt without logging a device ID, pairing code, token or credential.
  console.info(`[pairing] request=${request.requestId} stage=PAIRING_SESSION_CREATING identity=${deviceClaims.identity} deviceRef=${sha256(deviceClaims.did).slice(0, 12)}`)
  const result = await executeIdempotent(request, 'POST:/api/v1/device-pairings', async () => {
    const body = objectBody(request.body)
    const hardwareRevision = requiredString(body, 'hardwareRevision', 80)
    const softwareVersion = requiredString(body, 'softwareVersion', 80)
    const timezone = validateTimezone(requiredString(body, 'timezone', 80))
    const deviceNonce = optionalString(body, 'deviceNonce', 200)
    // Region (cn/sg) is decided ONCE here, at pairing, when the mirror is physically present: the
    // device's endpoint probe (probedRegion) wins, backend IP-geo validates, timezone is the fallback.
    const probedRegion = optionalString(body, 'probedRegion', 8)
    const regionSignals = resolveDeviceRegion({ probedRegion, ip: getClientIp(request), timezone })
    if (regionSignals.mismatch) {
      console.warn(`[region] device ${deviceClaims.did}: probe=${regionSignals.probed} != ip=${regionSignals.ip}; using probe (${regionSignals.region}).`)
    }
    const db = await getDb()
    const device = await db.collection<any>(collections.devices).findOne({
      _id: deviceClaims.did,
      ...(deviceClaims.serialHash ? { serialHash: deviceClaims.serialHash } : {}),
      status: { $ne: 'revoked' },
    })
    if (!device) throw unauthorized('The mirror identity has been revoked.')
    const existingAssignment = await db.collection<any>(collections.assignments).findOne({ deviceId: deviceClaims.did, status: 'active' }, { projection: { _id: 1 } })
    if (existingAssignment) throw conflict('DEVICE_ALREADY_CLAIMED', 'This mirror is already paired. Use the recovery flow before pairing it again.')
    await db.collection<any>(collections.pairings).updateMany({
      deviceId: deviceClaims.did, state: 'pending',
    }, { $set: { state: 'cancelled', cancelledAt: new Date() } })

    const expiresAt = new Date(Date.now() + PAIRING_TTL_MS)
    const pairingId = newId('pair')
    // Persist the region decision + its cross-validation signals on BOTH the pairing and the device
    // doc (same blob) so the pairing record is self-contained for audit at claim time.
    const regionSignalsDoc = { ...regionSignals, decidedAt: new Date() }
    const pairing = await insertPairingWithUniqueCode({
      pairingId, deviceId: deviceClaims.did, hardwareRevision, softwareVersion, timezone, deviceNonce,
      region: regionSignals.region, regionSignals: regionSignalsDoc, expiresAt,
    })
    await db.collection<any>(collections.devices).updateOne({ _id: deviceClaims.did }, { $set: {
      hardwareRevision, softwareVersion, timezone,
      region: regionSignals.region,
      regionSignals: regionSignalsDoc,
      updatedAt: new Date(),
    } })
    return { status: 201, data: {
      deviceId: deviceClaims.did, pairingId, displayCode: pairing.displayCode, pairingToken: pairing.pairingToken,
      state: 'pending', expiresAt: expiresAt.toISOString(), pollAfterSeconds: 2,
    } }
  }, `device:${deviceClaims.did}`, credentialCodec)
  response.setHeader('Cache-Control', 'no-store')
  console.info(`[pairing] request=${request.requestId} stage=READY_TO_PAIR identity=${deviceClaims.identity} deviceRef=${sha256(deviceClaims.did).slice(0, 12)}`)
  sendData(response, result.data, result.status)
}))

devicesRouter.get('/device-pairings/:pairingId', asyncHandler(async (request, response) => {
  const deviceClaims = await devicePairingClaims(request)
  const db = await getDb()
  const pairing = await db.collection<any>(collections.pairings).findOne({
    _id: request.params.pairingId, deviceId: deviceClaims.did,
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
    const pairingCode = optionalString(body, 'pairingCode', 6)
    const pairingToken = optionalString(body, 'pairingToken', 500)
    if (!pairingCode && !pairingToken) throw badRequest('PAIRING_INPUT_REQUIRED', 'pairingCode or pairingToken is required.')
    if (pairingCode && !/^\d{6}$/.test(pairingCode)) throw badRequest('INVALID_PAIRING_CODE', 'pairingCode must contain six digits.')
    const patientId = requiredString(body, 'patientId', 100)
    const mirrorName = optionalString(body, 'mirrorName', 80)
    const patient = await authorizePatient(request, patientId, 'device:assign')
    await enforcePairingClaimRate(principal.tenantId, principal.userId)
    await recordPairingEvent(principal.tenantId, principal.userId, 'pairing.claim_initiated', request.requestId)
    const db = await getDb()
    const pairing = await db.collection<any>(collections.pairings).findOne(
      pairingToken ? { pairingTokenHash: sha256(pairingToken) } : { codeHash: hmac(pairingCode!) },
    )
    if (!pairing) {
      await recordPairingFailure(principal.tenantId, principal.userId, request.requestId, 'PAIRING_CODE_INVALID')
      throw new ApiError(400, 'PAIRING_CODE_INVALID', 'Pairing code is invalid or expired.')
    }
    if (pairing.state === 'paired') {
      await recordPairingFailure(principal.tenantId, principal.userId, request.requestId, 'PAIRING_CODE_USED')
      throw new ApiError(400, 'PAIRING_CODE_USED', 'This pairing code has already been used.')
    }
    if (pairing.state !== 'pending' || new Date(pairing.expiresAt).getTime() <= Date.now()) {
      await recordPairingFailure(principal.tenantId, principal.userId, request.requestId, 'PAIRING_CODE_EXPIRED')
      throw new ApiError(400, 'PAIRING_CODE_EXPIRED', 'This pairing code has expired.')
    }
    if (Number(pairing.failedAttempts || 0) >= 5) throw new ApiError(429, 'PAIRING_ATTEMPTS_EXCEEDED', 'Too many pairing attempts. Try again later.', true)
    const assignmentId = newId('asg')
    const credentialId = newId('cred')
    const refreshCredential = randomSecret()
    const refreshExpiresAt = new Date(Date.now() + REFRESH_TTL_MS)
    const credential = deviceCredentialResponse(pairing.deviceId, principal.tenantId, patientId, credentialId, refreshCredential, refreshExpiresAt)
    const exchangeTicket = randomSecret()
    const exchangeTicketExpiresAt = new Date(Date.now() + EXCHANGE_TTL_MS)
    const credentialRecoveryExpiresAt = new Date(Date.now() + CREDENTIAL_RECOVERY_TTL_MS)
    const now = new Date()
    try {
      await inTransaction(async (transactionDb, session) => {
      const existingDeviceAssignment = await transactionDb.collection<any>(collections.assignments).findOne({ deviceId: pairing.deviceId, status: 'active' }, { session, projection: { _id: 1 } })
      if (existingDeviceAssignment) throw conflict('DEVICE_ALREADY_CLAIMED', 'This mirror is already paired. Unlink or reset it before pairing again.')
      const existingPatientAssignment = await transactionDb.collection<any>(collections.assignments).findOne({ tenantId: principal.tenantId, patientId, assignmentType: 'primary', status: 'active' }, { session, projection: { _id: 1 } })
      if (existingPatientAssignment) throw conflict('PATIENT_ALREADY_ASSIGNED', 'This loved one already has a mirror. Unlink it before pairing another one.')
      await transactionDb.collection<any>(collections.assignments).insertOne({
        _id: assignmentId, tenantId: principal.tenantId, deviceId: pairing.deviceId, patientId,
        assignmentType: 'primary', mirrorName: mirrorName || 'Reflexion Mirror', status: 'active',
        assignedAt: now, assignedBy: principal.userId, version: 1,
      }, { session })
      await transactionDb.collection<any>(collections.credentials).insertOne({
        _id: credentialId, deviceId: pairing.deviceId, tenantId: principal.tenantId, patientId,
        secretHash: hashSecret(refreshCredential), secretDigest: sha256(refreshCredential), version: 1, status: 'active', issuedAt: now, refreshExpiresAt,
      }, { session })
      const claimed = await transactionDb.collection<any>(collections.pairings).updateOne({
        _id: pairing._id, state: 'pending', expiresAt: { $gt: now },
      }, { $set: {
        state: 'paired', tenantId: principal.tenantId, claimedBy: principal.userId, claimedPatientId: patientId,
        patientDisplayName: patient.displayName, pairedAt: now, exchangeTicketHash: hashSecret(exchangeTicket),
        exchangeTicketDigest: sha256(exchangeTicket), exchangeTicketCipher: sealSecret(exchangeTicket),
        exchangeTicketExpiresAt, exchangeConsumedAt: null, credentialIssuedAt: now,
        credentialResponseCipher: sealSecret(JSON.stringify(credential)), credentialRecoveryExpiresAt,
      } }, { session })
      if (!claimed.modifiedCount) throw conflict('PAIRING_ALREADY_CLAIMED', 'This pairing session was already claimed.')
      await transactionDb.collection<any>(collections.devices).updateOne({ _id: pairing.deviceId }, { $set: {
        tenantId: principal.tenantId, status: 'active', displayName: mirrorName || 'Reflexion Mirror', updatedAt: now,
      } }, { session })
      await appendOutbox(transactionDb, {
        eventType: 'device.paired', tenantId: principal.tenantId, patientId, aggregateType: 'device',
        aggregateId: String(pairing.deviceId), correlationId: request.requestId,
        payload: { assignmentId, pairingId: pairing._id, credentialIssued: true },
      }, session)
      })
    } catch (error) {
      await recordPairingFailure(principal.tenantId, principal.userId, request.requestId, pairingFailureReason(error))
      throw error
    }
    await recordPairingEvent(principal.tenantId, principal.userId, 'pairing.claim_completed', request.requestId, { credentialIssued: true })
    return { status: 200, data: {
      assignmentId, deviceId: pairing.deviceId, patientId, mirrorName: mirrorName || 'Reflexion Mirror',
      status: 'active', assignedAt: now.toISOString(),
    } }
  })
  sendData(response, result.data, result.status)
}))

devicesRouter.post('/device-credentials/exchange', asyncHandler(async (request, response) => {
  const deviceClaims = await devicePairingClaims(request)
  const body = objectBody(request.body)
  const pairingId = requiredString(body, 'pairingId', 100)
  const exchangeTicket = requiredString(body, 'exchangeTicket', 500)
  const execute = () => deliverCredentialFromExchange(deviceClaims.did, pairingId, exchangeTicket)
  const credential = request.header('Idempotency-Key')
    ? (await executeIdempotent(request, 'POST:/api/v1/device-credentials/exchange', async () => ({ status: 200, data: await execute() }), `device:${deviceClaims.did}:exchange`, credentialCodec)).data
    : await execute()
  response.setHeader('Cache-Control', 'no-store')
  sendData(response, credential)
}))

/**
 * Every mirror assigned to a patient this caregiver can see, one row per loved one.
 *
 * v1 could only look a device up BY ID, so the mirror-management screen — which needs the whole list, and
 * needs to show loved ones with no mirror yet so they can be paired — had no v1 to move to. Built from
 * `device_assignments` (which carries the mirror's display name) joined to the caregiver's patients, so a
 * loved one always appears whether or not a mirror is attached.
 *
 * Deliberately NOT mounted at /devices: the assignment, not the device, is what the caregiver manages, and a
 * bare /devices would read as "all devices in the tenant".
 */
devicesRouter.get('/device-assignments', requireActor('human'), asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw forbidden()
  const db = await getDb()

  // Same visibility rule the rest of the caregiver API uses: an active care relationship, or tenant_admin.
  const isTenantAdmin = principal.roles.includes('tenant_admin')
  const relationships = isTenantAdmin ? [] : await db.collection<any>(collections.careRelationships).find({
    tenantId: principal.tenantId, userId: principal.userId, status: 'active', scopes: 'patient:read',
    $or: [{ validTo: null }, { validTo: { $gt: new Date() } }, { validTo: { $exists: false } }],
  }).project({ patientId: 1 }).toArray()

  const patientFilter: Record<string, unknown> = { tenantId: principal.tenantId, status: { $ne: 'archived' } }
  if (!isTenantAdmin) {
    patientFilter._id = { $in: [...new Set(relationships.map((relationship) => String(relationship.patientId)))] }
  }
  const patients = await db.collection<any>(collections.patients).find(patientFilter)
    .project({ displayName: 1, timezone: 1 }).sort({ _id: 1 }).toArray()
  if (!patients.length) {
    sendData(response, { assignments: [] })
    return
  }

  const patientIds = patients.map((patient) => String(patient._id))
  const assignments = await db.collection<any>(collections.assignments).find({
    tenantId: principal.tenantId, patientId: { $in: patientIds }, status: 'active',
  }).toArray()
  const byPatient = new Map(assignments.map((assignment) => [String(assignment.patientId), assignment]))

  const deviceIds = [...new Set(assignments.map((assignment) => String(assignment.deviceId)))]
  const devices = deviceIds.length
    ? await db.collection<any>(collections.devices).find({ _id: { $in: deviceIds } })
      .project({ serial: 1, softwareVersion: 1, lastHeartbeatAt: 1, status: 1 }).toArray()
    : []
  const byDevice = new Map(devices.map((device) => [String(device._id), device]))

  sendData(response, {
    assignments: patients.map((patient) => {
      const assignment = byPatient.get(String(patient._id))
      const device = assignment ? byDevice.get(String(assignment.deviceId)) : undefined
      return {
        patientId: String(patient._id),
        patientName: String(patient.displayName || ''),
        // Denormalised like patientName: the pairing screen seeds its timezone field from this, and the
        // alternative is a second round trip per loved one just to read one string.
        timezone: String(patient.timezone || 'Asia/Singapore'),
        // Null when this loved one has no mirror yet — the screen needs that row to offer pairing.
        assignmentId: assignment ? String(assignment._id) : null,
        deviceId: assignment ? String(assignment.deviceId) : null,
        mirrorName: assignment?.mirrorName || null,
        assignedAt: assignment?.assignedAt ? new Date(assignment.assignedAt).toISOString() : null,
        device: device ? {
          serial: device.serial || null,
          softwareVersion: device.softwareVersion || null,
          status: device.status || null,
          lastHeartbeatAt: device.lastHeartbeatAt ? new Date(device.lastHeartbeatAt).toISOString() : null,
        } : null,
      }
    }),
  })
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
  const currentDate = new Date().toISOString().slice(0, 10)
  const [configuration, patient, carePlan, productConsent, awayPeriod] = await Promise.all([
    db.collection<any>(collections.deviceConfigurations).findOne({ deviceId: device._id }, { sort: { configVersion: -1 } }),
    db.collection<any>(collections.patients).findOne({ _id: assignment.patientId, tenantId: assignment.tenantId }, { projection: { displayName: 1, preferredLanguage: 1, timezone: 1, version: 1 } }),
    db.collection<any>(collections.carePlans).findOne({ tenantId: assignment.tenantId, patientId: assignment.patientId, status: 'active' }, { sort: { version: -1 }, projection: { version: 1, communicationPreferences: 1, dailyRoutine: 1 } }),
    db.collection<any>(collections.consents).findOne({ tenantId: assignment.tenantId, patientId: assignment.patientId, purpose: 'home_cognitive_monitoring' }, { sort: { createdAt: -1, _id: -1 } }),
    db.collection<any>(collections.awayPeriods).findOne({ tenantId: assignment.tenantId, patientId: assignment.patientId, state: 'active', startsOn: { $lte: currentDate }, endsOn: { $gte: currentDate } }, { sort: { startsOn: -1, _id: -1 } }),
  ])
  const patientConfiguration = patient ? { patientId: patient._id, displayName: patient.displayName,
    preferredLanguage: patient.preferredLanguage, timezone: patient.timezone, version: patient.version,
    carePlan: carePlan ? { version: carePlan.version, communicationPreferences: carePlan.communicationPreferences, dailyRoutine: carePlan.dailyRoutine } : null } : null
  const desired = configuration?.desired || defaultDeviceConfiguration()
  const configuredProductControl = String((desired as Record<string, unknown>).productControl || '').toLowerCase() === 'paused' ? 'paused' : 'active'
  const configuredResearch = String((desired as Record<string, unknown>).researchState || '').toLowerCase()
  const research = ['not_invited', 'invitation_pending', 'consented', 'declined', 'withdrawn', 'study_closed'].includes(configuredResearch) ? configuredResearch : 'not_invited'
  const synchronizedState = {
    product: productConsent?.status === 'granted'
      ? 'accepted'
      : productConsent?.status === 'declined'
        ? 'declined'
        : productConsent?.status === 'withdrawn'
          ? 'withdrawn'
          : 'pending',
    control: configuredProductControl,
    research,
  }
  const away = awayPeriod ? { active: true, startsOn: awayPeriod.startsOn, endsOn: awayPeriod.endsOn, timezone: awayPeriod.timezone } : { active: false }
  sendData(response, configuration ? {
    deviceId: device._id, configVersion: configuration.configVersion, desired: configuration.desired,
    effectiveAt: configuration.effectiveAt, patient: patientConfiguration, consent: synchronizedState, away,
  } : {
    deviceId: device._id, configVersion: 1, desired, effectiveAt: new Date(0).toISOString(), patient: patientConfiguration, consent: synchronizedState, away,
  })
}))

/** The paired Mirror may record the loved one's ordinary product-consent choice. */
devicesRouter.post('/devices/:deviceId/consent', requireActor('device'), asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/devices/:deviceId/consent', async () => {
    const { assignment } = await authorizedDevice(request, request.params.deviceId, 'consent:write')
    const principal = getPrincipal(request)
    if (principal.kind !== 'device' || !principal.scopes.includes('consent:write')) throw forbidden()
    const body = objectBody(request.body)
    const status = enumValue(body.status, 'status', ['granted', 'declined', 'withdrawn'] as const)
    const documentVersion = requiredString(body, 'documentVersion', 80)
    const now = new Date()
    const consent = {
      _id: newId('con'), tenantId: principal.tenantId, patientId: assignment.patientId,
      purpose: DAILY_CHECKIN_CONSENT_PURPOSE, documentVersion, status,
      signedAt: status === 'granted' ? now : null, withdrawnAt: status === 'withdrawn' ? now : null,
      actorId: principal.deviceId, actorType: 'device', createdAt: now,
    }
    const db = await getDb()
    if (status !== 'granted') {
      await db.collection<any>(collections.consents).updateMany({
        tenantId: principal.tenantId, patientId: assignment.patientId,
        purpose: DAILY_CHECKIN_CONSENT_PURPOSE, status: 'granted',
      }, { $set: { status: 'withdrawn', withdrawnAt: now } })
    }
    await db.collection<any>(collections.consents).insertOne(consent)
    return { status: 201, data: { consentId: consent._id, purpose: consent.purpose, documentVersion: consent.documentVersion, status: consent.status, signedAt: consent.signedAt, withdrawnAt: consent.withdrawnAt } }
  })
  sendData(response, result.data, result.status)
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
    const diagnostics = sanitizeHeartbeatDiagnostics(body.diagnostics)
    const db = await getDb()
    await Promise.all([
      db.collection<any>(collections.deviceTelemetry).updateOne({
        'meta.deviceId': principal.deviceId, 'measurements.heartbeatId': heartbeatId,
      }, { $setOnInsert: {
        recordedAt, meta: { tenantId: principal.tenantId, deviceId: principal.deviceId, kind: 'heartbeat' },
        measurements: { heartbeatId, appVersion, networkStatus, micStatus, speakerStatus,
          backendReachable: body.backendReachable, diagnostics },
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

async function devicePairingClaims(request: Request, options: { createIfMissing?: boolean } = {}): Promise<DevicePairingClaims> {
  const token = request.header('X-Device-Bootstrap')?.trim()
  if (token) {
    const claims = await bootstrapClaims(request)
    return { did: claims.did!, serialHash: claims.serialHash, identity: 'bootstrap' }
  }

  const deviceId = request.header('X-Device-Id')?.trim()
  const installSecret = request.header('X-Device-Install-Secret')?.trim()
  if (!deviceId || !LOCAL_DEVICE_ID_PATTERN.test(deviceId) || !installSecret || installSecret.length < LOCAL_DEVICE_SECRET_MIN_LENGTH) {
    throw unauthorized('A mirror identity is required to start pairing.')
  }
  const db = await getDb()
  let device = await db.collection<any>(collections.devices).findOne({ _id: deviceId })
  if (!device && options.createIfMissing) {
    const serialHash = sha256(`local-device:${deviceId}`)
    try {
      await db.collection<any>(collections.devices).insertOne({
        _id: deviceId, serialHash, localDeviceId: deviceId, identitySource: 'local',
        pairingSecretHash: hashSecret(installSecret), status: 'unclaimed', createdAt: new Date(), updatedAt: new Date(),
      })
      device = await db.collection<any>(collections.devices).findOne({ _id: deviceId })
    } catch (error) {
      if (!(error instanceof MongoServerError) || error.code !== 11000) throw error
      device = await db.collection<any>(collections.devices).findOne({ _id: deviceId })
    }
  }
  if (!device || device.status === 'revoked') throw unauthorized('The mirror identity is unavailable.')
  if (device.identitySource !== 'local' || !device.pairingSecretHash || !verifySecret(installSecret, String(device.pairingSecretHash))) {
    throw new ApiError(401, 'DEVICE_RECOVERY_REQUIRED', 'This mirror needs recovery before it can be paired again.')
  }
  return { did: deviceId, serialHash: String(device.serialHash || ''), identity: 'local' }
}

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
    const pairingToken = randomSecret()
    try {
      await db.collection<any>(collections.pairings).insertOne({
        _id: input.pairingId, deviceId: input.deviceId, codeHash: hmac(displayCode), codeHint: displayCode.slice(-2),
        pairingTokenHash: sha256(pairingToken),
        state: 'pending', expiresAt: input.expiresAt, failedAttempts: 0, hardwareRevision: input.hardwareRevision,
        softwareVersion: input.softwareVersion, timezone: input.timezone, deviceNonce: input.deviceNonce,
        region: input.region, regionSignals: input.regionSignals, createdAt: new Date(),
      })
      return { displayCode, pairingToken }
    } catch (error) {
      if (!(error instanceof MongoServerError) || error.code !== 11000) throw error
    }
  }
  throw new ApiError(503, 'PAIRING_CODE_UNAVAILABLE', 'Unable to allocate a pairing code. Retry shortly.', true)
}

async function deliverCredentialFromExchange(deviceId: string, pairingId: string, exchangeTicket: string) {
  const pairingDb = await getDb()
  const pairing = await pairingDb.collection<any>(collections.pairings).findOne({ _id: pairingId, deviceId, state: 'paired' })
  if (!pairing?.exchangeTicketHash || !pairing.tenantId || !pairing.claimedPatientId
    || pairing.exchangeConsumedAt || new Date(pairing.exchangeTicketExpiresAt).getTime() <= Date.now()
    || !verifySecret(exchangeTicket, String(pairing.exchangeTicketHash))) {
    throw new ApiError(400, 'EXCHANGE_TICKET_INVALID', 'The exchange ticket is invalid, expired, or already used.')
  }
  if (pairing.credentialResponseCipher) {
    const credential = JSON.parse(openSecret(String(pairing.credentialResponseCipher))) as Record<string, unknown>
    const consumed = await pairingDb.collection<any>(collections.pairings).updateOne({
      _id: pairingId, deviceId, exchangeConsumedAt: null, exchangeTicketDigest: sha256(exchangeTicket),
    }, { $set: { exchangeConsumedAt: new Date(), credentialDeliveredAt: new Date() }, $unset: { exchangeTicketCipher: '' } })
    if (!consumed.modifiedCount) throw conflict('EXCHANGE_TICKET_USED', 'The exchange ticket has already been used.')
    return credential
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

function sanitizeHeartbeatDiagnostics(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const source = value as Record<string, unknown>
  const safe: Record<string, unknown> = {}
  for (const key of ['platform', 'configuredMode', 'recommendedMode']) {
    if (typeof source[key] === 'string') safe[key] = source[key].slice(0, 80)
  }
  if (typeof source.queueDepth === 'number' && Number.isFinite(source.queueDepth)) safe.queueDepth = Math.max(0, Math.min(10_000, source.queueDepth))
  if (source.checks && typeof source.checks === 'object' && !Array.isArray(source.checks)) {
    const checks: Record<string, string> = {}
    for (const [key, status] of Object.entries(source.checks as Record<string, unknown>)) {
      if (typeof status === 'string' && ['ok', 'warn', 'fail', 'unknown'].includes(status)) checks[key.slice(0, 40)] = status
    }
    safe.checks = checks
  }
  if (Array.isArray(source.pairingStages)) {
    const allowedStages = new Set([
      'WIFI_DISCONNECTED', 'WIFI_CONNECTED', 'INTERNET_UNAVAILABLE', 'INTERNET_AVAILABLE',
      'SERVICE_UNREACHABLE', 'SERVICE_REACHABLE', 'DEVICE_IDENTITY_FAILED', 'PAIRING_SESSION_CREATING',
      'PAIRING_SESSION_FAILED', 'READY_TO_PAIR', 'PAIRING_CODE_EXPIRED', 'PAIRING_CODE_INVALID',
      'PAIRING_CODE_USED', 'DEVICE_ALREADY_CLAIMED', 'CLAIM_IN_PROGRESS', 'ASSIGNMENT_FAILED',
      'CREDENTIAL_ISSUANCE_FAILED', 'CREDENTIAL_STORAGE_FAILED', 'AUTHENTICATION_FAILED', 'PAIRED',
      // Accept the pre-checklist event names while old APKs drain their heartbeat queue.
      'wifi_connected', 'internet_reachable', 'backend_reachable', 'authenticated', 'paired', 'assigned', 'AUTHENTICATION_READY',
      'pairing_session_created', 'pairing_code_displayed', 'credential_saved', 'ready', 'pairing_failed',
    ])
    const allowedReasons = new Set([
      'NETWORK_UNAVAILABLE', 'SERVICE_UNREACHABLE', 'API_BASE_NOT_CONFIGURED', 'TIMEOUT', 'DNS_FAILURE',
      'TLS_FAILURE', 'HEALTH_CHECK_FAILED', 'HTTP_ERROR', 'ROUTE_NOT_FOUND', 'UNAUTHORIZED',
      'DEVICE_RECOVERY_REQUIRED', 'PAIRING_CODE_INVALID', 'PAIRING_CODE_EXPIRED', 'PAIRING_CODE_USED',
      'PAIRING_ATTEMPTS_EXCEEDED', 'DEVICE_ALREADY_CLAIMED', 'CREDENTIAL_MISSING', 'CREDENTIAL_REJECTED',
      'EXCHANGE_TICKET_INVALID', 'ASSIGNMENT_NOT_ACTIVE', 'ASSIGNMENT_FAILED', 'SECURE_STORAGE_FAILURE',
      'PAIRING_CODE_UNAVAILABLE', 'PAIRING_FAILED', 'UNKNOWN',
    ])
    safe.pairingStages = source.pairingStages.filter((event): event is Record<string, unknown> => Boolean(event && typeof event === 'object' && !Array.isArray(event)))
      .map((event) => ({
        stage: event.stage,
        ok: event.ok,
        at: event.at,
        retryable: event.retryable,
        userMessage: event.userMessage,
        supportCode: event.supportCode,
        httpStatus: event.httpStatus,
        requestId: event.requestId,
        reasonCode: event.reasonCode,
      }))
      .filter((event) => allowedStages.has(String(event.stage))
        && typeof event.ok === 'boolean'
        && typeof event.at === 'string'
        && (event.retryable === undefined || typeof event.retryable === 'boolean')
        && (event.userMessage === undefined || typeof event.userMessage === 'string')
        && (event.supportCode === undefined || (typeof event.supportCode === 'string' && /^[A-Z0-9_-]{4,32}$/.test(event.supportCode)))
        && (event.httpStatus === undefined || (typeof event.httpStatus === 'number' && Number.isInteger(event.httpStatus) && event.httpStatus >= 100 && event.httpStatus <= 599))
        && (event.requestId === undefined || (typeof event.requestId === 'string' && /^[A-Za-z0-9._:-]{8,128}$/.test(event.requestId)))
        && (event.reasonCode === undefined || allowedReasons.has(String(event.reasonCode))))
      .slice(-32)
  }
  return safe
}

export async function authorizedDevice(request: Request, deviceId: string, scope = 'patient:read') {
  const principal = getPrincipal(request)
  const db = await getDb()
  const device = await db.collection<any>(collections.devices).findOne({ _id: deviceId, status: { $ne: 'revoked' } })
  if (!device) throw notFound('Device')
  // A device can carry stale active assignments from earlier pairings (to other patients/tenants).
  // For a device principal, match the assignment bound to THAT credential's patient — selecting an
  // arbitrary active assignment would spuriously 403 a legitimately-authenticated device.
  const assignment = await db.collection<any>(collections.assignments).findOne(
    principal.kind === 'device'
      ? { deviceId, patientId: principal.patientId, status: 'active' }
      : { deviceId, status: 'active' },
  )
  if (!assignment) throw forbidden()
  if (principal.kind === 'device') {
    if (principal.deviceId !== deviceId) throw forbidden()
  } else {
    await authorizePatient(request, String(assignment.patientId), scope)
  }
  return { device, assignment }
}

function serializeDevice(device: Record<string, unknown>, assignment: Record<string, unknown>) {
  return {
    deviceId: device._id, displayName: device.displayName || 'Reflexion Mirror', hardwareRevision: device.hardwareRevision,
    softwareVersion: device.softwareVersion, status: device.status, technicalState: device.technicalState || 'unknown', lastSeenAt: device.lastSeenAt || null,
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

async function recordPairingFailure(tenantId: string, userId: string, correlationId: string, reasonCode: string) {
  await recordPairingEvent(tenantId, userId, 'pairing.claim_failed', correlationId, { reasonCode })
}

function pairingFailureReason(error: unknown) {
  if (error instanceof ApiError && ['DEVICE_ALREADY_CLAIMED', 'PATIENT_ALREADY_ASSIGNED'].includes(error.code)) return error.code
  return 'ASSIGNMENT_FAILED'
}

function safePairingErrorCode(error: unknown) {
  return error instanceof ApiError ? error.code : 'INTERNAL_ERROR'
}

async function recordPairingEvent(tenantId: string, userId: string, action: string, correlationId: string, details: Record<string, unknown> = {}) {
  try {
    const db = await getDb()
    await db.collection<any>(collections.auditEvents).insertOne({
      _id: newId('audit'), tenantId, actor: { type: 'user', id: userId }, action,
      object: { type: 'pairing_session', id: 'redacted' }, outcome: action.endsWith('failed') ? 'failure' : 'success',
      details, correlationId, occurredAt: new Date(),
    })
  } catch (error) {
    console.error(`[${correlationId}] pairing diagnostic write failed`, error)
  }
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
