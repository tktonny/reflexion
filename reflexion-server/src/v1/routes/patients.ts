import { Router } from 'express'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { getDb, inTransaction } from '../../lib/mongo.js'
import { authorizePatient, getPrincipal, requireActor } from '../platform/auth.js'
import { collections } from '../platform/collections.js'
import { badRequest, conflict, notFound } from '../platform/errors.js'
import { sendData, sendPage } from '../platform/http.js'
import { newId } from '../platform/ids.js'
import { executeIdempotent } from '../platform/idempotency.js'
import { enumValue, objectBody, optionalString, pagination, requiredString } from '../platform/validation.js'

const PATIENT_STATUSES = ['active', 'inactive'] as const
const DEFAULT_RELATIONSHIP_SCOPES = [
  'patient:read', 'patient:write', 'device:assign', 'care_plan:read', 'care_plan:write', 'monitoring:read',
]

/**
 * The consent purpose a daily check-in is gated on. Kept here next to the consent routes and imported by
 * the session route, so the gate and the thing that reports on the gate cannot drift apart.
 */
export const DAILY_CHECKIN_CONSENT_PURPOSE = 'home_cognitive_monitoring'

const GENDERS = ['male', 'female', 'other'] as const
const SPEECH_SPEEDS = ['slow', 'normal', 'fast'] as const

/**
 * Where each part of a loved one's profile lives in v1, split by who consumes it.
 *
 * `patients.profile` holds what only the caregiver's app displays — exact age, gender, photo, a phone number
 * to call. v1 previously modelled none of it (only a coarse `ageBand`), so it existed solely in the legacy
 * document.
 *
 * Everything that changes how Aria TALKS belongs to the care plan instead, because that is what the device
 * already receives: GET /devices/:deviceId/configuration ships carePlan.dailyRoutine and
 * carePlan.communicationPreferences to the mirror. Putting wake time or topics in `patients` would mean
 * inventing a second delivery path for data the mirror is already wired to read.
 */
export const PATIENT_PROFILE_FIELDS = ['age', 'gender', 'photoUrl', 'phoneNumber'] as const

/** Derived from exact age so the coarse band the monitoring model uses can never disagree with it. */
export function ageBandForAge(age: number | null | undefined): string | null {
  const value = Number(age)
  if (!Number.isFinite(value) || value <= 0) return null
  if (value < 65) return 'under_65'
  if (value < 75) return '65_74'
  if (value < 85) return '75_84'
  return '85_plus'
}

function validateProfile(input: unknown): Record<string, unknown> {
  const body = objectBody(input)
  const profile: Record<string, unknown> = {}
  if ('age' in body) {
    if (body.age === null) profile.age = null
    else {
      const age = Number(body.age)
      if (!Number.isInteger(age) || age < 1 || age > 130) {
        throw badRequest('VALIDATION_FAILED', 'age must be a whole number between 1 and 130.')
      }
      profile.age = age
    }
  }
  if ('gender' in body) profile.gender = body.gender === null ? null : enumValue(body.gender, 'gender', GENDERS)
  if ('photoUrl' in body) profile.photoUrl = optionalString(body, 'photoUrl', 2000) || null
  if ('phoneNumber' in body) profile.phoneNumber = optionalString(body, 'phoneNumber', 40) || null
  if ('speechSpeed' in body) profile.speechSpeed = body.speechSpeed === null ? null : enumValue(body.speechSpeed, 'speechSpeed', SPEECH_SPEEDS)
  return profile
}

export const patientsRouter = Router()
const requireHuman = requireActor('human')

patientsRouter.get('/patients', requireHuman, asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw new Error('Human principal expected.')
  const { limit, cursor } = pagination(request.query as Record<string, unknown>)
  const db = await getDb()
  let patientIds: string[] | undefined
  if (!principal.roles.includes('tenant_admin')) {
    const relationships = await db.collection<any>(collections.careRelationships).find({
      tenantId: principal.tenantId, userId: principal.userId, status: 'active', scopes: 'patient:read',
    }, { projection: { patientId: 1 } }).toArray()
    patientIds = relationships.map((item) => String(item.patientId))
  }
  const filter: Record<string, unknown> = {
    tenantId: principal.tenantId,
    status: { $ne: 'archived' },
    ...(patientIds ? { _id: { $in: patientIds } } : {}),
  }
  if (cursor) filter._id = patientIds ? { $in: patientIds, $gt: cursor } : { $gt: cursor }
  const rows = await db.collection<any>(collections.patients).find(filter).sort({ _id: 1 }).limit(limit + 1).toArray()
  const hasMore = rows.length > limit
  const page = rows.slice(0, limit)
  sendPage(response, page.map(serializePatient), hasMore ? String(page.at(-1)?._id) : null)
}))

patientsRouter.post('/patients', requireHuman, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/patients', async () => {
    const principal = getPrincipal(request)
    if (principal.kind !== 'human') throw new Error('Human principal expected.')
    const body = objectBody(request.body)
    const displayName = requiredString(body, 'displayName', 120)
    const preferredLanguage = requiredString(body, 'preferredLanguage', 40)
    const timezone = validateTimezone(requiredString(body, 'timezone', 80))
    const profile = 'profile' in body ? validateProfile(body.profile) : {}
    // An explicit ageBand is still accepted, but an exact age wins — deriving it here is what stops the
    // coarse band the monitoring model reads from drifting away from the age the caregiver typed.
    const ageBand = typeof profile.age === 'number'
      ? ageBandForAge(profile.age)
      : optionalString(body, 'ageBand', 40)
    const relationshipType = optionalString(body, 'relationshipType', 80) || 'caregiver'
    const patientId = newId('pat')
    const relationshipId = newId('rel')
    const now = new Date()
    const patient = {
      _id: patientId, tenantId: principal.tenantId, displayName, preferredLanguage, timezone,
      ageBand: ageBand || null, profile, status: 'active', version: 1, createdAt: now, updatedAt: now,
    }
    await inTransaction(async (db, session) => {
      await db.collection<any>(collections.patients).insertOne(patient, { session })
      await db.collection<any>(collections.careRelationships).insertOne({
        _id: relationshipId, tenantId: principal.tenantId, patientId, userId: principal.userId,
        relationshipType, scopes: DEFAULT_RELATIONSHIP_SCOPES, status: 'active', validFrom: now, validTo: null,
        createdAt: now,
      }, { session })
      await db.collection<any>(collections.auditEvents).insertOne({
        _id: newId('audit'), tenantId: principal.tenantId, actor: { type: 'user', id: principal.userId },
        action: 'patient.created', object: { type: 'patient', id: patientId }, outcome: 'success',
        correlationId: request.requestId, occurredAt: now,
      }, { session })
    })
    return { status: 201, data: serializePatient(patient) }
  })
  sendData(response, result.data, result.status)
}))

patientsRouter.get('/patients/:patientId', requireHuman, asyncHandler(async (request, response) => {
  const patient = await authorizePatient(request, request.params.patientId, 'patient:read')
  sendData(response, serializePatient(patient))
}))

patientsRouter.patch('/patients/:patientId', requireHuman, asyncHandler(async (request, response) => {
  const patientId = request.params.patientId
  const current = await authorizePatient(request, patientId, 'patient:write')
  const expectedVersion = parseIfMatch(request.header('If-Match'))
  if (Number(current.version) !== expectedVersion) throw conflict('VERSION_CONFLICT', 'The patient was changed by another request. Refresh and retry.')
  const body = objectBody(request.body)
  const update: Record<string, unknown> = {}
  if ('displayName' in body) update.displayName = requiredString(body, 'displayName', 120)
  if ('preferredLanguage' in body) update.preferredLanguage = requiredString(body, 'preferredLanguage', 40)
  if ('timezone' in body) update.timezone = validateTimezone(requiredString(body, 'timezone', 80))
  if ('status' in body) update.status = enumValue(body.status, 'status', PATIENT_STATUSES)
  if ('profile' in body) {
    // Dotted paths, so sending one profile key leaves the others alone — the same partial-update rule the
    // rest of this API follows.
    const profile = validateProfile(body.profile)
    for (const [key, value] of Object.entries(profile)) update[`profile.${key}`] = value
    if ('age' in profile) update.ageBand = ageBandForAge(profile.age as number | null)
  }
  if (!Object.keys(update).length) throw badRequest('VALIDATION_FAILED', 'At least one supported patient field is required.')
  update.updatedAt = new Date()
  const db = await getDb()
  const changed = await db.collection<any>(collections.patients).findOneAndUpdate({
    _id: patientId, tenantId: getPrincipal(request).tenantId, version: expectedVersion,
  }, { $set: update, $inc: { version: 1 } }, { returnDocument: 'after' })
  if (!changed) throw conflict('VERSION_CONFLICT', 'The patient was changed by another request. Refresh and retry.')
  sendData(response, serializePatient(changed))
}))

patientsRouter.get('/patients/:patientId/care-relationships', requireHuman, asyncHandler(async (request, response) => {
  const patientId = request.params.patientId
  await authorizePatient(request, patientId, 'patient:read')
  const principal = getPrincipal(request)
  const db = await getDb()
  const rows = await db.collection<any>(collections.careRelationships).find({
    tenantId: principal.tenantId, patientId, status: 'active',
  }).project({ tenantId: 0 }).toArray()
  sendData(response, rows.map(({ _id, ...row }) => ({ relationshipId: _id, ...row })))
}))

/**
 * Whether this loved one has an active consent, and for what.
 *
 * There was no way to ask. That mattered because consent is a HARD gate, not bookkeeping:
 * POST /sessions throws 403 CONSENT_REQUIRED for a `daily_checkin` without a granted
 * `home_cognitive_monitoring` consent (routes/sessions.ts consentForSession), and the monitoring pipeline
 * excludes any session lacking a consentRef (monitoring/pipeline.ts evaluateConsent). Since no
 * patient-creation path has ever written a consent, the daily check-in — the product's core function —
 * cannot start, and the caregiver app could not even detect why.
 *
 * `requiredPurposes` is returned so a client does not have to hard-code which consent gates check-ins.
 */
patientsRouter.get('/patients/:patientId/consents', requireHuman, asyncHandler(async (request, response) => {
  const patientId = request.params.patientId
  await authorizePatient(request, patientId, 'patient:read')
  const principal = getPrincipal(request)
  const rows = await (await getDb()).collection<any>(collections.consents)
    .find({ tenantId: principal.tenantId, patientId })
    .sort({ signedAt: -1, _id: -1 })
    .limit(50)
    .toArray()
  const granted = rows.filter((row) => row.status === 'granted' && !row.withdrawnAt)
  sendData(response, {
    patientId,
    consents: rows.map(serializeConsent),
    /** Purposes a daily check-in requires. Missing any of these means check-ins cannot run. */
    requiredPurposes: [DAILY_CHECKIN_CONSENT_PURPOSE],
    missingPurposes: [DAILY_CHECKIN_CONSENT_PURPOSE].filter(
      (purpose) => !granted.some((row) => row.purpose === purpose),
    ),
  })
}))

patientsRouter.post('/patients/:patientId/consents', requireHuman, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/patients/:patientId/consents', async () => {
    const patientId = request.params.patientId
    await authorizePatient(request, patientId, 'patient:write')
    const principal = getPrincipal(request)
    const body = objectBody(request.body)
    const purpose = requiredString(body, 'purpose', 100)
    const documentVersion = requiredString(body, 'documentVersion', 80)
    const status = enumValue(body.status, 'status', ['granted', 'withdrawn'] as const)
    const now = new Date()
    const consent = {
      _id: newId('con'), tenantId: principal.tenantId, patientId, purpose, documentVersion, status,
      signedAt: status === 'granted' ? now : null, withdrawnAt: status === 'withdrawn' ? now : null,
      actorId: principal.kind === 'human' ? principal.userId : principal.deviceId, createdAt: now,
    }
    const db = await getDb()
    if (status === 'withdrawn') {
      await db.collection<any>(collections.consents).updateMany({
        tenantId: principal.tenantId, patientId, purpose, status: 'granted',
      }, { $set: { status: 'withdrawn', withdrawnAt: now } })
    }
    await db.collection<any>(collections.consents).insertOne(consent)
    return { status: 201, data: serializeConsent(consent) }
  })
  sendData(response, result.data, result.status)
}))

patientsRouter.get('/patients/:patientId/program-enrollments/current', requireHuman, asyncHandler(async (request, response) => {
  const patientId = request.params.patientId
  await authorizePatient(request, patientId, 'patient:read')
  const principal = getPrincipal(request)
  const db = await getDb()
  const enrollment = await db.collection<any>(collections.programEnrollments).findOne({
    tenantId: principal.tenantId, patientId, status: 'active',
  }, { sort: { enrolledAt: -1 } })
  if (!enrollment) throw notFound('Program enrollment')
  const { _id, ...rest } = enrollment
  sendData(response, { enrollmentId: _id, ...rest })
}))

export function serializePatient(patient: Record<string, unknown>) {
  return {
    patientId: String(patient._id),
    displayName: String(patient.displayName || ''),
    preferredLanguage: String(patient.preferredLanguage || ''),
    timezone: String(patient.timezone || 'UTC'),
    ageBand: patient.ageBand || null,
    profile: {
      age: (patient.profile as Record<string, unknown> | undefined)?.age ?? null,
      gender: (patient.profile as Record<string, unknown> | undefined)?.gender ?? null,
      photoUrl: (patient.profile as Record<string, unknown> | undefined)?.photoUrl ?? null,
      phoneNumber: (patient.profile as Record<string, unknown> | undefined)?.phoneNumber ?? null,
      speechSpeed: (patient.profile as Record<string, unknown> | undefined)?.speechSpeed ?? null,
    },
    status: String(patient.status || 'active'),
    version: Number(patient.version || 1),
  }
}

function serializeConsent(consent: Record<string, unknown>) {
  return { consentId: consent._id, purpose: consent.purpose, documentVersion: consent.documentVersion, status: consent.status,
    signedAt: consent.signedAt, withdrawnAt: consent.withdrawnAt }
}

function validateTimezone(timezone: string) {
  try { new Intl.DateTimeFormat('en', { timeZone: timezone }).format() } catch { throw badRequest('INVALID_TIMEZONE', 'timezone must be a valid IANA timezone.') }
  return timezone
}

function parseIfMatch(value?: string) {
  const normalized = value?.replace(/^W\//, '').replaceAll('"', '')
  const version = Number(normalized)
  if (!Number.isInteger(version) || version < 1) throw badRequest('IF_MATCH_REQUIRED', 'If-Match must contain the current integer version.')
  return version
}
