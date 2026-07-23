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
    const ageBand = optionalString(body, 'ageBand', 40)
    const relationshipType = optionalString(body, 'relationshipType', 80) || 'caregiver'
    const patientId = newId('pat')
    const relationshipId = newId('rel')
    const now = new Date()
    const patient = {
      _id: patientId, tenantId: principal.tenantId, displayName, preferredLanguage, timezone,
      ageBand: ageBand || null, status: 'active', version: 1, createdAt: now, updatedAt: now,
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
