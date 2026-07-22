import { Router } from 'express'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { getDb } from '../../lib/mongo.js'
import { authorizePatient, getPrincipal, requireActor } from '../platform/auth.js'
import { collections } from '../platform/collections.js'
import { badRequest, conflict, forbidden, notFound } from '../platform/errors.js'
import { sendData } from '../platform/http.js'
import { newId } from '../platform/ids.js'
import { executeIdempotent } from '../platform/idempotency.js'
import { appendOutbox } from '../platform/outbox.js'
import { enumValue, isoDate, objectBody, optionalString, requiredString } from '../platform/validation.js'

export const carePlanRouter = Router()
const requireCarePlanActor = requireActor('human', 'device')

carePlanRouter.get('/patients/:patientId/care-plan', requireCarePlanActor, asyncHandler(async (request, response) => {
  const patientId = request.params.patientId
  await authorizePatient(request, patientId, 'care_plan:read')
  const principal = getPrincipal(request)
  const db = await getDb()
  const plan = await db.collection<any>(collections.carePlans).findOne({
    tenantId: principal.tenantId, patientId, status: 'active',
  }, { sort: { version: -1 } })
  if (!plan) throw notFound('Care plan')
  sendData(response, serializeCarePlan(plan))
}))

carePlanRouter.put('/patients/:patientId/care-plan', requireCarePlanActor, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'PUT:/api/v1/patients/:patientId/care-plan', async () => {
    const principal = getPrincipal(request)
    if (principal.kind !== 'human') throw forbidden('Only an authorized caregiver or provider can change a care plan.')
    const patientId = request.params.patientId
    await authorizePatient(request, patientId, 'care_plan:write')
    const body = objectBody(request.body)
    const db = await getDb()
    const current = await db.collection<any>(collections.carePlans).findOne({ tenantId: principal.tenantId, patientId, status: 'active' }, { sort: { version: -1 } })
    const expected = current ? parseIfMatch(request.header('If-Match')) : 0
    if (current && Number(current.version) !== expected) throw conflict('VERSION_CONFLICT', 'The care plan changed. Refresh and retry.')
    const now = new Date()
    const next = {
      _id: current?._id || newId('plan'), tenantId: principal.tenantId, patientId, version: expected + 1,
      status: 'active', effectiveFrom: isoDate(body.effectiveFrom || now.toISOString(), 'effectiveFrom'),
      effectiveTo: body.effectiveTo ? isoDate(body.effectiveTo, 'effectiveTo') : null,
      ownerId: principal.userId,
      dailyRoutine: validObject(body.dailyRoutine, 'dailyRoutine'),
      communicationPreferences: validObject(body.communicationPreferences, 'communicationPreferences'),
      safetyNotes: optionalString(body, 'safetyNotes', 2000), updatedAt: now,
    }
    await db.collection<any>(collections.carePlans).replaceOne({ _id: next._id, ...(current ? { version: expected } : {}) }, {
      ...next, createdAt: current?.createdAt || now,
    }, { upsert: !current })
    return { status: current ? 200 : 201, data: serializeCarePlan(next) }
  })
  sendData(response, result.data, result.status)
}))

carePlanRouter.get('/patients/:patientId/medication-plans', requireCarePlanActor, asyncHandler(async (request, response) => {
  const patientId = request.params.patientId
  await authorizePatient(request, patientId, 'care_plan:read')
  const principal = getPrincipal(request)
  const db = await getDb()
  const plans = await db.collection<any>(collections.medicationPlans).find({
    tenantId: principal.tenantId, patientId, status: { $in: ['active', 'paused'] },
  }).sort({ createdAt: -1 }).toArray()
  sendData(response, plans.map(serializeMedicationPlan))
}))

carePlanRouter.post('/patients/:patientId/medication-plans', requireCarePlanActor, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/patients/:patientId/medication-plans', async () => {
    const principal = getPrincipal(request)
    if (principal.kind !== 'human') throw forbidden('The mirror cannot create or change medication instructions.')
    const patientId = request.params.patientId
    await authorizePatient(request, patientId, 'care_plan:write')
    const body = objectBody(request.body)
    const schedule = validateSchedule(body.schedule)
    const source = enumValue(body.source, 'source', ['caregiver', 'provider'] as const)
    if (source === 'provider' && !principal.roles.includes('provider')) throw forbidden('Provider source requires a provider role.')
    const plan = {
      _id: newId('plan'), tenantId: principal.tenantId, patientId,
      displayName: requiredString(body, 'displayName', 160), instructions: optionalString(body, 'instructions', 1000),
      schedule, source, status: 'active', version: 1, createdBy: principal.userId, createdAt: new Date(), updatedAt: new Date(),
    }
    const db = await getDb()
    await db.collection<any>(collections.medicationPlans).insertOne(plan)
    await appendOutbox(db, { eventType: 'medication_plan.changed', tenantId: principal.tenantId, patientId,
      aggregateType: 'medication_plan', aggregateId: plan._id, correlationId: request.requestId, payload: { change: 'created' } })
    return { status: 201, data: serializeMedicationPlan(plan) }
  })
  sendData(response, result.data, result.status)
}))

carePlanRouter.patch('/medication-plans/:planId', requireCarePlanActor, asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw forbidden('The mirror cannot create or change medication instructions.')
  const db = await getDb()
  const plan = await db.collection<any>(collections.medicationPlans).findOne({ _id: request.params.planId, tenantId: principal.tenantId })
  if (!plan) throw notFound('Medication plan')
  await authorizePatient(request, String(plan.patientId), 'care_plan:write')
  const expected = parseIfMatch(request.header('If-Match'))
  if (Number(plan.version) !== expected) throw conflict('VERSION_CONFLICT', 'The medication plan changed. Refresh and retry.')
  const body = objectBody(request.body)
  const update: Record<string, unknown> = { updatedAt: new Date() }
  if ('displayName' in body) update.displayName = requiredString(body, 'displayName', 160)
  if ('instructions' in body) update.instructions = optionalString(body, 'instructions', 1000)
  if ('schedule' in body) update.schedule = validateSchedule(body.schedule)
  if ('status' in body) update.status = enumValue(body.status, 'status', ['active', 'paused', 'ended'] as const)
  if (Object.keys(update).length === 1) throw badRequest('VALIDATION_FAILED', 'At least one supported field is required.')
  const changed = await db.collection<any>(collections.medicationPlans).findOneAndUpdate({
    _id: plan._id, version: expected,
  }, { $set: update, $inc: { version: 1 } }, { returnDocument: 'after' })
  if (!changed) throw conflict('VERSION_CONFLICT', 'The medication plan changed. Refresh and retry.')
  await appendOutbox(db, { eventType: 'medication_plan.changed', tenantId: principal.tenantId, patientId: String(plan.patientId),
    aggregateType: 'medication_plan', aggregateId: String(plan._id), correlationId: request.requestId, payload: { change: 'updated' } })
  sendData(response, serializeMedicationPlan(changed))
}))

carePlanRouter.get('/patients/:patientId/reminder-occurrences', requireCarePlanActor, asyncHandler(async (request, response) => {
  const patientId = request.params.patientId
  await authorizePatient(request, patientId, 'care_plan:read')
  const from = isoDate(request.query.from, 'from')
  const to = isoDate(request.query.to, 'to')
  if (to <= from || to.getTime() - from.getTime() > 32 * 24 * 60 * 60 * 1000) throw badRequest('INVALID_TIME_WINDOW', 'Reminder window must be positive and no longer than 32 days.')
  const principal = getPrincipal(request)
  const db = await getDb()
  const rows = await db.collection<any>(collections.reminderOccurrences).find({
    tenantId: principal.tenantId, patientId, scheduledAt: { $gte: from, $lt: to },
  }).sort({ scheduledAt: 1 }).toArray()
  sendData(response, rows.map(serializeOccurrence))
}))

carePlanRouter.post('/reminder-occurrences/:occurrenceId/responses', requireCarePlanActor, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/reminder-occurrences/:occurrenceId/responses', async () => {
    const principal = getPrincipal(request)
    const db = await getDb()
    const occurrence = await db.collection<any>(collections.reminderOccurrences).findOne({ _id: request.params.occurrenceId, tenantId: principal.tenantId })
    if (!occurrence) throw notFound('Reminder occurrence')
    await authorizePatient(request, String(occurrence.patientId), principal.kind === 'device' ? 'reminder:respond' : 'care_plan:read')
    const body = objectBody(request.body)
    const status = enumValue(body.status, 'status', ['taken', 'skipped', 'snoozed', 'unknown'] as const)
    const respondedAt = isoDate(body.respondedAt, 'respondedAt')
    const note = optionalString(body, 'note', 500)
    const changed = await db.collection<any>(collections.reminderOccurrences).findOneAndUpdate({
      _id: occurrence._id, status: { $nin: ['cancelled'] },
    }, { $set: { status, respondedAt, response: { note, actorType: principal.kind, actorId: principal.subjectId }, updatedAt: new Date() } }, { returnDocument: 'after' })
    if (!changed) throw conflict('REMINDER_NOT_RESPONDABLE', 'This reminder cannot be updated.')
    return { status: 200, data: serializeOccurrence(changed) }
  })
  sendData(response, result.data, result.status)
}))

carePlanRouter.post('/patients/:patientId/caregiver-tasks', requireCarePlanActor, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/patients/:patientId/caregiver-tasks', async () => {
    const principal = getPrincipal(request)
    const patientId = request.params.patientId
    await authorizePatient(request, patientId, principal.kind === 'device' ? 'session:write' : 'care_plan:write')
    const body = objectBody(request.body)
    const task = {
      _id: newId('task'), tenantId: principal.tenantId, patientId,
      category: enumValue(body.category, 'category', ['follow_up', 'appointment', 'medication_review', 'technical', 'custom'] as const),
      priority: enumValue(body.priority, 'priority', ['routine', 'elevated', 'urgent'] as const),
      title: requiredString(body, 'title', 160), details: optionalString(body, 'details', 1000),
      sourceRef: optionalString(body, 'sourceRef', 160), dueAt: body.dueAt ? isoDate(body.dueAt, 'dueAt') : null,
      status: 'open', createdBy: { type: principal.kind, id: principal.subjectId }, createdAt: new Date(),
    }
    const db = await getDb()
    await db.collection<any>(collections.caregiverTasks).insertOne(task)
    return { status: 201, data: { taskId: task._id, ...task } }
  })
  sendData(response, result.data, result.status)
}))

function serializeCarePlan(plan: Record<string, unknown>) {
  return { carePlanId: plan._id, patientId: plan.patientId, version: plan.version, status: plan.status,
    effectiveFrom: plan.effectiveFrom, effectiveTo: plan.effectiveTo, dailyRoutine: plan.dailyRoutine,
    communicationPreferences: plan.communicationPreferences, safetyNotes: plan.safetyNotes }
}

function serializeMedicationPlan(plan: Record<string, unknown>) {
  return { planId: plan._id, patientId: plan.patientId, displayName: plan.displayName, instructions: plan.instructions,
    schedule: plan.schedule, source: plan.source, status: plan.status, version: plan.version }
}

function serializeOccurrence(item: Record<string, unknown>) {
  return { occurrenceId: item._id, patientId: item.patientId, scheduledAt: item.scheduledAt, type: item.type,
    displayText: item.displayText, status: item.status, respondedAt: item.respondedAt || null }
}

function validateSchedule(value: unknown) {
  const body = objectBody(value)
  const timezone = validateTimezone(requiredString(body, 'timezone', 80))
  if (!Array.isArray(body.times) || !body.times.length || body.times.some((time) => typeof time !== 'string' || !/^(?:[01]\d|2[0-3]):[0-5]\d$/.test(time))) {
    throw badRequest('VALIDATION_FAILED', 'schedule.times must contain valid 24-hour HH:mm values.')
  }
  return { timezone, times: [...new Set(body.times)].sort(), recurrence: optionalString(body, 'recurrence', 200) || 'daily' }
}

function validObject(value: unknown, field: string) {
  if (value === undefined) return {}
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw badRequest('VALIDATION_FAILED', `${field} must be an object.`)
  return value
}

function validateTimezone(timezone: string) {
  try { new Intl.DateTimeFormat('en', { timeZone: timezone }).format() } catch { throw badRequest('INVALID_TIMEZONE', 'timezone must be a valid IANA timezone.') }
  return timezone
}

function parseIfMatch(value?: string) {
  const version = Number(value?.replace(/^W\//, '').replaceAll('"', ''))
  if (!Number.isInteger(version) || version < 1) throw badRequest('IF_MATCH_REQUIRED', 'If-Match must contain the current integer version.')
  return version
}
