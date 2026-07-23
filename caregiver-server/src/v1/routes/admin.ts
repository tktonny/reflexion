import { Router } from 'express'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { getDb, inTransaction } from '../../lib/mongo.js'
import { getPrincipal, requireActor, requireHumanRole } from '../platform/auth.js'
import { collections } from '../platform/collections.js'
import { badRequest, forbidden, notFound } from '../platform/errors.js'
import { sendData, sendPage } from '../platform/http.js'
import { newId } from '../platform/ids.js'
import { executeIdempotent } from '../platform/idempotency.js'
import { enumValue, objectBody, optionalString, pagination, requiredString } from '../platform/validation.js'

// Admin / Onboarding backend (doc 1.1 — the third component). Reuses the v1 platform: same MongoDB,
// same JWT auth, same tenant model. Admin endpoints require an operator/tenant_admin human; support
// endpoints are usable by any authenticated caregiver so the two sides of a support conversation meet.
export const adminRouter = Router()

const requireHuman = requireActor('human')
const requireAdmin = requireHumanRole('operator', 'tenant_admin')
const THREAD_STATUSES = ['open', 'closed'] as const

function validateTimezone(timezone: string) {
  try { new Intl.DateTimeFormat('en', { timeZone: timezone }).format() } catch { throw badRequest('INVALID_TIMEZONE', 'timezone must be a valid IANA timezone.') }
  return timezone
}

// Every route here is gated by requireActor('human'); this narrows the principal union so userId is safe.
function humanPrincipal(request: Parameters<typeof getPrincipal>[0]) {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw forbidden()
  return principal
}

// --- Admin overview ---------------------------------------------------------
adminRouter.get('/admin/overview', requireHuman, requireAdmin, asyncHandler(async (request, response) => {
  const { tenantId } = humanPrincipal(request)
  const db = await getDb()
  const [users, patients, openThreads, devices] = await Promise.all([
    db.collection<any>(collections.users).countDocuments({ tenantId, status: { $ne: 'archived' } }),
    db.collection<any>(collections.patients).countDocuments({ tenantId, status: { $ne: 'archived' } }),
    db.collection<any>(collections.supportThreads).countDocuments({ tenantId, status: 'open' }),
    db.collection<any>(collections.devices).countDocuments({ tenantId }),
  ])
  sendData(response, { users, patients, openThreads, devices })
}))

// --- Users ------------------------------------------------------------------
adminRouter.get('/admin/users', requireHuman, requireAdmin, asyncHandler(async (request, response) => {
  const { tenantId } = humanPrincipal(request)
  const { limit, cursor } = pagination(request.query as Record<string, unknown>)
  const filter: Record<string, unknown> = { tenantId }
  if (cursor) filter._id = { $lt: cursor }
  const rows = await (await getDb()).collection<any>(collections.users).find(filter).sort({ _id: -1 }).limit(limit + 1).toArray()
  const hasMore = rows.length > limit
  const page = rows.slice(0, limit).map((user) => ({
    userId: user._id, name: user.name || '', email: user.email || '', roles: user.roles || [],
    status: user.status || 'active', createdAt: user.createdAt || null,
  }))
  sendPage(response, page, hasMore ? String(rows[limit - 1]._id) : null)
}))

// --- Patients (loved ones) --------------------------------------------------
function serializeAdminPatient(patient: Record<string, any>) {
  return {
    patientId: patient._id, displayName: patient.displayName, preferredLanguage: patient.preferredLanguage,
    timezone: patient.timezone, ageBand: patient.ageBand ?? null, status: patient.status || 'active',
    version: Number(patient.version || 1), createdAt: patient.createdAt || null, updatedAt: patient.updatedAt || null,
  }
}

adminRouter.get('/admin/patients', requireHuman, requireAdmin, asyncHandler(async (request, response) => {
  const { tenantId } = humanPrincipal(request)
  const { limit, cursor } = pagination(request.query as Record<string, unknown>)
  const filter: Record<string, unknown> = { tenantId }
  if (cursor) filter._id = { $lt: cursor }
  if (typeof request.query.q === 'string' && request.query.q.trim()) {
    filter.displayName = { $regex: request.query.q.trim().slice(0, 80), $options: 'i' }
  }
  const rows = await (await getDb()).collection<any>(collections.patients).find(filter).sort({ _id: -1 }).limit(limit + 1).toArray()
  const hasMore = rows.length > limit
  sendPage(response, rows.slice(0, limit).map(serializeAdminPatient), hasMore ? String(rows[limit - 1]._id) : null)
}))

adminRouter.post('/admin/patients', requireHuman, requireAdmin, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/admin/patients', async () => {
    const principal = humanPrincipal(request)
    const body = objectBody(request.body)
    const displayName = requiredString(body, 'displayName', 120)
    const preferredLanguage = requiredString(body, 'preferredLanguage', 40)
    const timezone = validateTimezone(requiredString(body, 'timezone', 80))
    const ageBand = optionalString(body, 'ageBand', 40)
    const caregiverUserId = optionalString(body, 'caregiverUserId', 100)
    const now = new Date()
    const patient = {
      _id: newId('pat'), tenantId: principal.tenantId, displayName, preferredLanguage, timezone,
      ageBand: ageBand || null, status: 'active', version: 1, createdAt: now, updatedAt: now,
    }
    const db = await getDb()
    // Optionally link the new patient to an existing caregiver so it appears in their app immediately.
    let linkedCaregiver: string | null = null
    if (caregiverUserId) {
      const caregiver = await db.collection<any>(collections.users).findOne({ _id: caregiverUserId, tenantId: principal.tenantId })
      if (!caregiver) throw badRequest('CAREGIVER_NOT_FOUND', 'caregiverUserId does not match a user in this tenant.')
      linkedCaregiver = caregiverUserId
    }
    await inTransaction(async (transactionDb, session) => {
      await transactionDb.collection<any>(collections.patients).insertOne(patient, { session })
      if (linkedCaregiver) {
        await transactionDb.collection<any>(collections.careRelationships).insertOne({
          _id: newId('rel'), tenantId: principal.tenantId, patientId: patient._id, userId: linkedCaregiver,
          relationshipType: 'caregiver', scopes: ['patient:read', 'patient:write', 'monitoring:read', 'session:read', 'care_plan:read', 'care_plan:write'],
          status: 'active', validFrom: now, validTo: null, createdAt: now,
        }, { session })
      }
      await transactionDb.collection<any>(collections.auditEvents).insertOne({
        _id: newId('audit'), tenantId: principal.tenantId, actor: { type: 'user', id: principal.userId },
        action: 'admin.patient.created', object: { type: 'patient', id: patient._id }, outcome: 'success',
        correlationId: request.requestId, occurredAt: now,
      }, { session })
    })
    return { status: 201, data: serializeAdminPatient(patient) }
  })
  sendData(response, result.data, result.status)
}))

adminRouter.patch('/admin/patients/:patientId', requireHuman, requireAdmin, asyncHandler(async (request, response) => {
  const principal = humanPrincipal(request)
  const body = objectBody(request.body)
  const update: Record<string, unknown> = {}
  if ('displayName' in body) update.displayName = requiredString(body, 'displayName', 120)
  if ('preferredLanguage' in body) update.preferredLanguage = requiredString(body, 'preferredLanguage', 40)
  if ('timezone' in body) update.timezone = validateTimezone(requiredString(body, 'timezone', 80))
  if ('ageBand' in body) update.ageBand = optionalString(body, 'ageBand', 40) || null
  if ('status' in body) update.status = enumValue(body.status, 'status', ['active', 'paused', 'archived'] as const)
  if (!Object.keys(update).length) throw badRequest('VALIDATION_FAILED', 'At least one supported field is required.')
  update.updatedAt = new Date()
  const db = await getDb()
  const changed = await db.collection<any>(collections.patients).findOneAndUpdate(
    { _id: request.params.patientId, tenantId: principal.tenantId },
    { $set: update, $inc: { version: 1 } }, { returnDocument: 'after' },
  )
  if (!changed) throw notFound('Patient')
  sendData(response, serializeAdminPatient(changed))
}))

// --- Support conversations --------------------------------------------------
function serializeThread(thread: Record<string, any>) {
  return {
    threadId: thread._id, subject: thread.subject, status: thread.status,
    caregiverUserId: thread.caregiverUserId, caregiverName: thread.caregiverName || '',
    lastMessageAt: thread.lastMessageAt || thread.createdAt, lastMessagePreview: thread.lastMessagePreview || '',
    adminUnread: Boolean(thread.adminUnread), caregiverUnread: Boolean(thread.caregiverUnread),
    createdAt: thread.createdAt,
  }
}
function serializeMessage(message: Record<string, any>) {
  return { messageId: message._id, threadId: message.threadId, authorType: message.authorType,
    authorId: message.authorId, authorName: message.authorName || '', body: message.body, createdAt: message.createdAt }
}

async function postMessage(request: any, threadId: string, authorType: 'caregiver' | 'admin') {
  const principal = humanPrincipal(request)
  const db = await getDb()
  const thread = await db.collection<any>(collections.supportThreads).findOne({ _id: threadId, tenantId: principal.tenantId })
  if (!thread) throw notFound('Support thread')
  if (authorType === 'caregiver' && thread.caregiverUserId !== principal.userId) throw forbidden()
  const body = requiredString(objectBody(request.body), 'body', 4000)
  const now = new Date()
  const message = {
    _id: newId('msg'), tenantId: principal.tenantId, threadId, authorType, authorId: principal.userId,
    authorName: principal.userId, body, createdAt: now,
  }
  await db.collection<any>(collections.supportMessages).insertOne(message)
  await db.collection<any>(collections.supportThreads).updateOne({ _id: threadId }, { $set: {
    lastMessageAt: now, lastMessagePreview: body.slice(0, 140), updatedAt: now, status: 'open',
    // The recipient side gains an unread marker.
    ...(authorType === 'caregiver' ? { adminUnread: true } : { caregiverUnread: true }),
  } })
  return message
}

// Caregiver side (any authenticated human): open + read + reply to their own threads.
adminRouter.get('/support/threads', requireHuman, asyncHandler(async (request, response) => {
  const principal = humanPrincipal(request)
  const rows = await (await getDb()).collection<any>(collections.supportThreads)
    .find({ tenantId: principal.tenantId, caregiverUserId: principal.userId }).sort({ lastMessageAt: -1 }).limit(100).toArray()
  sendData(response, rows.map(serializeThread))
}))

adminRouter.post('/support/threads', requireHuman, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/support/threads', async () => {
    const principal = humanPrincipal(request)
    const body = objectBody(request.body)
    const subject = requiredString(body, 'subject', 160)
    const messageBody = requiredString(body, 'body', 4000)
    const now = new Date()
    const thread = {
      _id: newId('thr'), tenantId: principal.tenantId, caregiverUserId: principal.userId, caregiverName: principal.userId,
      subject, status: 'open', createdAt: now, updatedAt: now, lastMessageAt: now,
      lastMessagePreview: messageBody.slice(0, 140), adminUnread: true, caregiverUnread: false,
    }
    const db = await getDb()
    await db.collection<any>(collections.supportThreads).insertOne(thread)
    await db.collection<any>(collections.supportMessages).insertOne({
      _id: newId('msg'), tenantId: principal.tenantId, threadId: thread._id, authorType: 'caregiver',
      authorId: principal.userId, authorName: principal.userId, body: messageBody, createdAt: now,
    })
    return { status: 201, data: serializeThread(thread) }
  })
  sendData(response, result.data, result.status)
}))

adminRouter.post('/support/threads/:threadId/messages', requireHuman, asyncHandler(async (request, response) => {
  sendData(response, serializeMessage(await postMessage(request, request.params.threadId, 'caregiver')), 201)
}))

// Admin side: see every thread in the tenant, read a thread, reply, open/close.
adminRouter.get('/admin/support/threads', requireHuman, requireAdmin, asyncHandler(async (request, response) => {
  const { tenantId } = humanPrincipal(request)
  const filter: Record<string, unknown> = { tenantId }
  if (typeof request.query.status === 'string') filter.status = enumValue(request.query.status, 'status', THREAD_STATUSES)
  const rows = await (await getDb()).collection<any>(collections.supportThreads).find(filter).sort({ lastMessageAt: -1 }).limit(200).toArray()
  sendData(response, rows.map(serializeThread))
}))

adminRouter.get('/admin/support/threads/:threadId', requireHuman, requireAdmin, asyncHandler(async (request, response) => {
  const { tenantId } = humanPrincipal(request)
  const db = await getDb()
  const thread = await db.collection<any>(collections.supportThreads).findOne({ _id: request.params.threadId, tenantId })
  if (!thread) throw notFound('Support thread')
  const messages = await db.collection<any>(collections.supportMessages).find({ threadId: thread._id }).sort({ createdAt: 1 }).toArray()
  await db.collection<any>(collections.supportThreads).updateOne({ _id: thread._id }, { $set: { adminUnread: false } })
  sendData(response, { ...serializeThread(thread), messages: messages.map(serializeMessage) })
}))

adminRouter.post('/admin/support/threads/:threadId/messages', requireHuman, requireAdmin, asyncHandler(async (request, response) => {
  sendData(response, serializeMessage(await postMessage(request, request.params.threadId, 'admin')), 201)
}))

adminRouter.patch('/admin/support/threads/:threadId', requireHuman, requireAdmin, asyncHandler(async (request, response) => {
  const { tenantId } = humanPrincipal(request)
  const status = enumValue(objectBody(request.body).status, 'status', THREAD_STATUSES)
  const changed = await (await getDb()).collection<any>(collections.supportThreads).findOneAndUpdate(
    { _id: request.params.threadId, tenantId }, { $set: { status, updatedAt: new Date() } }, { returnDocument: 'after' },
  )
  if (!changed) throw notFound('Support thread')
  sendData(response, serializeThread(changed))
}))
