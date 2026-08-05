import { Router } from 'express'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { getDb } from '../../lib/mongo.js'
import { authorizePatient, getPrincipal, requireActor } from '../platform/auth.js'
import { collections } from '../platform/collections.js'
import { badRequest } from '../platform/errors.js'
import { sendData } from '../platform/http.js'
import { newId } from '../platform/ids.js'
import { executeIdempotent } from '../platform/idempotency.js'
import { appendOutbox } from '../platform/outbox.js'
import { objectBody, stringArray } from '../platform/validation.js'
import { DAILY_CHECKIN_CONSENT_PURPOSE, publicConsentStatus } from '../platform/consent.js'

/**
 * Product privacy contract. Consent is separate from optional research participation, and a caregiver
 * can request deletion of selected data without receiving a pretend synchronous success while the queue
 * is still working. The worker owns the destructive operation and records its result for the app to read.
 */
export const privacyRouter = Router()
const requireHuman = requireActor('human')
const DELETION_CATEGORIES = ['sessions', 'messages', 'routine-responses', 'device-events'] as const

privacyRouter.get('/patients/:patientId/privacy', requireHuman, asyncHandler(async (request, response) => {
  const patientId = request.params.patientId
  await authorizePatient(request, patientId, 'patient:read')
  const principal = getPrincipal(request)
  const db = await getDb()
  const [consents, deletionRequests] = await Promise.all([
    db.collection<any>(collections.consents).find({ tenantId: principal.tenantId, patientId }).sort({ createdAt: -1 }).limit(50).toArray(),
    db.collection<any>(collections.dataDeletionRequests).find({ tenantId: principal.tenantId, patientId }).sort({ createdAt: -1 }).limit(20).toArray(),
  ])
  const latestConsent = consents.find((row) => row.purpose === DAILY_CHECKIN_CONSENT_PURPOSE)
  const consentStatus = publicConsentStatus(latestConsent)
  sendData(response, {
    patientId,
    consent: {
      status: consentStatus,
      requiredPurpose: DAILY_CHECKIN_CONSENT_PURPOSE,
      history: consents.map(serializeConsent),
    },
    research: { status: 'separate', message: 'Research participation is optional and is not required for Reflexion care.' },
    retention: {
      structuredData: 'Kept while your account is active, unless you request selected deletion.',
      sessionMedia: 'Stored only when the device uploads it and removed with selected session deletion.',
      operationalLogs: 'Restricted operational records may be retained for security and reliability.',
      configuredByServer: true,
    },
    deletionCategories: DELETION_CATEGORIES.map((category) => ({ category, label: deletionLabel(category) })),
    deletionRequests: deletionRequests.map(serializeDeletionRequest),
  })
}))

privacyRouter.post('/patients/:patientId/data-deletion-requests', requireHuman, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/patients/:patientId/data-deletion-requests', async () => {
    const patientId = request.params.patientId
    await authorizePatient(request, patientId, 'patient:write')
    const body = objectBody(request.body)
    if (body.confirm !== true) throw badRequest('CONFIRMATION_REQUIRED', 'Set confirm to true to request deletion.')
    const categories = [...new Set(stringArray(body.categories, 'categories', DELETION_CATEGORIES.length))]
    if (!categories.length || categories.some((category) => !(DELETION_CATEGORIES as readonly string[]).includes(category))) {
      throw badRequest('VALIDATION_FAILED', `categories must contain one or more of: ${DELETION_CATEGORIES.join(', ')}.`)
    }
    const principal = getPrincipal(request)
    const requestKey = request.header('Idempotency-Key')!.trim()
    if (principal.kind !== 'human') throw badRequest('HUMAN_REQUIRED', 'A caregiver account is required.')
    const now = new Date()
    const deletionRequest = {
      _id: newId('del'), tenantId: principal.tenantId, patientId, requestedBy: principal.userId,
      requestKey, categories, state: 'queued', createdAt: now, updatedAt: now,
    }
    const db = await getDb()
    await db.collection<any>(collections.dataDeletionRequests).insertOne(deletionRequest)
    await appendOutbox(db, {
      eventType: 'data.deletion.requested', tenantId: principal.tenantId, patientId,
      aggregateType: 'data_deletion_request', aggregateId: deletionRequest._id, correlationId: request.requestId,
      payload: { categories },
    })
    return { status: 202, data: serializeDeletionRequest(deletionRequest) }
  })
  sendData(response, result.data, result.status)
}))

function serializeConsent(row: Record<string, any>) {
  return { consentId: row._id, purpose: row.purpose, documentVersion: row.documentVersion, status: row.status, signedAt: row.signedAt || null, withdrawnAt: row.withdrawnAt || null }
}

export function serializeDeletionRequest(row: Record<string, any>) {
  return {
    requestId: row._id, categories: Array.isArray(row.categories) ? row.categories : [], state: row.state,
    createdAt: new Date(row.createdAt).toISOString(), updatedAt: new Date(row.updatedAt || row.createdAt).toISOString(),
    remainingObjectKeys: Array.isArray(row.remainingObjectKeys) ? row.remainingObjectKeys : [],
    error: row.error || null,
  }
}

function deletionLabel(category: typeof DELETION_CATEGORIES[number]) {
  switch (category) {
    case 'sessions': return 'Conversation sessions and transcripts'
    case 'messages': return 'Caregiver messages'
    case 'routine-responses': return 'Routine reminder responses'
    case 'device-events': return 'Technical device events'
  }
}

export { DELETION_CATEGORIES }
