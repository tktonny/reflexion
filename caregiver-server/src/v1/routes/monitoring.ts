import { Router } from 'express'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { getDb } from '../../lib/mongo.js'
import { authorizePatient, getPrincipal, requireActor, requireHumanRole } from '../platform/auth.js'
import { collections } from '../platform/collections.js'
import { badRequest, forbidden, notFound } from '../platform/errors.js'
import { sendData, sendPage } from '../platform/http.js'
import { newId } from '../platform/ids.js'
import { executeIdempotent } from '../platform/idempotency.js'
import { appendOutbox } from '../platform/outbox.js'
import { enumValue, objectBody, optionalString, pagination, requiredString } from '../platform/validation.js'
import { researchEligibility } from '../monitoring/algorithms.js'
import { zonedDateTimeToUtc } from '../care/reminderScheduler.js'

export const monitoringRouter = Router()
const requireHuman = requireActor('human')

monitoringRouter.get('/patients/:patientId/status', requireHuman, asyncHandler(async (request, response) => {
  const patient = await authorizePatient(request, request.params.patientId, 'monitoring:read')
  sendData(response, await computeCaregiverStatus(getPrincipal(request).tenantId, String(patient._id), String(patient.timezone || 'UTC')))
}))

monitoringRouter.post('/patients/:patientId/away-periods', requireHuman, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/patients/:patientId/away-periods', async () => {
    const patientId = request.params.patientId
    await authorizePatient(request, patientId, 'monitoring:read')
    const principal = getPrincipal(request)
    if (principal.kind !== 'human') throw forbidden()
    const body = objectBody(request.body)
    const startsOn = dateOnly(body.startsOn, 'startsOn')
    const endsOn = dateOnly(body.endsOn, 'endsOn')
    if (endsOn < startsOn) throw badRequest('INVALID_DATE_RANGE', 'endsOn must be on or after startsOn.')
    const away = {
      _id: newId('away'), tenantId: principal.tenantId, patientId, startsOn, endsOn,
      timezone: validTimezone(requiredString(body, 'timezone', 80)), reason: optionalString(body, 'reason', 500),
      createdBy: principal.userId, state: 'active', createdAt: new Date(),
    }
    const db = await getDb()
    await db.collection<any>(collections.awayPeriods).insertOne(away)
    return { status: 201, data: { awayPeriodId: away._id, patientId, startsOn, endsOn, timezone: away.timezone, state: away.state } }
  })
  sendData(response, result.data, result.status)
}))

monitoringRouter.post('/patients/:patientId/manual-flags', requireHuman, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/patients/:patientId/manual-flags', async () => {
    const patientId = request.params.patientId
    await authorizePatient(request, patientId, 'monitoring:read')
    const principal = getPrincipal(request)
    if (principal.kind !== 'human') throw forbidden()
    const body = objectBody(request.body)
    const flag = {
      _id: newId('flag'), tenantId: principal.tenantId, patientId,
      severity: enumValue(body.severity, 'severity', ['worth_checking', 'needs_attention'] as const),
      reason: requiredString(body, 'reason', 1000), state: 'active', createdBy: principal.userId, createdAt: new Date(),
    }
    const db = await getDb()
    await db.collection<any>(collections.manualFlags).insertOne(flag)
    return { status: 201, data: { manualFlagId: flag._id, patientId, severity: flag.severity, reason: flag.reason, state: flag.state, createdAt: flag.createdAt.toISOString() } }
  })
  sendData(response, result.data, result.status)
}))

monitoringRouter.get('/patients/:patientId/monitoring/summary', requireHuman, asyncHandler(async (request, response) => {
  const patientId = request.params.patientId
  await authorizePatient(request, patientId, 'monitoring:read')
  const principal = getPrincipal(request)
  const db = await getDb()
  const [snapshots, completedSessions, baseline, latest] = await Promise.all([
    db.collection<any>(collections.featureSnapshots).find({
      tenantId: principal.tenantId, patientId, sessionType: 'daily_checkin', inclusion: { $in: ['include', 'include_with_caveats'] },
    }).project({ capturedAt: 1 }).sort({ capturedAt: 1 }).toArray(),
    db.collection<any>(collections.sessions).find({
      tenantId: principal.tenantId, patientId, type: 'daily_checkin', localCompletedAt: { $exists: true },
    }).project({ localCompletedAt: 1 }).sort({ localCompletedAt: 1 }).toArray(),
    db.collection<any>(collections.baselineModels).findOne({ tenantId: principal.tenantId, patientId, state: 'active', baselineType: 'longitudinal_research' }, { sort: { revision: -1 } }),
    db.collection<any>(collections.monitoringWindows).findOne({ tenantId: principal.tenantId, patientId }, { sort: { windowEnd: -1 } }),
  ])
  const usableDates = snapshots.map((item) => new Date(item.capturedAt))
  const coverageWindow = usableDates.length ? completedSessions.filter((item) => {
    const completedAt = new Date(item.localCompletedAt)
    return completedAt >= usableDates[0] && completedAt <= usableDates[usableDates.length - 1]
  }) : []
  const qualityCoverage = coverageWindow.length ? Math.min(usableDates.length / coverageWindow.length, 1) : 0
  const eligibility = researchEligibility(usableDates, qualityCoverage)
  const baselineState = baseline ? 'complete' : snapshots.length ? 'building' : 'not_started'
  const displayState = String(latest?.status || 'building')
  const data: Record<string, unknown> = {
    patientId, baselineState, displayState,
    coverage: { usableSessions: eligibility.usableSessions, requiredSessions: 12, distinctWeeks: eligibility.distinctWeeks,
      spanDays: eligibility.spanDays, qualityCoverage: eligibility.qualityCoverage },
    updatedAt: new Date(latest?.updatedAt || snapshots.at(-1)?.capturedAt || Date.now()).toISOString(),
  }
  if (isProvider(principal)) data.providerDetail = {
    baselineId: baseline?._id || null, revision: baseline?.revision || null,
    latestScoreId: latest?.sourceScoreId || null, pipelineVersion: baseline?.pipelineVersion || null,
  }
  sendData(response, data)
}))

monitoringRouter.get('/patients/:patientId/monitoring/timeline', requireHuman, asyncHandler(async (request, response) => {
  const patientId = request.params.patientId
  await authorizePatient(request, patientId, 'monitoring:read')
  const principal = getPrincipal(request)
  const { limit, cursor } = pagination(request.query as Record<string, unknown>)
  const filter: Record<string, any> = { tenantId: principal.tenantId, patientId }
  if (cursor) filter._id = { $lt: cursor }
  if (typeof request.query.from === 'string' || typeof request.query.to === 'string') {
    filter.windowEnd = {}
    if (typeof request.query.from === 'string') filter.windowEnd.$gte = validDateTime(request.query.from, 'from')
    if (typeof request.query.to === 'string') filter.windowEnd.$lte = validDateTime(request.query.to, 'to')
  }
  const rows = await (await getDb()).collection<any>(collections.monitoringWindows).find(filter).sort({ _id: -1 }).limit(limit + 1).toArray()
  const hasMore = rows.length > limit
  const page = rows.slice(0, limit).map((item) => ({
    sessionId: item.sessionId, recordedAt: new Date(item.windowEnd).toISOString(), inclusion: item.inclusion,
    displayState: item.status,
    ...(isProvider(principal) ? { providerDetail: { sourceScoreId: item.sourceScoreId || null, ruleVersion: item.ruleVersion } } : {}),
  }))
  sendPage(response, page, hasMore ? String(rows[limit - 1]._id) : null)
}))

monitoringRouter.get('/patients/:patientId/monitoring/baseline', requireHuman, asyncHandler(async (request, response) => {
  const patientId = request.params.patientId
  await authorizePatient(request, patientId, 'monitoring:read')
  const principal = getPrincipal(request)
  const db = await getDb()
  const [operational, research] = await Promise.all([
    db.collection<any>(collections.operationalBaselines).findOne({
      tenantId: principal.tenantId, patientId, baselineType: 'reassurance_mvp', state: { $in: ['establishing', 'complete'] },
    }, { sort: { revision: -1 } }),
    db.collection<any>(collections.baselineModels).findOne({ tenantId: principal.tenantId, patientId, baselineType: 'longitudinal_research', state: 'active' }, { sort: { revision: -1 } }),
  ])
  const data: Record<string, unknown> = {
    patientId,
    operational: operational ? { state: operational.state, sessionCount: operational.sessionCount, window: operational.window,
      algorithmVersion: operational.algorithmVersion, revision: operational.revision } : { state: 'not_started', sessionCount: 0 },
    longitudinal: research ? { state: 'complete', eligibility: research.eligibility, revision: research.revision } : { state: 'building' },
  }
  if (isProvider(principal) && research) data.providerDetail = {
    baselineId: research._id, scalarAggregates: research.scalarAggregates,
    embedding: research.embeddingBaseline ? { family: research.embeddingBaseline.family, modelId: research.embeddingBaseline.modelId,
      dimensions: research.embeddingBaseline.dimensions, sampleCount: research.embeddingBaseline.sampleCount } : null,
    pipelineVersion: research.pipelineVersion,
  }
  sendData(response, data)
}))

monitoringRouter.get('/review-cases', requireHuman, requireHumanRole('provider', 'reviewer', 'tenant_admin'), asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw forbidden()
  const { limit, cursor } = pagination(request.query as Record<string, unknown>)
  const db = await getDb()
  let patientIds: string[] | undefined
  if (!principal.roles.includes('tenant_admin')) {
    patientIds = (await db.collection<any>(collections.careRelationships).find({
      tenantId: principal.tenantId, userId: principal.userId, status: 'active', scopes: 'review:read',
    }).project({ patientId: 1 }).toArray()).map((item) => String(item.patientId))
  }
  const filter: Record<string, any> = { tenantId: principal.tenantId, ...(patientIds ? { patientId: { $in: patientIds } } : {}) }
  if (typeof request.query.state === 'string') filter.state = enumValue(request.query.state, 'state', ['open', 'assigned', 'awaiting_follow_up', 'closed'] as const)
  if (typeof request.query.patientId === 'string') filter.patientId = request.query.patientId
  if (cursor) filter._id = { $lt: cursor }
  const rows = await db.collection<any>(collections.reviewCases).find(filter).sort({ _id: -1 }).limit(limit + 1).toArray()
  const hasMore = rows.length > limit
  const page = rows.slice(0, limit)
  sendPage(response, page.map(serializeReviewCase), hasMore ? String(page.at(-1)?._id) : null)
}))

monitoringRouter.get('/review-cases/:caseId', requireHuman, requireHumanRole('provider', 'reviewer', 'tenant_admin'), asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  const db = await getDb()
  const reviewCase = await db.collection<any>(collections.reviewCases).findOne({ _id: request.params.caseId, tenantId: principal.tenantId })
  if (!reviewCase) throw notFound('Review case')
  if (principal.kind !== 'human' || !isProvider(principal)) throw forbidden()
  if (!principal.roles.includes('tenant_admin')) await authorizePatient(request, String(reviewCase.patientId), 'review:read')
  const dispositions = await db.collection<any>(collections.reviewDispositions).find({ caseId: reviewCase._id }).sort({ createdAt: 1 }).toArray()
  sendData(response, { ...serializeReviewCase(reviewCase), dispositions: dispositions.map(serializeDisposition) })
}))

monitoringRouter.post('/review-cases/:caseId/dispositions', requireHuman, requireHumanRole('provider', 'reviewer', 'tenant_admin'), asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/review-cases/:caseId/dispositions', async () => {
    const principal = getPrincipal(request)
    if (principal.kind !== 'human') throw forbidden()
    const db = await getDb()
    const reviewCase = await db.collection<any>(collections.reviewCases).findOne({ _id: request.params.caseId, tenantId: principal.tenantId })
    if (!reviewCase) throw notFound('Review case')
    if (!principal.roles.includes('tenant_admin')) await authorizePatient(request, String(reviewCase.patientId), 'review:write')
    const body = objectBody(request.body)
    const outcome = enumValue(body.outcome, 'outcome', ['confirmed_meaningful_change', 'no_meaningful_change', 'acute_or_reversible_context',
      'poor_quality', 'wrong_identity', 'protocol_issue', 'follow_up_ordered', 'insufficient_information'] as const)
    const disposition = {
      _id: newId('disp'), tenantId: principal.tenantId, patientId: reviewCase.patientId, caseId: reviewCase._id,
      reviewerId: principal.userId, outcome, notes: optionalString(body, 'notes', 5000), createdAt: new Date(),
    }
    const closeCase = body.closeCase === true
    await db.collection<any>(collections.reviewDispositions).insertOne(disposition)
    await db.collection<any>(collections.reviewCases).updateOne({ _id: reviewCase._id }, { $set: {
      currentDisposition: { dispositionId: disposition._id, outcome, reviewerId: principal.userId },
      state: closeCase ? 'closed' : outcome === 'follow_up_ordered' ? 'awaiting_follow_up' : 'assigned', updatedAt: new Date(),
    } })
    await appendOutbox(db, { eventType: 'review_case.disposition_recorded', tenantId: principal.tenantId,
      patientId: String(reviewCase.patientId), aggregateType: 'review_case', aggregateId: String(reviewCase._id),
      correlationId: request.requestId, payload: { dispositionId: disposition._id, outcome, closeCase } })
    return { status: 201, data: serializeDisposition(disposition) }
  })
  sendData(response, result.data, result.status)
}))

async function computeCaregiverStatus(tenantId: string, patientId: string, timezone: string) {
  const db = await getDb()
  const now = new Date()
  const local = localYmd(now, timezone)
  const dayStart = zonedDateTimeToUtc(local.year, local.month, local.day, 0, 0, timezone)
  const nextDate = new Date(Date.UTC(local.year, local.month - 1, local.day + 1))
  const dayEnd = zonedDateTimeToUtc(nextDate.getUTCFullYear(), nextDate.getUTCMonth() + 1, nextDate.getUTCDate(), 0, 0, timezone)
  const [baseline, today, lastSession, assignment, flag, away] = await Promise.all([
    db.collection<any>(collections.operationalBaselines).findOne({ tenantId, patientId, baselineType: 'reassurance_mvp', state: { $in: ['establishing', 'complete'] } }, { sort: { revision: -1 } }),
    db.collection<any>(collections.sessions).findOne({ tenantId, patientId, type: 'daily_checkin', localCompletedAt: { $gte: dayStart, $lt: dayEnd } }),
    db.collection<any>(collections.sessions).findOne({ tenantId, patientId, localCompletedAt: { $exists: true } }, { sort: { localCompletedAt: -1 } }),
    db.collection<any>(collections.assignments).findOne({ tenantId, patientId, status: 'active' }),
    db.collection<any>(collections.manualFlags).findOne({ tenantId, patientId, state: 'active' }, { sort: { createdAt: -1 } }),
    db.collection<any>(collections.awayPeriods).findOne({ tenantId, patientId, state: 'active', startsOn: { $lte: ymdText(local) }, endsOn: { $gte: ymdText(local) } }),
  ])
  const device = assignment ? await db.collection<any>(collections.devices).findOne({ _id: assignment.deviceId }) : null
  const lastSeenAt = device?.lastSeenAt ? new Date(device.lastSeenAt) : null
  const technicalState = !device || !lastSeenAt ? 'unknown' : now.getTime() - lastSeenAt.getTime() > 15 * 60_000 ? 'unreachable'
    : device.technicalState === 'ok' ? 'ok' : 'possible_issue'
  const baselineState = baseline?.state === 'complete' ? 'complete' : 'establishing'
  let status: 'establishing' | 'doing_well' | 'worth_checking' | 'needs_attention' = baselineState === 'complete' ? 'doing_well' : 'establishing'
  let primaryReason = baselineState === 'complete' ? 'DAILY_PATTERN_ON_TRACK' : 'LEARNING_PERSONAL_ROUTINE'
  const secondaryReasons: string[] = []
  const daysSinceInteraction = lastSession?.localCompletedAt ? Math.floor((now.getTime() - new Date(lastSession.localCompletedAt).getTime()) / 86_400_000) : Infinity
  if (flag?.severity === 'needs_attention') { status = 'needs_attention'; primaryReason = 'CAREGIVER_FLAG_NEEDS_ATTENTION' }
  else if (flag?.severity === 'worth_checking') { status = 'worth_checking'; primaryReason = 'CAREGIVER_FLAG_WORTH_CHECKING' }
  else if (!away && baselineState === 'complete' && daysSinceInteraction >= 2) { status = 'worth_checking'; primaryReason = 'CHECKIN_MISSED_REPEATEDLY' }
  else if (technicalState === 'unreachable') { status = baselineState === 'complete' ? 'worth_checking' : 'establishing'; primaryReason = 'DEVICE_UNREACHABLE' }
  if (today) secondaryReasons.push('CHECKIN_COMPLETED_TODAY')
  if (away) secondaryReasons.push('AWAY_PERIOD_ACTIVE')
  return {
    patientId, baselineState, baselineProgress: { completedSessions: Number(baseline?.sessionCount || 0), requiredSessions: 7, windowDays: 14 },
    status, primaryReason, secondaryReasons, completedToday: Boolean(today), technicalState,
    lastInteractionAt: lastSession?.localCompletedAt ? new Date(lastSession.localCompletedAt).toISOString() : null,
    updatedAt: now.toISOString(),
  }
}

function serializeReviewCase(item: Record<string, unknown>) {
  return { caseId: item._id, patientId: item.patientId, reason: item.reason, priority: item.priority,
    state: item.state, sourceRefs: item.sourceRefs, createdAt: item.createdAt }
}
function serializeDisposition(item: Record<string, unknown>) {
  return { dispositionId: item._id, caseId: item.caseId, reviewerId: item.reviewerId, outcome: item.outcome, createdAt: item.createdAt }
}
function isProvider(principal: ReturnType<typeof getPrincipal>): boolean {
  return principal.kind === 'human' && principal.roles.some((role) => ['provider', 'reviewer', 'tenant_admin'].includes(role))
}
function dateOnly(value: unknown, field: string) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value) || Number.isNaN(Date.parse(`${value}T00:00:00Z`))) throw badRequest('VALIDATION_FAILED', `${field} must be YYYY-MM-DD.`)
  return value
}
function validDateTime(value: string, field: string) {
  const date = new Date(value); if (Number.isNaN(date.getTime())) throw badRequest('VALIDATION_FAILED', `${field} must be an ISO date-time.`); return date
}
function validTimezone(timezone: string) {
  try { new Intl.DateTimeFormat('en', { timeZone: timezone }).format() } catch { throw badRequest('INVALID_TIMEZONE', 'timezone must be a valid IANA timezone.') }
  return timezone
}
function localYmd(date: Date, timezone: string) {
  const parts = new Intl.DateTimeFormat('en-CA', { timeZone: timezone, year: 'numeric', month: '2-digit', day: '2-digit' }).formatToParts(date)
  const get = (type: Intl.DateTimeFormatPartTypes) => Number(parts.find((part) => part.type === type)?.value)
  return { year: get('year'), month: get('month'), day: get('day') }
}
function ymdText(value: { year: number; month: number; day: number }) { return `${value.year}-${String(value.month).padStart(2, '0')}-${String(value.day).padStart(2, '0')}` }
