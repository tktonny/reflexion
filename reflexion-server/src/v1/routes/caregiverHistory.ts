import { Router } from 'express'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { getDb } from '../../lib/mongo.js'
import {
  getSingaporeDayBoundsFromKey,
  getSingaporeDayOfMonth,
  getSingaporeMonthBounds,
} from '../../lib/dates.js'
import {
  getV1DailyStats,
  getV1SessionsForPatientRange,
  getV1TurnsBySession,
  isV1SessionCompleted,
  serializeV1Session,
} from '../../lib/v1Conversations.js'
import { authorizePatient, requireActor } from '../platform/auth.js'
import { badRequest } from '../platform/errors.js'
import { sendData } from '../platform/http.js'

/**
 * The session history a caregiver browses: a month calendar, one day in detail, and a duration trend.
 *
 * These three read models existed only as LEGACY routes (/conversation-session-counts,
 * /conversation-sessions-by-day, /patient-trend), which is why the caregiver app could not leave the legacy
 * API even after its status and alerts moved to v1. The computation is NOT reinvented here: those legacy
 * routes already derive everything from v1 `sessions` and `transcript_turns` through lib/v1Conversations.ts,
 * so this moves the same helpers behind v1 auth and the {data, meta} envelope.
 *
 * The difference that matters is authorization. The legacy routes are tokenless — identity is a patient id in
 * a query string, so anyone holding an id could read a stranger's transcripts. Every route here goes through
 * authorizePatient, which requires an active care relationship carrying the scope.
 */
export const caregiverHistoryRouter = Router()
const requireHuman = requireActor('human')

const MONTH_PATTERN = /^\d{4}-\d{2}$/
const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/
const TREND_RANGES = [7, 30] as const

/** Per-day session counts for one calendar month, in the patient's local calendar. */
caregiverHistoryRouter.get('/patients/:patientId/session-days', requireHuman, asyncHandler(async (request, response) => {
  const patientId = request.params.patientId
  await authorizePatient(request, patientId, 'monitoring:read')
  const month = typeof request.query.month === 'string' ? request.query.month : ''
  if (!MONTH_PATTERN.test(month)) throw badRequest('VALIDATION_FAILED', 'month must be formatted YYYY-MM.')

  const { start, end, daysInMonth } = getSingaporeMonthBounds(month)
  const sessions = await getV1SessionsForPatientRange(await getDb(), patientId, start, end)

  const counts = new Map<number, { count: number; completedCount: number }>()
  for (let day = 1; day <= daysInMonth; day++) counts.set(day, { count: 0, completedCount: 0 })
  for (const session of sessions) {
    if (!session.createdAt) continue
    const day = getSingaporeDayOfMonth(session.createdAt)
    const bucket = counts.get(day)
    if (!bucket) continue
    bucket.count += 1
    if (isV1SessionCompleted(session)) bucket.completedCount += 1
  }

  sendData(response, {
    patientId,
    month,
    days: Array.from({ length: daysInMonth }, (_, index) => {
      const day = index + 1
      const bucket = counts.get(day) as { count: number; completedCount: number }
      return {
        date: `${month}-${String(day).padStart(2, '0')}`,
        day,
        count: bucket.count,
        completedCount: bucket.completedCount,
        hasCompletedSession: bucket.completedCount > 0,
      }
    }),
  })
}))

/**
 * Every session on one local day, with its transcript.
 *
 * Gated on `session:read` rather than `monitoring:read`: this returns what the person actually said, which is
 * a materially more sensitive thing to hand out than a status colour.
 */
caregiverHistoryRouter.get('/patients/:patientId/session-days/:date', requireHuman, asyncHandler(async (request, response) => {
  const patientId = request.params.patientId
  const date = request.params.date
  if (!DATE_PATTERN.test(date)) throw badRequest('VALIDATION_FAILED', 'date must be formatted YYYY-MM-DD.')
  const patient = await authorizePatient(request, patientId, 'session:read')

  const { start, end } = getSingaporeDayBoundsFromKey(date)
  const db = await getDb()
  const sessions = await getV1SessionsForPatientRange(db, patientId, start, end)
  const turnsBySession = await getV1TurnsBySession(db, sessions.map((session) => session._id))
  const displayName = String(patient.displayName || '')

  sendData(response, {
    patientId,
    date,
    patientName: displayName,
    sessions: sessions.map((session) => serializeV1Session(
      session,
      turnsBySession.get(session._id) || [],
      patientId,
      displayName,
      String(patient.preferredLanguage || ''),
    )),
  })
}))

/**
 * Daily conversation duration over the last 7 or 30 local days, oldest first.
 *
 * `missed` means no COMPLETED session that day — the same definition the legacy trend used, kept identical so
 * the client's chart does not change meaning as it migrates.
 */
caregiverHistoryRouter.get('/patients/:patientId/session-trend', requireHuman, asyncHandler(async (request, response) => {
  const patientId = request.params.patientId
  await authorizePatient(request, patientId, 'monitoring:read')
  const requested = Number(typeof request.query.days === 'string' ? request.query.days : '7')
  if (!TREND_RANGES.includes(requested as never)) {
    throw badRequest('VALIDATION_FAILED', `days must be one of ${TREND_RANGES.join(', ')}.`)
  }

  const db = await getDb()
  const today = new Date()
  const dates: string[] = []
  for (let offset = requested - 1; offset >= 0; offset--) {
    const day = new Date(today.getTime() - offset * 86_400_000)
    dates.push(getSingaporeDayKey(day))
  }

  const trend = await Promise.all(dates.map(async (date) => {
    const { start, end } = getSingaporeDayBoundsFromKey(date)
    const stats = await getV1DailyStats(db, patientId, start, end)
    const missed = stats.completedSessionCount === 0
    return { date, duration: stats.duration, sessionCount: stats.sessionCount, missed }
  }))

  sendData(response, { patientId, days: requested, trend })
}))

/** Local (Asia/Singapore) YYYY-MM-DD, matching how the server buckets a patient's day everywhere else. */
function getSingaporeDayKey(date: Date): string {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Asia/Singapore', year: 'numeric', month: '2-digit', day: '2-digit',
  }).formatToParts(date)
  const value = Object.fromEntries(parts.map((part) => [part.type, part.value])) as Record<string, string>
  return `${value.year}-${value.month}-${value.day}`
}
