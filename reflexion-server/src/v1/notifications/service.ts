import type { Db } from 'mongodb'
import { collections } from '../platform/collections.js'
import { newId } from '../platform/ids.js'

export function reviewCaseNotificationDedupeKey(caseId: string) {
  return `review_case:${caseId}`
}

/**
 * Every caregiver authorized to see a patient's monitoring signal. This is the ONLY way a notification
 * acquires a recipient: `GET /notifications` filters on `{tenantId, recipientUserId}`, so a row written
 * without one is invisible to every client (the bug that made the whole daily-alert feed unreadable).
 */
export async function notificationRecipients(db: Db, tenantId: string, patientId: string): Promise<string[]> {
  const relationships = await db.collection<any>(collections.careRelationships).find({
    tenantId,
    patientId,
    status: 'active',
    scopes: 'monitoring:read',
    $or: [{ validTo: null }, { validTo: { $gt: new Date() } }, { validTo: { $exists: false } }],
  }).project({ userId: 1 }).toArray()
  return [...new Set(relationships.map((relationship) => String(relationship.userId)).filter(Boolean))]
}

/**
 * Upserts one notification per recipient, keyed on the unique index {tenantId, recipientUserId, dedupeKey}
 * so a re-run of a job is a no-op rather than a duplicate. Returns how many rows were newly created.
 */
export async function materializeNotifications(db: Db, input: {
  tenantId: string
  patientId: string
  recipientUserIds: string[]
  type: string
  title: string
  body: string
  dedupeKey: string
  source: { type: string; id: string }
  extra?: Record<string, unknown>
  /** When true the new row is queued for a phone push (pushState:'pending'); the dispatcher delivers it.
   *  Set only on insert, so re-running a job never re-pushes an already-delivered notification. */
  push?: boolean
}): Promise<number> {
  const now = new Date()
  let created = 0
  for (const recipientUserId of input.recipientUserIds) {
    const result = await db.collection<any>(collections.notifications).updateOne({
      tenantId: input.tenantId,
      recipientUserId,
      dedupeKey: input.dedupeKey,
    }, { $setOnInsert: {
      _id: newId('notif'),
      tenantId: input.tenantId,
      recipientUserId,
      patientId: input.patientId,
      type: input.type,
      state: 'unread',
      title: input.title,
      body: input.body,
      dedupeKey: input.dedupeKey,
      source: input.source,
      // pushState gates the dispatcher: 'pending' → deliver once to the phone, 'none' → in-app only.
      pushState: input.push ? 'pending' : 'none',
      ...(input.extra || {}),
      createdAt: now,
      updatedAt: now,
    } }, { upsert: true })
    created += result.upsertedCount
  }
  return created
}

export type DailyCheckNotificationType =
  | 'completion' | 'missed_7pm' | 'red_missed_streak' | 'technical_issue' | 'late_completion'

/**
 * Caregiver-facing copy for the daily check-in feed. Wording is a product decision (CLAUDE.md): warm,
 * reassurance-first, never clinical or diagnostic, and a device problem is always framed as a connection
 * issue rather than a change in the person. The mirror/status engine decides WHICH type fires; this only
 * decides how it reads.
 */
export function dailyCheckNotificationCopy(type: DailyCheckNotificationType, displayName?: string) {
  const first = String(displayName || '').trim().split(/\s+/)[0] || ''
  const who = first || 'Your loved one'
  const whos = first ? `${first}'s` : 'their'
  switch (type) {
    case 'completion':
      return { title: `${who} checked in today`, body: 'Today\'s check-in is done.' }
    case 'late_completion':
      return { title: `${who} checked in later than usual`, body: `Today's check-in happened outside ${whos} usual time.` }
    case 'red_missed_streak':
      return { title: `Worth checking in on ${who}`, body: 'There has not been a check-in for about three days.' }
    case 'technical_issue':
      return {
        title: 'The mirror may be offline',
        body: `We cannot reach ${whos} mirror right now. This looks like a device connection issue, not a change in how they are doing.`,
      }
    case 'missed_7pm':
    default:
      return { title: 'No check-in yet today', body: `${who} has not had a check-in yet today.` }
  }
}

/** Materializes one durable in-app notification per authorized caregiver. */
export async function materializeReviewCaseNotifications(db: Db, caseId: string) {
  const reviewCase = await db.collection<any>(collections.reviewCases).findOne({ _id: caseId })
  if (!reviewCase) return { created: 0 }
  const recipientUserIds = await notificationRecipients(db, String(reviewCase.tenantId), String(reviewCase.patientId))
  const created = await materializeNotifications(db, {
    tenantId: String(reviewCase.tenantId),
    patientId: String(reviewCase.patientId),
    recipientUserIds,
    type: reviewCase.priority === 'urgent' ? 'needs_attention' : 'worth_checking',
    title: reviewCase.priority === 'urgent' ? 'This may need attention' : 'Worth checking in',
    body: 'A new review item is available for this patient.',
    dedupeKey: reviewCaseNotificationDedupeKey(caseId),
    source: { type: 'review_case', id: caseId },
    push: true,
  })
  return { created }
}

function localDateInTz(date: Date, timezone: string): string {
  const parts = new Intl.DateTimeFormat('en-CA', { timeZone: timezone, year: 'numeric', month: '2-digit', day: '2-digit' }).formatToParts(date)
  const value = Object.fromEntries(parts.map((part) => [part.type, part.value])) as Record<string, string>
  return `${value.year}-${value.month}-${value.day}`
}

/**
 * On a completed daily check-in, notify caregivers their loved one checked in today (the "daily summary
 * ready" touchpoint — completion IS the ready signal; the app fetches the summary on tap). If a
 * `missed_7pm` alert already went out for the same local day, ALSO fire a warmer late_completion follow-up
 * so a caregiver told "no check-in yet" learns it happened after all. Both are deduped per
 * (patient, localDate, type) and pushed. Companion / non-check-in sessions never notify, and a zero-turn
 * (abandoned) session is not a completion.
 */
export async function materializeSessionCompletionNotifications(db: Db, sessionId: string) {
  const session = await db.collection<any>(collections.sessions).findOne({ _id: sessionId })
  if (!session || session.type !== 'daily_checkin') return { created: 0 }
  if (Number(session.acquisition?.patientTurns || 0) < 1) return { created: 0 }
  const tenantId = String(session.tenantId)
  const patientId = String(session.patientId)
  const patient = await db.collection<any>(collections.patients).findOne({ _id: patientId }, { projection: { displayName: 1, timezone: 1 } })
  const timezone = String(session.acquisition?.timezone || patient?.timezone || 'Asia/Singapore')
  const when = session.localCompletedAt ? new Date(session.localCompletedAt)
    : session.processingSummary?.completedAt ? new Date(session.processingSummary.completedAt) : new Date()
  const localDate = localDateInTz(when, timezone)
  const recipientUserIds = await notificationRecipients(db, tenantId, patientId)
  if (!recipientUserIds.length) return { created: 0 }
  const displayName = patient?.displayName
  let created = 0
  const completion = dailyCheckNotificationCopy('completion', displayName)
  created += await materializeNotifications(db, {
    tenantId, patientId, recipientUserIds, type: 'completion',
    title: completion.title, body: completion.body,
    dedupeKey: `${patientId}:${localDate}:completion`,
    source: { type: 'daily_check', id: `${patientId}:${localDate}:completion` },
    extra: { localDate, channel: 'push' }, push: true,
  })
  // Warmer follow-up only when we already told the caregiver today was missed.
  const missedAlreadySent = await db.collection<any>(collections.notifications).findOne({
    tenantId, patientId, dedupeKey: `${patientId}:${localDate}:missed_7pm`,
  })
  if (missedAlreadySent) {
    const late = dailyCheckNotificationCopy('late_completion', displayName)
    created += await materializeNotifications(db, {
      tenantId, patientId, recipientUserIds, type: 'late_completion',
      title: late.title, body: late.body,
      dedupeKey: `${patientId}:${localDate}:late_completion`,
      source: { type: 'daily_check', id: `${patientId}:${localDate}:late_completion` },
      extra: { localDate, channel: 'push' }, push: true,
    })
  }
  return { created }
}
