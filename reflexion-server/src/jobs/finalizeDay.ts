import 'dotenv/config'
import type { Db } from 'mongodb'
import { withMongo } from '../lib/mongo.js'
import { collections } from '../v1/platform/collections.js'
import { newId } from '../v1/platform/ids.js'
import {
  dailyCheckNotificationCopy,
  materializeNotifications,
  notificationRecipients,
  type DailyCheckNotificationType,
} from '../v1/notifications/service.js'
import { computeCaregiverStatus } from '../v1/routes/monitoring.js'

// v1 timezone-aware scheduled jobs (doc "Signal-to-Status Algorithm" §12 + §16). Two responsibilities:
//   evaluate7pm   — catch a not-yet-completed day at 19:00 local and queue a deduped amber/red notice.
//   finalizeDay   — at ~23:59 local, write the authoritative daily_statuses row + update missed streak.
// Both reuse computeCaregiverStatus so the persisted record and the live read model can never diverge.
// Real cron registration is deploy-time (implementation baseline §6); this module is idempotent so a
// once-a-minute supervisor loop is safe.

const DEFAULT_TZ = process.env.DEFAULT_TIMEZONE || 'Asia/Singapore'
type NotificationType = DailyCheckNotificationType

function colorFor(status: string): 'green' | 'amber' | 'red' | null {
  return status === 'doing_well' ? 'green' : status === 'worth_checking' ? 'amber' : status === 'needs_attention' ? 'red' : null
}

function localParts(now: Date, timezone: string) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: timezone, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false,
  }).formatToParts(now)
  const value = Object.fromEntries(parts.map((part) => [part.type, part.value])) as Record<string, string>
  return { date: `${value.year}-${value.month}-${value.day}`, hour: Number(value.hour), minute: Number(value.minute) }
}

async function activePatients(db: Db) {
  return db.collection<any>(collections.patients).find({ status: { $ne: 'archived' } })
    .project({ _id: 1, tenantId: 1, timezone: 1, displayName: 1 }).toArray()
}

// Deduplicates on (patient, localDate, type) per doc §13.2 — at most one notification of each type per
// recipient per day, while still allowing different types (e.g. technical_issue and late_completion) to
// fire. Fans out to every caregiver holding `monitoring:read` on the patient: the read model filters on
// {tenantId, recipientUserId}, so a row without a recipient is invisible to every client.
async function queueNotification(
  db: Db, patient: any, recipientUserIds: string[], localDate: string,
  type: NotificationType, status: string, reason: string,
) {
  const tenantId = String(patient.tenantId)
  const patientId = String(patient._id)
  if (!recipientUserIds.length) return false
  const copy = dailyCheckNotificationCopy(type, patient.displayName)
  const dedupeKey = `${patientId}:${localDate}:${type}`
  const created = await materializeNotifications(db, {
    tenantId,
    patientId,
    recipientUserIds,
    type,
    title: copy.title,
    body: copy.body,
    dedupeKey,
    source: { type: 'daily_check', id: dedupeKey },
    // Analytics/debug fields the status engine writes alongside the caregiver-facing copy.
    extra: { statusAtSend: colorFor(status), reason, localDate },
    // Deliver the missed/technical/streak alert to the caregiver's phone, not just the in-app feed.
    push: true,
  })
  return created > 0
}

export async function evaluate7pm(db: Db, options: { patientId?: string; now?: Date } = {}) {
  const now = options.now || new Date()
  const patients = options.patientId
    ? await db.collection<any>(collections.patients).find({ _id: options.patientId }).toArray()
    : await activePatients(db)
  const queued: string[] = []
  // Patients whose alert had nowhere to go — no active care relationship carries `monitoring:read`, so no
  // caregiver can ever see it. Surfaced so "my caregiver got no alert" is diagnosable without a Mongo dig.
  const withoutRecipient: string[] = []
  for (const patient of patients) {
    const timezone = String(patient.timezone || DEFAULT_TZ)
    const local = localParts(now, timezone)
    if (local.hour < 19) continue
    const status = await computeCaregiverStatus(String(patient.tenantId), String(patient._id), timezone)
    if (status.completedToday) continue // completion notice already handled on session end
    if (status.awayActive) continue // away days never notify (doc §6.5)
    let type: NotificationType
    let reason: string
    if (status.technicalState === 'unreachable') { type = 'technical_issue'; reason = 'MIRROR_OFFLINE_OR_UNREACHABLE' }
    else if (status.missedStreak >= 3) { type = 'red_missed_streak'; reason = 'CHECKIN_MISSED_3_DAYS' }
    else { type = 'missed_7pm'; reason = 'CHECKIN_MISSED_TODAY' }
    const recipientUserIds = await notificationRecipients(db, String(patient.tenantId), String(patient._id))
    if (!recipientUserIds.length) {
      withoutRecipient.push(`${patient._id}:${type}`)
      continue
    }
    if (await queueNotification(db, patient, recipientUserIds, status.localDate, type, status.status, reason)) {
      queued.push(`${patient._id}:${type}`)
    }
  }
  if (withoutRecipient.length) {
    console.warn('[evaluate7pm] no caregiver holds monitoring:read for', withoutRecipient.join(', '))
  }
  return { queued, withoutRecipient }
}

export async function finalizeDay(db: Db, options: { patientId?: string; now?: Date; force?: boolean } = {}) {
  const now = options.now || new Date()
  const patients = options.patientId
    ? await db.collection<any>(collections.patients).find({ _id: options.patientId }).toArray()
    : await activePatients(db)
  const finalized: string[] = []
  for (const patient of patients) {
    const timezone = String(patient.timezone || DEFAULT_TZ)
    const local = localParts(now, timezone)
    // Finalize only at end-of-day (23:xx) or the early-morning catch-up window, unless forced (tests).
    if (!options.force && !(local.hour === 23 || local.hour < 5)) continue
    const status = await computeCaregiverStatus(String(patient.tenantId), String(patient._id), timezone)
    const dailyStatus = status.awayActive ? 'away'
      : status.completedToday ? 'completed'
        : status.technicalState === 'unreachable' ? 'technical_issue' : 'missed'
    const finalStatus = dailyStatus === 'away' ? null : colorFor(status.status)
    const row = {
      tenantId: patient.tenantId, patientId: String(patient._id), localDate: status.localDate, timezone,
      dailyStatus, completedByMidnight: status.completedToday,
      missedStreakAfterToday: status.completedToday ? 0 : status.missedStreak,
      finalStatus, primaryReason: finalStatus ? status.primaryReason : null, secondaryReasons: status.secondaryReasons,
      ruleVersion: status.ruleVersion, metricEvaluations: status.metricEvaluations,
      finalizedAt: now, updatedAt: now,
    }
    await db.collection<any>(collections.dailyStatuses).updateOne(
      { tenantId: patient.tenantId, patientId: String(patient._id), localDate: status.localDate },
      { $set: row, $setOnInsert: { _id: newId('day'), createdAt: now } },
      { upsert: true },
    )
    finalized.push(`${patient._id}:${dailyStatus}`)
  }
  return { finalized }
}

// Runnable supervisor loop (idempotent; both functions gate on local time internally).
async function runOnce() {
  try {
    await withMongo(async (client) => {
      const db = client.db()
      await evaluate7pm(db)
      await finalizeDay(db)
    })
  } catch (error) {
    console.error('[finalizeDay] failed', error)
  }
}

if (process.argv[1] && process.argv[1].endsWith('finalizeDay.ts') || process.env.RUN_FINALIZE_JOB === '1') {
  void runOnce()
  const interval = setInterval(() => { void runOnce() }, 60_000)
  process.once('SIGINT', () => { clearInterval(interval); process.exit(0) })
  process.once('SIGTERM', () => { clearInterval(interval); process.exit(0) })
}
