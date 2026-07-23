import 'dotenv/config'
import type { Db } from 'mongodb'
import { withMongo } from '../lib/mongo.js'
import { collections } from '../v1/platform/collections.js'
import { newId } from '../v1/platform/ids.js'
import { computeCaregiverStatus } from '../v1/routes/monitoring.js'

// v1 timezone-aware scheduled jobs (doc "Signal-to-Status Algorithm" §12 + §16). Two responsibilities:
//   evaluate7pm   — catch a not-yet-completed day at 19:00 local and queue a deduped amber/red notice.
//   finalizeDay   — at ~23:59 local, write the authoritative daily_statuses row + update missed streak.
// Both reuse computeCaregiverStatus so the persisted record and the live read model can never diverge.
// Real cron registration is deploy-time (implementation baseline §6); this module is idempotent so a
// once-a-minute supervisor loop is safe.

const DEFAULT_TZ = process.env.DEFAULT_TIMEZONE || 'Asia/Singapore'
type NotificationType = 'completion' | 'missed_7pm' | 'red_missed_streak' | 'technical_issue' | 'late_completion'

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
    .project({ _id: 1, tenantId: 1, timezone: 1 }).toArray()
}

// Deduplicates on (patient, localDate, type) per doc §13.2 — at most one notification of each type per
// user per day, while still allowing different types (e.g. technical_issue and late_completion) to fire.
async function queueNotification(db: Db, patient: any, localDate: string, type: NotificationType, status: string, reason: string) {
  const dedupeKey = `${patient._id}:${localDate}:${type}`
  const existing = await db.collection<any>(collections.notifications).findOne({ dedupeKey })
  if (existing) return false
  await db.collection<any>(collections.notifications).insertOne({
    _id: newId('notif'), tenantId: patient.tenantId, patientId: String(patient._id), dedupeKey,
    type, statusAtSend: colorFor(status), reason, localDate, channel: 'push', state: 'queued', createdAt: new Date(),
  })
  return true
}

export async function evaluate7pm(db: Db, options: { patientId?: string; now?: Date } = {}) {
  const now = options.now || new Date()
  const patients = options.patientId
    ? await db.collection<any>(collections.patients).find({ _id: options.patientId }).toArray()
    : await activePatients(db)
  const queued: string[] = []
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
    if (await queueNotification(db, patient, status.localDate, type, status.status, reason)) queued.push(`${patient._id}:${type}`)
  }
  return { queued }
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
