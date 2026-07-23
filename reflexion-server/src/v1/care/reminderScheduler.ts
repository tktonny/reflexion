import type { Db } from 'mongodb'
import { collections } from '../platform/collections.js'
import { newId } from '../platform/ids.js'

const HORIZON_DAYS = 14

export async function materializeMedicationReminders(db: Db, planId: string, now = new Date()) {
  const plan = await db.collection<any>(collections.medicationPlans).findOne({ _id: planId })
  if (!plan) return 0
  if (plan.status !== 'active') {
    await db.collection<any>(collections.reminderOccurrences).updateMany({
      tenantId: plan.tenantId, sourceId: planId, scheduledAt: { $gt: now }, status: 'scheduled',
    }, { $set: { status: 'cancelled', updatedAt: now } })
    return 0
  }
  const schedule = plan.schedule as { timezone?: string; times?: string[]; recurrence?: string } | undefined
  if (!schedule?.timezone || !Array.isArray(schedule.times) || schedule.recurrence !== 'daily') return 0
  const today = localDateParts(now, schedule.timezone)
  let created = 0
  for (let dayOffset = 0; dayOffset < HORIZON_DAYS; dayOffset++) {
    const date = new Date(Date.UTC(today.year, today.month - 1, today.day + dayOffset))
    for (const time of schedule.times) {
      const [hour, minute] = time.split(':').map(Number)
      const scheduledAt = zonedDateTimeToUtc(date.getUTCFullYear(), date.getUTCMonth() + 1, date.getUTCDate(), hour, minute, schedule.timezone)
      if (scheduledAt < now) continue
      const result = await db.collection<any>(collections.reminderOccurrences).updateOne({
        tenantId: plan.tenantId, sourceId: planId, scheduledAt,
      }, { $setOnInsert: {
        _id: newId('rem'), tenantId: plan.tenantId, patientId: plan.patientId, sourceId: planId,
        planId, scheduledAt, timezone: schedule.timezone, type: 'medication', displayText: plan.displayName,
        status: 'scheduled', createdAt: now,
      } }, { upsert: true })
      created += result.upsertedCount
    }
  }
  return created
}

export async function materializeAllMedicationReminders(db: Db) {
  const plans = await db.collection<any>(collections.medicationPlans).find({ status: { $in: ['active', 'paused', 'ended'] } }).project({ _id: 1 }).toArray()
  let created = 0
  for (const plan of plans) created += await materializeMedicationReminders(db, String(plan._id))
  return created
}

export function zonedDateTimeToUtc(year: number, month: number, day: number, hour: number, minute: number, timezone: string) {
  const wallClockAsUtc = Date.UTC(year, month - 1, day, hour, minute)
  let guess = wallClockAsUtc
  for (let iteration = 0; iteration < 2; iteration++) {
    const parts = localDateTimeParts(new Date(guess), timezone)
    const represented = Date.UTC(parts.year, parts.month - 1, parts.day, parts.hour, parts.minute)
    guess += wallClockAsUtc - represented
  }
  return new Date(guess)
}

function localDateParts(date: Date, timezone: string) {
  const parts = localDateTimeParts(date, timezone)
  return { year: parts.year, month: parts.month, day: parts.day }
}

function localDateTimeParts(date: Date, timezone: string) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: timezone, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
  }).formatToParts(date)
  const value = (type: Intl.DateTimeFormatPartTypes) => Number(parts.find((part) => part.type === type)?.value)
  return { year: value('year'), month: value('month'), day: value('day'), hour: value('hour'), minute: value('minute') }
}
