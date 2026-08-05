import { createDailyConversationPlan, type DailyConversationPlan } from '../orchestration/deterministicSpeech'
import { deviceFetch } from '../storage/deviceCredentials'
import { dataOrThrow } from './devicePairing'

export type ReminderOccurrence = {
  occurrenceId: string
  patientId: string
  scheduledAt: string
  type: string
  displayText: string
  category?: string
  status: string
  respondedAt?: string | null
}

const LOOKBACK_MS = 6 * 60 * 60 * 1000
const DUE_SOON_MS = 15 * 60 * 1000
const CLOSED_STATUSES = new Set(['taken', 'skipped', 'cancelled', 'reported-complete', 'deferred', 'declined', 'no-response'])

/** Pick the closest open medication or generic routine reminder for the daily conversation. Medication
 * remains first when both are due so an older-adult check-in never loses the medication adjunct. */
export function selectDueReminderOccurrence(
  occurrences: ReminderOccurrence[],
  now = new Date(),
): ReminderOccurrence | null {
  const latestAllowed = now.getTime() + DUE_SOON_MS
  return occurrences
    .filter((item) => (
      (item.type === 'medication' || item.type === 'routine') &&
      !item.respondedAt &&
      !CLOSED_STATUSES.has(item.status) &&
      Number.isFinite(Date.parse(item.scheduledAt)) &&
      Date.parse(item.scheduledAt) <= latestAllowed &&
      item.displayText.trim().length > 0
    ))
    .sort((left, right) => {
      const typeOrder = Number(left.type !== 'medication') - Number(right.type !== 'medication')
      if (typeOrder !== 0) return typeOrder
      return Math.abs(Date.parse(left.scheduledAt) - now.getTime()) - Math.abs(Date.parse(right.scheduledAt) - now.getTime())
    })[0] ?? null
}

export function selectDueMedicationOccurrence(
  occurrences: ReminderOccurrence[],
  now = new Date(),
): ReminderOccurrence | null {
  const latestAllowed = now.getTime() + DUE_SOON_MS
  return occurrences
    .filter((item) => (
      item.type === 'medication' &&
      !item.respondedAt &&
      !CLOSED_STATUSES.has(item.status) &&
      Number.isFinite(Date.parse(item.scheduledAt)) &&
      Date.parse(item.scheduledAt) <= latestAllowed &&
      item.displayText.trim().length > 0
    ))
    .sort((left, right) => Math.abs(Date.parse(left.scheduledAt) - now.getTime()) - Math.abs(Date.parse(right.scheduledAt) - now.getTime()))[0] ?? null
}

export async function loadDailyConversationPlan(options: {
  patientId: string
  patientName?: string | null
  now?: Date
}): Promise<DailyConversationPlan> {
  const now = options.now ?? new Date()
  const fallback = createDailyConversationPlan({ patientName: options.patientName, now })
  const from = new Date(now.getTime() - LOOKBACK_MS).toISOString()
  const to = new Date(now.getTime() + DUE_SOON_MS + 1).toISOString()
  try {
    const response = await deviceFetch(
      `/api/v1/patients/${encodeURIComponent(options.patientId)}/reminder-occurrences?from=${encodeURIComponent(from)}&to=${encodeURIComponent(to)}`,
    )
    const occurrences = await dataOrThrow<ReminderOccurrence[]>(response)
    const reminder = selectDueReminderOccurrence(occurrences, now)
    if (!reminder) return fallback
    return createDailyConversationPlan({
      patientName: options.patientName,
      now,
      ...(reminder.type === 'medication'
        ? { medicationReminder: { occurrenceId: reminder.occurrenceId, displayText: reminder.displayText, scheduledAt: reminder.scheduledAt } }
        : { routineReminder: { occurrenceId: reminder.occurrenceId, displayText: reminder.displayText, scheduledAt: reminder.scheduledAt, category: reminder.category } }),
    })
  } catch {
    // A reminder lookup must never block the daily conversation. Omitting the medication question
    // is safer than guessing when trusted caregiver/provider data is unavailable.
    return fallback
  }
}
