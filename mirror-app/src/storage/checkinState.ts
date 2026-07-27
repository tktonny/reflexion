import AsyncStorage from '@react-native-async-storage/async-storage'

// Tracks whether TODAY's daily check-in has already been completed, so the mirror can decide the
// conversation kind ITSELF instead of making the elder choose. Rule (product):
//   - the FIRST conversation of the day (before today's check-in is done) → check-in
//   - a long-press → always a check-in (force), even if one was already done today
//   - otherwise → companion (free chat)
// The "day" is bounded by the elder's WAKE time, not midnight: a conversation before wake-up belongs to
// the PREVIOUS day's cycle (5am when wake=7am is still "yesterday"). Only a COMPLETED check-in with real
// turns marks the day done — an abandoned/0-turn attempt never blocks it.

const LAST_CHECKIN_DATE_KEY = 'reflexion:lastCheckinLocalDate'

// Default "new day" boundary when the patient's usual wake time is unknown — 4am, so a very-early-morning
// conversation still rolls into the previous day (few elders are up before 4am).
export const DEFAULT_WAKE_HOUR = 4

function localDateKey(date: Date): string {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`
}

/** The check-in "day" key, bounded by the elder's WAKE hour: before wake-up rolls to the previous day. */
export function checkinDayKey(wakeHour: number = DEFAULT_WAKE_HOUR, date = new Date()): string {
  const d = new Date(date.getTime())
  if (d.getHours() < wakeHour) d.setDate(d.getDate() - 1)
  return localDateKey(d)
}

/** Parse "HH:MM" / "H:MM" to an hour 0-23; falls back to DEFAULT_WAKE_HOUR when absent/unparseable. */
export function wakeHourFrom(usualWakeTime?: string | null): number {
  if (!usualWakeTime) return DEFAULT_WAKE_HOUR
  const match = /^(\d{1,2}):(\d{2})$/.exec(usualWakeTime.trim())
  const hour = match ? Number(match[1]) : NaN
  return Number.isInteger(hour) && hour >= 0 && hour <= 23 ? hour : DEFAULT_WAKE_HOUR
}

/** True once a check-in has been COMPLETED (real turns) within the current wake-bounded day. */
export async function isCheckinDoneToday(wakeHour: number = DEFAULT_WAKE_HOUR): Promise<boolean> {
  try {
    return (await AsyncStorage.getItem(LAST_CHECKIN_DATE_KEY)) === checkinDayKey(wakeHour)
  } catch {
    return false // fail open → treat as "not done" so the first conversation runs the check-in
  }
}

/** Record today's check-in as complete. Call only when a screening session finalized with real turns. */
export async function markCheckinDoneToday(wakeHour: number = DEFAULT_WAKE_HOUR): Promise<void> {
  try {
    await AsyncStorage.setItem(LAST_CHECKIN_DATE_KEY, checkinDayKey(wakeHour))
  } catch {
    // best-effort; a failed write just means the next conversation may re-run the check-in.
  }
}
