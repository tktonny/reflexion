import 'dotenv/config'
import { ObjectId, type Db } from 'mongodb'
import { DB_NAME, NURSE_CONFIG_COLLECTION, TIME_ZONE } from '../lib/constants.js'
import { getDailyStatus, refreshDailyStatusForDate } from '../lib/dailyStatus.js'
import { withMongo } from '../lib/mongo.js'
import type { StoredPatient } from '../lib/types.js'

type NurseConfig = {
  _id?: ObjectId
  preferredDailySummaryTime?: string
  patients?: StoredPatient[]
}

const intervalMs = 60 * 1000

void runOnce()
const interval = setInterval(() => {
  void runOnce()
}, intervalMs)

process.once('SIGINT', () => {
  clearInterval(interval)
  process.exit(0)
})

process.once('SIGTERM', () => {
  clearInterval(interval)
  process.exit(0)
})

async function runOnce() {
  try {
    await withMongo(async (client) => {
      await createDueDailyStatuses(client.db(DB_NAME), new Date())
    })
  } catch (error) {
    console.error('[dailySummaryCron] failed', error)
  }
}

async function createDueDailyStatuses(db: Db, now: Date) {
  const localNow = getLocalDateTime(now, TIME_ZONE)
  const configs = await db.collection<NurseConfig>(NURSE_CONFIG_COLLECTION).find({
    patients: { $exists: true, $ne: [] },
  }).project({
    _id: 1,
    patients: 1,
    preferredDailySummaryTime: 1,
  }).toArray()

  for (const config of configs) {
    const preferredDailySummaryTime = normalizeSummaryTime(config.preferredDailySummaryTime)
    if (localNow.time < preferredDailySummaryTime) {
      continue
    }

    for (const patient of config.patients || []) {
      if (!patient._id) continue

      const existing = await getDailyStatus(db, patient._id, localNow.date)
      if (existing) continue

      await refreshDailyStatusForDate(db, patient._id, localNow.date)
      console.log(
        `[dailySummaryCron] created DailyPatientStatus patient=${patient._id.toHexString()} date=${localNow.date}`,
      )
    }
  }
}

function normalizeSummaryTime(value: unknown) {
  return typeof value === 'string' && /^\d{2}:\d{2}$/.test(value) ? value : '19:00'
}

function getLocalDateTime(date: Date, timezone: string) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    day: '2-digit',
    hour: '2-digit',
    hour12: false,
    minute: '2-digit',
    month: '2-digit',
    timeZone: timezone,
    year: 'numeric',
  }).formatToParts(date)
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]))

  return {
    date: `${values.year}-${values.month}-${values.day}`,
    time: `${values.hour}:${values.minute}`,
  }
}
