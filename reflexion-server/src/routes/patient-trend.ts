import { Router } from 'express'
import { ObjectId, type Db } from 'mongodb'
import { asyncHandler } from '../lib/asyncHandler.js'
import { DB_NAME, NURSE_CONFIG_COLLECTION, TIME_ZONE } from '../lib/constants.js'
import { getSingaporeDateKey, getSingaporeDayBoundsFromKey } from '../lib/dates.js'
import { withMongo } from '../lib/mongo.js'
import { getV1DailyStats } from '../lib/v1Conversations.js'

// Trend/daily-status are now derived live from the v1 `sessions` pipeline. There is no v1 `daily_statuses`
// collection and the legacy `DailyPatientStatus` cache no longer receives mirror data, so we recompute
// each day from v1 on read. Status policy mirrors the legacy statusEngine: a completed conversation that
// day → green, otherwise → red/missed. (See LEGACY_V1_ADAPTER.md §7.)

type TrendDay = {
  date: string
  duration: number
  status: 'green' | 'yellow' | 'red'
  missed: boolean
}

type RefreshDailyStatusBody = {
  id?: string
  date?: string
}

export const patientTrendRouter = Router()

patientTrendRouter.get('/', asyncHandler(async (request, response) => {
  const id = typeof request.query.id === 'string' ? request.query.id : ''
  const days = Number(typeof request.query.days === 'string' ? request.query.days : 7)

  if (!id || !ObjectId.isValid(id)) {
    response.status(400).json({ error: 'Valid patient id is required' })
    return
  }

  if (days !== 7 && days !== 30) {
    response.status(400).json({ error: 'Only 7-day and 30-day trends are available.' })
    return
  }

  const patientHex = new ObjectId(id).toHexString()
  const cacheDate = getSingaporeDateKey(new Date())

  await withMongo(async (client) => {
    const db = client.db(DB_NAME)
    const dates = getRecentLocalDateKeys(days)
    const trend = await Promise.all(dates.map((date) => buildTrendDay(db, patientHex, date)))
    response.json({ cacheDate, days, trend })
  })
}))

patientTrendRouter.post('/daily-status', asyncHandler(async (request, response) => {
  const body = request.body as RefreshDailyStatusBody
  const id = typeof body.id === 'string' ? body.id : ''
  const date = typeof body.date === 'string' ? body.date : getSingaporeDateKey(new Date())

  if (!id || !ObjectId.isValid(id)) {
    response.status(400).json({ error: 'Valid patient id is required' })
    return
  }

  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    response.status(400).json({ error: 'Date must be YYYY-MM-DD.' })
    return
  }

  const patientId = new ObjectId(id)

  await withMongo(async (client) => {
    const db = client.db(DB_NAME)
    const dailyStatus = await computeDailyStatus(db, patientId, date)
    response.json({ dailyStatus })
  })
}))

async function buildTrendDay(db: Db, patientHex: string, date: string): Promise<TrendDay> {
  const { start, end } = getSingaporeDayBoundsFromKey(date)
  const stats = await getV1DailyStats(db, patientHex, start, end)
  const missed = stats.completedSessionCount === 0
  return {
    date,
    duration: stats.duration,
    status: missed ? 'red' : 'green',
    missed,
  }
}

// Recompute-and-return the legacy DailyPatientStatus shape from v1 (no persistence — the real recompute
// happens inside the v1 session pipeline on check-in completion).
async function computeDailyStatus(db: Db, patientId: ObjectId, date: string) {
  const { start, end } = getSingaporeDayBoundsFromKey(date)
  const [stats, config] = await Promise.all([
    getV1DailyStats(db, patientId.toHexString(), start, end),
    db.collection(NURSE_CONFIG_COLLECTION).findOne(
      { 'patients._id': patientId },
      { projection: { _id: 1, preferredDailySummaryTime: 1 } },
    ),
  ])
  const missed = stats.completedSessionCount === 0
  const now = new Date()
  return {
    id: '',
    patientId: patientId.toHexString(),
    nurseId: config?._id?.toHexString?.() || null,
    date,
    timezone: TIME_ZONE,
    preferredDailySummaryTime: typeof config?.preferredDailySummaryTime === 'string'
      ? config.preferredDailySummaryTime
      : '19:00',
    status: missed ? 'red' : 'green',
    missed,
    sessionCount: stats.sessionCount,
    completedSessionCount: stats.completedSessionCount,
    createdAt: now.toISOString(),
    updatedAt: now.toISOString(),
  }
}

function getRecentLocalDateKeys(days: number) {
  const today = getSingaporeDateKey(new Date())
  const dates: string[] = []
  const [year, month, day] = today.split('-').map(Number)
  const end = Date.UTC(year, month - 1, day)

  for (let index = days - 1; index >= 0; index--) {
    const date = new Date(end - index * 24 * 60 * 60 * 1000)
    dates.push(date.toISOString().slice(0, 10))
  }

  return dates
}
