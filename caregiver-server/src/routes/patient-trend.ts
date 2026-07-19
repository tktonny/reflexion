import { Router } from 'express'
import { ObjectId, type Db } from 'mongodb'
import { asyncHandler } from '../lib/asyncHandler.js'
import { DB_NAME } from '../lib/constants.js'
import { getDailyStatusesForRange, refreshDailyStatusForDate } from '../lib/dailyStatus.js'
import { getSingaporeDateKey } from '../lib/dates.js'
import { withMongo } from '../lib/mongo.js'
import { getDailyConversationStats } from '../lib/statusEngine.js'
import type { DailyPatientStatus } from '../lib/types.js'

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

  const patientId = new ObjectId(id)
  const cacheDate = getSingaporeDateKey(new Date())

  await withMongo(async (client) => {
    const db = client.db(DB_NAME)
    const trend = await buildTrend(db, patientId, days)
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
    const status = await refreshDailyStatusForDate(client.db(DB_NAME), patientId, date)
    response.json({ dailyStatus: serializeDailyStatus(status) })
  })
}))

async function buildTrend(db: Db, patientId: ObjectId, days: number) {
  const dates = getRecentLocalDateKeys(days)
  const statuses = await getDailyStatusesForRange(db, patientId, dates)
  const statusByDate = new Map(statuses.map((status) => [status.date, status]))

  return Promise.all(dates.map((date) => {
    const dailyStatus = statusByDate.get(date)
    return dailyStatus ? toTrendDay(db, patientId, dailyStatus) : getEmptyTrendDay(date)
  }))
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

async function toTrendDay(db: Db, patientId: ObjectId, dailyStatus: DailyPatientStatus) {
  const stats = await getDailyConversationStats(db, patientId, dailyStatus.date)
  return {
    date: dailyStatus.date,
    duration: stats.duration,
    status: dailyStatus.status || 'red',
    missed: dailyStatus.missed,
  } satisfies TrendDay
}

function getEmptyTrendDay(date: string) {
  return {
    date,
    duration: 0,
    status: 'red',
    missed: true,
  } satisfies TrendDay
}

function serializeDailyStatus(dailyStatus: DailyPatientStatus) {
  return {
    id: dailyStatus._id?.toHexString?.() || '',
    patientId: dailyStatus.patientId.toHexString(),
    nurseId: dailyStatus.nurseId?.toHexString?.() || null,
    date: dailyStatus.date,
    timezone: dailyStatus.timezone,
    preferredDailySummaryTime: dailyStatus.preferredDailySummaryTime,
    status: dailyStatus.status,
    missed: dailyStatus.missed,
    sessionCount: dailyStatus.sessionCount,
    completedSessionCount: dailyStatus.completedSessionCount,
    createdAt: dailyStatus.createdAt?.toISOString?.() || null,
    updatedAt: dailyStatus.updatedAt?.toISOString?.() || null,
  }
}
