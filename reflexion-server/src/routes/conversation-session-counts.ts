import { Router } from 'express'
import { ObjectId } from 'mongodb'
import { asyncHandler } from '../lib/asyncHandler.js'
import { DB_NAME } from '../lib/constants.js'
import {
  getSingaporeDayOfMonth,
  getSingaporeMonthBounds,
  getSingaporeMonthKey,
} from '../lib/dates.js'
import { withMongo } from '../lib/mongo.js'
import { getV1SessionsForPatientRange, isV1SessionCompleted } from '../lib/v1Conversations.js'

export const conversationSessionCountsRouter = Router()

conversationSessionCountsRouter.get('/', asyncHandler(async (request, response) => {
  const id = typeof request.query.id === 'string' ? request.query.id : ''
  const month = typeof request.query.month === 'string' ? request.query.month : getSingaporeMonthKey(new Date())

  if (!id || !ObjectId.isValid(id)) {
    response.status(400).json({ error: 'Valid patient id is required' })
    return
  }
  if (!/^\d{4}-\d{2}$/.test(month)) {
    response.status(400).json({ error: 'Month must be YYYY-MM.' })
    return
  }

  const patientId = new ObjectId(id)
  const patientHex = patientId.toHexString()
  const { start, end, daysInMonth } = getSingaporeMonthBounds(month)

  await withMongo(async (client) => {
    const db = client.db(DB_NAME)
    const sessions = await getV1SessionsForPatientRange(db, patientHex, start, end)

    const counts: Record<string, number> = {}
    const completedCounts: Record<string, number> = {}
    for (let day = 1; day <= daysInMonth; day++) {
      counts[String(day)] = 0
      completedCounts[String(day)] = 0
    }

    for (const session of sessions) {
      if (!session.createdAt) continue
      const day = getSingaporeDayOfMonth(session.createdAt)
      if (day >= 1 && day <= daysInMonth) {
        counts[String(day)] = (counts[String(day)] || 0) + 1
        if (isV1SessionCompleted(session)) {
          completedCounts[String(day)] = (completedCounts[String(day)] || 0) + 1
        }
      }
    }

    const days = Array.from({ length: daysInMonth }, (_, index) => {
      const day = index + 1
      return {
        date: `${month}-${String(day).padStart(2, '0')}`,
        day,
        count: counts[String(day)] || 0,
        completedCount: completedCounts[String(day)] || 0,
        hasCompletedSession: (completedCounts[String(day)] || 0) > 0,
      }
    })

    response.json({
      patientId: patientHex,
      month,
      counts,
      days,
    })
  })
}))
