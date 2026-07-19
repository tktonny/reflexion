import { Router } from 'express'
import { ObjectId } from 'mongodb'
import { asyncHandler } from '../lib/asyncHandler.js'
import {
  CONVERSATION_COLLECTION,
  CONVERSATION_MAP_COLLECTION,
  DB_NAME,
} from '../lib/constants.js'
import {
  getSingaporeDayOfMonth,
  getSingaporeMonthBounds,
  getSingaporeMonthKey,
} from '../lib/dates.js'
import { withMongo } from '../lib/mongo.js'
import type { Conversation, ConversationMap } from '../lib/types.js'

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
  const { start, end, daysInMonth } = getSingaporeMonthBounds(month)

  await withMongo(async (client) => {
    const db = client.db(DB_NAME)
    const maps = await db.collection<ConversationMap>(CONVERSATION_MAP_COLLECTION).find({
      patientId,
      createdAt: { $gte: start, $lt: end },
    }).project({ conversationId: 1, createdAt: 1 }).toArray()
    const conversationIds = maps
      .map((map) => map.conversationId)
      .filter((conversationId): conversationId is ObjectId => Boolean(conversationId))
    const conversations = conversationIds.length
      ? await db.collection<Conversation>(CONVERSATION_COLLECTION).find({
          $or: [
            { _id: { $in: conversationIds } },
            { conversationId: { $in: conversationIds } },
          ],
        }).project({ _id: 1, conversationId: 1, sessionStatus: 1 }).toArray()
      : []
    const conversationById = new Map(
      conversations.flatMap((conversation) => {
        const keys = [conversation._id?.toHexString?.(), conversation.conversationId?.toHexString?.()]
          .filter((key): key is string => Boolean(key))
        return keys.map((key) => [key, conversation] as const)
      }),
    )

    const counts: Record<string, number> = {}
    const completedCounts: Record<string, number> = {}
    for (let day = 1; day <= daysInMonth; day++) {
      counts[String(day)] = 0
      completedCounts[String(day)] = 0
    }

    for (const map of maps) {
      if (!map.createdAt) continue
      const day = getSingaporeDayOfMonth(map.createdAt)
      if (day >= 1 && day <= daysInMonth) {
        counts[String(day)] = (counts[String(day)] || 0) + 1
        const conversationId = map.conversationId?.toHexString?.() || ''
        const conversation = conversationById.get(conversationId)
        if (conversation?.sessionStatus === 'completed') {
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
      patientId: patientId.toHexString(),
      month,
      counts,
      days,
    })
  })
}))
