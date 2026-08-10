import type { Db, ObjectId } from 'mongodb'
import {
  CONVERSATION_COLLECTION,
  CONVERSATION_MAP_COLLECTION,
  NURSE_CONFIG_COLLECTION,
  TIME_ZONE,
} from './constants.js'
import { getSingaporeDayBoundsFromKey } from './dates.js'
import type { Conversation, ConversationMap, DailyPatientStatus } from './types.js'

type ConversationForDate = {
  conversation: Conversation
  map: ConversationMap
}

type PatientConfig = {
  nurseId: ObjectId | null
  preferredDailySummaryTime: string
}

export type DailyConversationStats = {
  duration: number
  sessionCount: number
  completedSessionCount: number
}

export async function computeDailyStatusFromConversations(
  db: Db,
  patientId: ObjectId,
  date: string,
): Promise<DailyPatientStatus> {
  const [patientConfig, stats] = await Promise.all([
    findPatientConfig(db, patientId),
    getDailyConversationStats(db, patientId, date),
  ])
  const missed = stats.completedSessionCount === 0
  const status = missed ? 'red' : 'green'

  return {
    patientId,
    ...(patientConfig.nurseId ? { nurseId: patientConfig.nurseId } : {}),
    date,
    timezone: TIME_ZONE,
    preferredDailySummaryTime: patientConfig.preferredDailySummaryTime,
    status,
    missed,
    sessionCount: stats.sessionCount,
    completedSessionCount: stats.completedSessionCount,
  }
}

export async function getDailyConversationStats(
  db: Db,
  patientId: ObjectId,
  date: string,
): Promise<DailyConversationStats> {
  const conversations = await getConversationsForPatientDate(db, patientId, date)
  const completedConversations = conversations.filter(({ conversation }) => isCompletedConversation(conversation))
  const latestCompletedConversation = completedConversations[0]?.conversation || null

  return {
    duration: Number(latestCompletedConversation?.duration || 0),
    sessionCount: conversations.length,
    completedSessionCount: completedConversations.length,
  }
}

async function getConversationsForPatientDate(db: Db, patientId: ObjectId, date: string) {
  const { start, end } = getSingaporeDayBoundsFromKey(date)
  const maps = await db.collection<ConversationMap>(CONVERSATION_MAP_COLLECTION).find({
    patientId,
    createdAt: { $gte: start, $lt: end },
  }).sort({ createdAt: -1, updatedAt: -1 }).toArray()
  const conversationIds = maps
    .map((map) => map.conversationId)
    .filter((conversationId): conversationId is ObjectId => Boolean(conversationId))
  const conversations = conversationIds.length
    ? await db.collection<Conversation>(CONVERSATION_COLLECTION).find({
        $or: [
          { _id: { $in: conversationIds } },
          { conversationId: { $in: conversationIds } },
        ],
      }).toArray()
    : []
  const conversationById = new Map(
    conversations.flatMap((conversation) => {
      const keys = [conversation._id?.toHexString?.(), conversation.conversationId?.toHexString?.()]
        .filter((key): key is string => Boolean(key))
      return keys.map((key) => [key, conversation] as const)
    }),
  )

  const items: ConversationForDate[] = []
  for (const map of maps) {
    const conversationId = map.conversationId?.toHexString?.() || ''
    const conversation = conversationById.get(conversationId)
    if (conversation) {
      items.push({ conversation, map })
    }
  }

  return items
}

function isCompletedConversation(conversation: Conversation) {
  return conversation.sessionStatus === 'completed'
}

async function findPatientConfig(db: Db, patientId: ObjectId): Promise<PatientConfig> {
  const config = await db.collection(NURSE_CONFIG_COLLECTION).findOne(
    { 'patients._id': patientId },
    { projection: { _id: 1, preferredDailySummaryTime: 1 } },
  )

  return {
    nurseId: config?._id || null,
    preferredDailySummaryTime: typeof config?.preferredDailySummaryTime === 'string'
      ? config.preferredDailySummaryTime
      : '19:00',
  }
}
