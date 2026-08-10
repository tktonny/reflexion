import type { Db, ObjectId } from 'mongodb'
import {
  CONVERSATION_COLLECTION,
  CONVERSATION_MAP_COLLECTION,
  NURSE_CONFIG_COLLECTION,
} from './constants.js'
import type { Conversation, ConversationMap, ConversationLog, StoredPatient } from './types.js'

export async function findPatient(db: Db, patientId: ObjectId) {
  const config = await db.collection(NURSE_CONFIG_COLLECTION).findOne(
    { 'patients._id': patientId },
    { projection: { patients: { $elemMatch: { _id: patientId } } } },
  )

  return (config?.patients?.[0] || null) as StoredPatient | null
}

export function serializeConversation(
  id: string,
  patientId: ObjectId,
  patientName: string,
  conversation: Conversation,
  map: ConversationMap,
) {
  const createdAt = conversation.createdAt || map.createdAt || null

  return {
    id,
    patientId: patientId.toHexString(),
    patientName,
    duration: conversation.duration || 0,
    words: conversation.words || 0,
    exchanges: conversation.exchanges || 0,
    avgLatency: conversation.avgLatency || 0,
    createdAt: createdAt?.toISOString?.() || null,
    updatedAt: conversation.updatedAt?.toISOString?.() || map.updatedAt?.toISOString?.() || null,
    logs: (conversation.logs || []).map((log) => ({
      sentence: log.sentence || '',
      role: log.role || '',
      words: log.words || 0,
      duration: log.duration || 0,
      wordsPerSecond: log.wordsPerSecond || 0,
    })),
  }
}

export async function getConversationsByMaps(db: Db, maps: ConversationMap[]) {
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

  return new Map(
    conversations.flatMap((conversation) => {
      const keys = [conversation._id?.toHexString?.(), conversation.conversationId?.toHexString?.()]
        .filter((key): key is string => Boolean(key))
      return keys.map((key) => [key, conversation] as const)
    }),
  )
}

export async function getLogsForMaps(db: Db, maps: ConversationMap[]) {
  const conversationById = await getConversationsByMaps(db, maps)
  return maps.flatMap((map) => {
    const conversationId = map.conversationId?.toHexString?.() || ''
    return conversationById.get(conversationId)?.logs || []
  }) as ConversationLog[]
}

export async function getMapsForPatientRange(db: Db, patientId: ObjectId, start: Date, end: Date) {
  return db.collection<ConversationMap>(CONVERSATION_MAP_COLLECTION).find({
    patientId,
    createdAt: { $gte: start, $lt: end },
  }).sort({ createdAt: -1, updatedAt: -1 }).toArray()
}
