import type { RequestHandler } from 'expo-router/server'
import { MongoClient, ObjectId } from 'mongodb'

import { verifyDevice } from '../../src/server/deviceAuth'

declare const process: {
  env: Record<string, string | undefined>
}

const DB_NAME = 'ref'
const CONVERSATION_COLLECTION = 'Conversation'
const CONVERSATION_MAP_COLLECTION = 'ConversationIdToPatientIdMap'

type ConversationLogEntry = {
  sentence: string
  role: 'Patient' | 'AI'
  words: number
  duration: number
  wordsPerSecond: number
}

type SaveConversationInput = {
  clientSessionId?: string
  deviceId?: string
  authToken?: string
  startedAt?: string
  endedAt?: string
  sessionStatus?: 'completed' | 'incomplete'
  totalSessionSeconds?: number
  userSpeechSeconds?: number
  ariaSpeechSeconds?: number
  userTurnCount?: number
  aiTurnCount?: number
  language?: string
  appVersion?: string
  networkStatus?: string
  technicalError?: string
  nurseId: string
  patientId: string
  duration: number
  words: number
  exchanges: number
  avgLatency: number
  logs: ConversationLogEntry[]
  assessment?: ConversationAssessment
}

type ConversationAssessment = {
  risk_score?: number | null
  risk_tier?: string | null
  screening_classification?: string | null
  summary?: string
  findings?: string[]
  evidence_for_risk?: string[]
  evidence_against_risk?: string[]
}

function toNumber(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function round1(value: number) {
  return parseFloat(value.toFixed(1))
}

function isSaveConversationInput(data: unknown): data is SaveConversationInput {
  if (!data || typeof data !== 'object') return false
  const input = data as SaveConversationInput
  return (
    typeof input.nurseId === 'string' &&
    typeof input.patientId === 'string' &&
    Array.isArray(input.logs)
  )
}

export const POST: RequestHandler = async (request) => {
  const uri = process.env.MONGODB_URI
  if (!uri) {
    return Response.json(
      { success: false, reason: 'missing_mongodb_uri' },
      { status: 500 },
    )
  }

  const data = await request.json().catch(() => null)
  if (!isSaveConversationInput(data)) {
    return Response.json(
      { success: false, reason: 'invalid_conversation_payload' },
      { status: 400 },
    )
  }
  if (!ObjectId.isValid(data.nurseId)) {
    return Response.json(
      { success: false, reason: 'invalid_nurse_id' },
      { status: 400 },
    )
  }
  if (!ObjectId.isValid(data.patientId)) {
    return Response.json(
      { success: false, reason: 'invalid_patient_id' },
      { status: 400 },
    )
  }

  // Device auth (production): only accept saves from a paired device. Enable via REFLEXION_ENFORCE_DEVICE_AUTH=true.
  if (process.env.REFLEXION_ENFORCE_DEVICE_AUTH === 'true') {
    const verified = await verifyDevice(uri, data.deviceId, data.authToken)
    if (!verified.ok) {
      return Response.json({ success: false, reason: 'unauthorized_device' }, { status: 401 })
    }
  }

  const client = new MongoClient(uri)
  await client.connect()

  try {
    const database = client.db(DB_NAME)
    const now = new Date()
    const clientSessionId =
      typeof data.clientSessionId === 'string' && data.clientSessionId.trim()
        ? data.clientSessionId.trim()
        : null
    if (clientSessionId) {
      const existing = await database.collection(CONVERSATION_COLLECTION).findOne({
        clientSessionId,
        nurseId: new ObjectId(data.nurseId),
        patientId: new ObjectId(data.patientId),
      })
      if (existing?._id) {
        return Response.json({
          success: true,
          conversationId: existing._id.toHexString(),
        })
      }
    }

    const conversation = {
      clientSessionId,
      deviceId: typeof data.deviceId === 'string' ? data.deviceId : '',
      nurseId: new ObjectId(data.nurseId),
      patientId: new ObjectId(data.patientId),
      startedAt: typeof data.startedAt === 'string' ? data.startedAt : now.toISOString(),
      endedAt: typeof data.endedAt === 'string' ? data.endedAt : now.toISOString(),
      sessionStatus: data.sessionStatus === 'completed' ? 'completed' : 'incomplete',
      totalSessionSeconds: Math.round(toNumber(data.totalSessionSeconds)),
      userSpeechSeconds: round1(toNumber(data.userSpeechSeconds)),
      ariaSpeechSeconds: round1(toNumber(data.ariaSpeechSeconds)),
      userTurnCount: Math.round(toNumber(data.userTurnCount)),
      aiTurnCount: Math.round(toNumber(data.aiTurnCount)),
      language: typeof data.language === 'string' ? data.language : '',
      appVersion: typeof data.appVersion === 'string' ? data.appVersion : '0.0.1',
      networkStatus: typeof data.networkStatus === 'string' ? data.networkStatus : 'online',
      technicalError: typeof data.technicalError === 'string' ? data.technicalError : 'false',
      duration: toNumber(data.duration),
      words: toNumber(data.words),
      exchanges: toNumber(data.exchanges),
      avgLatency: toNumber(data.avgLatency),
      logs: data.logs.map((log) => ({
        sentence: log.sentence,
        role: log.role,
        words: toNumber(log.words),
        duration: toNumber(log.duration),
        wordsPerSecond: toNumber(log.wordsPerSecond),
      })),
      // Reflexion cognitive-screening judgment (extra fields beyond the caregiver base schema).
      assessment: data.assessment && typeof data.assessment === 'object' ? data.assessment : null,
      riskScore:
        data.assessment && typeof data.assessment.risk_score === 'number'
          ? data.assessment.risk_score
          : null,
      riskTier: data.assessment?.risk_tier ?? null,
      screeningClassification: data.assessment?.screening_classification ?? null,
      createdAt: now,
      updatedAt: now,
    }

    const insertResult = await database
      .collection(CONVERSATION_COLLECTION)
      .insertOne(conversation)

    await database.collection(CONVERSATION_MAP_COLLECTION).insertOne({
      conversationId: insertResult.insertedId,
      nurseId: new ObjectId(data.nurseId),
      patientId: new ObjectId(data.patientId),
      createdAt: now,
      updatedAt: now,
    })

    return Response.json({
      success: true,
      conversationId: insertResult.insertedId.toHexString(),
    })
  } catch (error) {
    return Response.json(
      {
        success: false,
        reason:
          error instanceof Error ? error.message : 'unknown_save_conversation_error',
      },
      { status: 500 },
    )
  } finally {
    await client.close()
  }
}
