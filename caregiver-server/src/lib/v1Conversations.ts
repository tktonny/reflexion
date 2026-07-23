// v1-backed conversation reads for the legacy caregiver endpoints.
//
// The mirror now writes conversations to the v1 pipeline (`sessions` + `transcript_turns`), NOT the
// legacy `Conversation` / `ConversationIdToPatientIdMap` collections. This module reads v1 and returns
// the EXACT legacy shapes the caregiver app expects, so the conversation routes stay drop-in.
//
// ID convention (LEGACY_V1_ADAPTER.md §1): a patient's v1 id === the legacy 24-hex patientId, and v1
// `sessions.patientId` stores that hex directly — so we scope purely by patientId (globally unique),
// which mirrors what the legacy read routes did (they never re-checked the nurse either).
//
// Policy choices (flagged for review — see LEGACY_V1_ADAPTER.md §7):
//  - A "conversation" = session.type in { daily_checkin, companion }; `device_test` (hardware self-check)
//    is excluded. Companion IS included because the mirror's tap-to-start opens a companion session.
//  - "Completed" = the mirror called .../complete (localCompletedAt set) OR the session reached a
//    post-active state. A freshly-finished check-in sits in `ingesting` until the async pipeline (which
//    may not run in this deployment) promotes it, so we must NOT require state==='completed'.
//  - duration is in SECONDS (the app formats it as `${m}m ${s}s`) ← acquisition.durationMs / 1000.
//  - avgLatency has NO v1 source (the mirror stamps every turn with one identical timestamp and records
//    no latency), so it is 0. Per-line `duration` is an even split of the total so the replay timeline
//    still advances; it is not real per-turn timing.
import type { Db } from 'mongodb'
import { collections } from '../v1/platform/collections.js'

const CONVERSATION_TYPES = ['daily_checkin', 'companion'] as const
const COMPLETED_STATES = ['ingesting', 'processing', 'completed', 'excluded', 'review_pending']

export type V1Session = {
  _id: string
  patientId: string
  tenantId?: string
  type?: string
  state?: string
  createdAt?: Date
  updatedAt?: Date
  localCompletedAt?: Date
  acquisition?: { durationMs?: number; patientTurns?: number; language?: string }
}

export type V1TranscriptTurn = {
  sessionId: string
  sequence?: number
  role?: string
  text?: string
}

export type LegacyLog = {
  sentence: string
  role: string
  words: number
  duration: number
  wordsPerSecond: number
}

export type LegacyConversation = {
  id: string
  patientId: string
  patientName: string
  duration: number
  words: number
  exchanges: number
  avgLatency: number
  createdAt: string | null
  updatedAt: string | null
  logs: LegacyLog[]
}

export function isV1SessionCompleted(session: V1Session): boolean {
  if (session.localCompletedAt) return true
  return COMPLETED_STATES.includes(session.state || '')
}

export function mapV1SessionStatus(session: V1Session): string {
  if (isV1SessionCompleted(session)) return 'completed'
  if (session.state === 'abandoned' || session.state === 'processing_failed') return 'incomplete'
  return 'in_progress'
}

export async function getV1SessionsForPatientRange(db: Db, patientId: string, start: Date, end: Date): Promise<V1Session[]> {
  return db.collection<V1Session>(collections.sessions).find({
    patientId,
    type: { $in: CONVERSATION_TYPES as unknown as string[] },
    createdAt: { $gte: start, $lt: end },
  }).sort({ createdAt: -1 }).toArray()
}

export async function getLatestV1SessionByPatientIds(db: Db, patientIds: string[]): Promise<Map<string, V1Session>> {
  const latest = new Map<string, V1Session>()
  if (!patientIds.length) return latest
  const sessions = await db.collection<V1Session>(collections.sessions).find({
    patientId: { $in: patientIds },
    type: { $in: CONVERSATION_TYPES as unknown as string[] },
  }).sort({ createdAt: -1 }).toArray()
  for (const session of sessions) {
    if (session.patientId && !latest.has(session.patientId)) {
      latest.set(session.patientId, session)
    }
  }
  return latest
}

export async function getV1TurnsBySession(db: Db, sessionIds: string[]): Promise<Map<string, V1TranscriptTurn[]>> {
  const bySession = new Map<string, V1TranscriptTurn[]>()
  if (!sessionIds.length) return bySession
  const turns = await db.collection<V1TranscriptTurn>(collections.transcriptTurns)
    .find({ sessionId: { $in: sessionIds } })
    .sort({ sequence: 1 })
    .toArray()
  for (const turn of turns) {
    const list = bySession.get(turn.sessionId) || []
    list.push(turn)
    bySession.set(turn.sessionId, list)
  }
  return bySession
}

// Word count that also works for languages without whitespace tokens (Mandarin/Cantonese/Japanese/Korean):
// count CJK/kana/hangul characters, plus any whitespace-delimited latin runs.
export function countWords(text: string, language?: string): number {
  const trimmed = (text || '').trim()
  if (!trimmed) return 0
  const cjk = /[㐀-鿿豈-﫿぀-ヿ가-힯]/g
  if (cjk.test(trimmed)) {
    const chars = (trimmed.match(cjk) || []).length
    const latin = (trimmed.replace(cjk, ' ').match(/\S+/g) || []).length
    return chars + latin
  }
  return (trimmed.match(/\S+/g) || []).length
}

// Mirror v1 roles are 'patient' | 'assistant'. The app treats role 'ai' as Aria and everything else as
// Patient (session/[id].tsx:230, patient-summary normalizeRole), so map assistant → 'ai'.
function roleToLegacy(role?: string): string {
  return role === 'assistant' ? 'ai' : 'user'
}

export function serializeV1Session(
  session: V1Session,
  turns: V1TranscriptTurn[],
  patientId: string,
  patientName: string,
  fallbackLanguage?: string,
): LegacyConversation {
  const language = session.acquisition?.language || fallbackLanguage
  const durationSec = session.acquisition?.durationMs ? Math.round(session.acquisition.durationMs / 1000) : 0
  const contentTurns = turns.filter((turn) => (turn.text || '').trim())
  const perLine = contentTurns.length ? durationSec / contentTurns.length : 0
  const logs: LegacyLog[] = contentTurns.map((turn) => {
    const words = countWords(turn.text || '', language)
    return {
      sentence: turn.text || '',
      role: roleToLegacy(turn.role),
      words,
      duration: Math.round(perLine),
      wordsPerSecond: perLine > 0 ? Number((words / perLine).toFixed(2)) : 0,
    }
  })
  const words = logs.reduce((sum, log) => sum + log.words, 0)
  const exchanges = typeof session.acquisition?.patientTurns === 'number'
    ? session.acquisition.patientTurns
    : contentTurns.filter((turn) => turn.role === 'patient').length

  return {
    id: session._id,
    patientId,
    patientName,
    duration: durationSec,
    words,
    exchanges,
    avgLatency: 0,
    createdAt: session.createdAt?.toISOString?.() || null,
    updatedAt: (session.localCompletedAt || session.updatedAt)?.toISOString?.() || null,
    logs,
  }
}

// Legacy DailyConversationStats shape (statusEngine): latest completed session's duration + counts.
export type V1DailyStats = { duration: number; sessionCount: number; completedSessionCount: number }

export async function getV1DailyStats(db: Db, patientId: string, start: Date, end: Date): Promise<V1DailyStats> {
  const sessions = await getV1SessionsForPatientRange(db, patientId, start, end)
  const completed = sessions.filter((session) => isV1SessionCompleted(session))
  const latestCompleted = completed[0] // sessions are sorted createdAt desc
  const durationMs = latestCompleted?.acquisition?.durationMs
  return {
    duration: durationMs ? Math.round(durationMs / 1000) : 0,
    sessionCount: sessions.length,
    completedSessionCount: completed.length,
  }
}
