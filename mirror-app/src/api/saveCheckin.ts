import AsyncStorage from '@react-native-async-storage/async-storage'

import { DEMO_IDS } from '../config/conversationMode'
import { DEVICE_AUTH_TOKEN_STORAGE_KEY } from '../constants/nursePatientConfig'
import { queuePendingConversation } from '../storage/conversationQueue'
import { getStoredConversationOwnerIds, getStoredMirrorSessionMetadata } from '../storage/mirrorStorage'
import { randomId } from '../utils/id'
import { saveConversation, type ConversationLogEntry, type SaveConversationInput } from './conversation'
import type { ScreeningAssessment } from './assess'
import type { ChatMessage } from '../hooks/conversationTypes'

function countWords(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length
}

export type CheckinArgs = {
  messages: ChatMessage[]
  startedAt: Date
  endedAt: Date
  nurseId: string
  patientId: string
  deviceId?: string
  authToken?: string
  language?: string
  assessment?: ScreeningAssessment | null
}

/** Build the exact Conversation payload the caregiver ecosystem expects (logs + metrics + judgment). */
export function buildCheckinPayload(a: CheckinArgs): SaveConversationInput {
  const turns = a.messages.filter((m) => m.role === 'user' || m.role === 'assistant')
  const logs: ConversationLogEntry[] = turns.map((m) => ({
    sentence: m.text.trim(),
    // lowercase to match the caregiver-server contract: it converts role 'ai' -> "Aria" and labels
    // everything else as the patient. Uppercase 'AI' would make it count Aria's lines as the patient.
    role: m.role === 'assistant' ? 'ai' : 'patient',
    words: countWords(m.text),
    duration: 0,
    wordsPerSecond: 0,
  }))
  const userTurnCount = logs.filter((l) => l.role === 'patient').length
  const aiTurnCount = logs.filter((l) => l.role === 'ai').length
  const totalWords = logs.reduce((sum, l) => sum + l.words, 0)
  const durationSec = Math.max(0, Math.round((a.endedAt.getTime() - a.startedAt.getTime()) / 1000))
  // Heuristic (we don't measure per-utterance speech seconds like the old WebRTC hook did):
  // treat >=3 patient turns as a completed check-in.
  const sessionStatus: 'completed' | 'incomplete' = userTurnCount >= 3 ? 'completed' : 'incomplete'

  return {
    clientSessionId: randomId('session'),
    deviceId: a.deviceId ?? '',
    authToken: a.authToken,
    startedAt: a.startedAt.toISOString(),
    endedAt: a.endedAt.toISOString(),
    sessionStatus,
    totalSessionSeconds: durationSec,
    userSpeechSeconds: 0,
    ariaSpeechSeconds: 0,
    userTurnCount,
    aiTurnCount,
    language: a.language ?? 'en',
    appVersion: '0.0.1',
    networkStatus: 'online',
    technicalError: 'false',
    nurseId: a.nurseId,
    patientId: a.patientId,
    duration: durationSec,
    words: totalWords,
    exchanges: logs.length,
    avgLatency: 0,
    logs,
    assessment: a.assessment ?? undefined,
  }
}

/** Real paired IDs if the mirror is paired, else the demo IDs (for testing without pairing). */
export async function resolveOwnerIds(): Promise<{ nurseId: string; patientId: string; deviceId?: string; authToken?: string; language?: string }> {
  const owner = await getStoredConversationOwnerIds()
  const meta = await getStoredMirrorSessionMetadata()
  const authToken = (await AsyncStorage.getItem(DEVICE_AUTH_TOKEN_STORAGE_KEY)) ?? undefined
  if (owner) {
    return { nurseId: owner.nurseId, patientId: owner.patientId, deviceId: meta.deviceId ?? undefined, authToken, language: meta.language ?? undefined }
  }
  return { nurseId: DEMO_IDS.nurseId, patientId: DEMO_IDS.patientId, deviceId: 'demo-device', authToken, language: undefined }
}

/** Persist a finished check-in (Conversation + ConversationIdToPatientIdMap via /api/conversations). */
export async function saveCheckin(a: CheckinArgs): Promise<{ saved: boolean; reason?: string }> {
  const payload = buildCheckinPayload(a)
  try {
    const r = await saveConversation(payload)
    if (r.success) return { saved: true }
    await queuePendingConversation(payload)
    return { saved: false, reason: r.reason }
  } catch (e) {
    await queuePendingConversation(payload)
    return { saved: false, reason: e instanceof Error ? e.message : 'network_error' }
  }
}
