import { apiPost } from './client'

export type ConversationLogEntry = {
  sentence: string
  role: 'Patient' | 'AI'
  words: number
  duration: number
  wordsPerSecond: number
}

export type SaveConversationInput = {
  clientSessionId?: string
  deviceId?: string
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
}

export type SaveConversationResponse =
  | { success: true; conversationId: string }
  | { success: false; reason: string }

export function saveConversation(input: SaveConversationInput) {
  return apiPost<SaveConversationResponse>('/api/conversations', input)
}
