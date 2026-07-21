import type { ConversationMode } from '../config/conversationMode'

export type ChatRole = 'system' | 'user' | 'assistant'
export type StatusKind = 'idle' | 'listening' | 'processing' | 'speaking' | 'error'

export type ChatMessage = {
  id: string
  role: ChatRole
  text: string
  streaming?: boolean
  userMetrics?: unknown
}

/** Common shape every conversation version returns, so screens are version-agnostic. */
export interface ConversationApi {
  mode: ConversationMode
  statusKind: StatusKind
  statusText: string
  messages: ChatMessage[]
  startConversation: () => void | Promise<void>
  stopConversation: () => void | Promise<void>
  connecting: boolean
  sessionActive: boolean
  userSpeaking: boolean
  /** Flips true once the assistant delivers its closing goodbye, so screens auto-finalize. */
  ended?: boolean
  // Turn-based (v2) extras — present only in 'http' mode.
  recording?: boolean
  toggleRecording?: () => void
}
