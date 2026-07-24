import type { ConversationMode } from '../config/conversationMode'
import type { DailyConversationPlan } from '../orchestration/deterministicSpeech'
import type { SessionTelemetry } from '../orchestration/sessionTelemetry'
import type { TurnTakingPhase } from '../orchestration/turnTaking'

export type ChatRole = 'system' | 'user' | 'assistant'
export type StatusKind = 'idle' | 'listening' | 'processing' | 'speaking' | 'error'

export type ChatMessage = {
  id: string
  role: ChatRole
  text: string
  streaming?: boolean
  /** The user spoke over this assistant turn, so device playback did not reach the end. */
  interrupted?: boolean
  userMetrics?: unknown
}

export type ConversationOptions = {
  patientId?: string
  patientName?: string
  language?: string
  persona?: 'screening' | 'companion'
  dailyPlan?: DailyConversationPlan
  pushToTalk?: boolean
  /** Aria TTS speech rate (doc default 0.85× for elderly listeners; configurable per patient). */
  speechRate?: number
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
  /** True while a user utterance is actively interrupting assistant playback. */
  bargeInActive?: boolean
  /** Observable realtime lifecycle; present on transports that implement the Phase 0 turn contract. */
  turnState?: TurnTakingPhase
  /** Flips true once the assistant delivers its closing goodbye, so screens auto-finalize. */
  ended?: boolean
  /**
   * Why the session ended, once `ended` is true: 'goodbye' = a real completed check-in; 'error' =
   * failClosed (connection/audio/provider failure). The screen uses this to avoid saving or announcing
   * a bogus check-in when a startup failure ends a session before the patient ever answered.
   */
  endReason?: 'goodbye' | 'error'
  // Turn-based (v2) extras — present only in 'http' mode.
  recording?: boolean
  /** Press-in starts capture; press-out ends and submits exactly that held utterance. */
  beginPushToTalk?: () => void
  endPushToTalk?: () => void
  /** Legacy tap-to-toggle control retained for non-touch callers and diagnostics. */
  toggleRecording?: () => void
  /**
   * Structured per-turn + per-session telemetry captured during the conversation (baseline §3).
   * Present only on transports that implement the Phase-1 turn contract; used at finalize to build
   * the raw session-upload payload. Returns null before a session has produced any turns.
   */
  getSessionTelemetry?: () => SessionTelemetry | null
  /**
   * Raw patient-speech PCM16 frames captured this session (mono @ the given sampleRate), for the
   * session audio artifact (transcription + Phase-6 acoustic). Only frames captured while the mic was
   * open are recorded, so Aria's own voice is excluded. Returns null when no audio was captured.
   */
  getSessionAudio?: () => { base64Frames: string[]; sampleRate: number } | null
}
