import {
  getActiveQwenEndpoint,
  getActiveQwenHttpBase,
  getActiveQwenHttpModel,
  getActiveQwenModel,
  getActiveQwenTicket,
} from './sessionSync'

// Every Qwen call uses a short-lived ticket issued for the active backend session.
// The long-lived provider key never enters the application bundle.

export function clearTokenCache(): void {
  // Session ticket cache is owned by sessionSync and cleared when the session completes.
}

/** Bearer credential for a Qwen HTTP/WS call. Also (re)populates the active ticket's region endpoints. */
export async function getBearer(): Promise<string> {
  return getActiveQwenTicket()
}

// Region-adaptive endpoints the backend chose for this device, read AFTER getBearer() has resolved.
// Undefined before the first ticket → callers fall back to the build-time defaults in conversationMode.
export function getQwenRealtimeEndpoint(): string | undefined { return getActiveQwenEndpoint() }
export function getQwenHttpBase(): string | undefined { return getActiveQwenHttpBase() }
export function getQwenRealtimeModel(): string | undefined { return getActiveQwenModel() }
/** Region-correct HTTP model name (tts/asr/chat/vision) from the active ticket, else undefined. */
export function getQwenHttpModel(kind: 'tts' | 'asr' | 'chat' | 'vision'): string | undefined {
  return getActiveQwenHttpModel(kind)
}
