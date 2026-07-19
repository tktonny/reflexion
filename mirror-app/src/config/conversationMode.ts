// Conversation version switch + client-side Qwen config.
//
//   relay = 版本一: browser/client <-WS-> Node relay <-WS-> Qwen (server holds key). Web + native.
//   http  = 版本二: on-device turn-based HTTP (ASR -> chat -> TTS). Web + native, no relay.
//   ws    = 版本三: native direct realtime WebSocket to Qwen (header auth). Native only; web falls back to relay.
//
// Set EXPO_PUBLIC_CONVERSATION_MODE to pick the version (default 'relay').

export type ConversationMode = 'relay' | 'http' | 'ws'

export const CONVERSATION_MODE: ConversationMode =
  (process.env.EXPO_PUBLIC_CONVERSATION_MODE as ConversationMode) || 'relay'

// Client-reachable Qwen config. Used ONLY by the local modes (http / ws).
// SECURITY: in http/ws modes the API key is reachable by the client — acceptable for a
// self-owned kiosk/demo only. For production, leave apiKey empty and fetch a short-lived
// token from the /api/qwen-token endpoint (see docs/ON_DEVICE_LLM.md §4).
export const QWEN = {
  apiKey: process.env.EXPO_PUBLIC_QWEN_API_KEY || '',
  // China-region host (our key is China-region: relay showed intl 401 -> china OK).
  base: process.env.EXPO_PUBLIC_QWEN_BASE || 'https://dashscope.aliyuncs.com',
  chatModel: process.env.EXPO_PUBLIC_QWEN_CHAT_MODEL || 'qwen-plus',
  ttsModel: process.env.EXPO_PUBLIC_QWEN_TTS_MODEL || 'qwen-tts',
  asrModel: process.env.EXPO_PUBLIC_QWEN_ASR_MODEL || 'qwen3-asr-flash',
  realtimeModel: process.env.EXPO_PUBLIC_QWEN_REALTIME_MODEL || 'qwen3-omni-flash-realtime',
  realtimeUrl: process.env.EXPO_PUBLIC_QWEN_REALTIME_URL || 'wss://dashscope.aliyuncs.com/api-ws/v1/realtime',
  // voice profiles (match server/qwenConfig.mjs)
  defaultVoice: 'Cherry',
  englishVoice: 'Cherry',
  minnanVoice: 'Roy',
  cantoneseVoice: 'Kiki',
}
