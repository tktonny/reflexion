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

// Initial conversation language when nothing is configured/paired. Chinese (Mandarin) by default;
// a paired patient's preferredLanguage and the /settings picker override it. Value is a hint string
// understood by normalizeLanguageKey (voice.ts).
export const DEFAULT_LANGUAGE = process.env.EXPO_PUBLIC_DEFAULT_LANGUAGE || 'mandarin'

// Experimental: let an omni model produce the screening judgment in ONE multimodal call, with an
// automatic fallback to the reliable two-stage qwen-vl-max + qwen-plus pipeline. Toggle with
// EXPO_PUBLIC_OMNI_JUDGMENT=false to force the two-stage path only.
export const OMNI_JUDGMENT = (process.env.EXPO_PUBLIC_OMNI_JUDGMENT ?? 'true') !== 'false'

// Client-reachable Qwen config. Used ONLY by the local modes (http / ws).
// SECURITY: in http/ws modes the API key is reachable by the client — acceptable for a
// self-owned kiosk/demo only. For production, leave apiKey empty and fetch a short-lived
// token from the /api/qwen-token endpoint (see docs/ON_DEVICE_LLM.md §4).
export const QWEN = {
  apiKey: process.env.EXPO_PUBLIC_QWEN_API_KEY || '',
  // China-region host (our key is China-region: relay showed intl 401 -> china OK).
  base: process.env.EXPO_PUBLIC_QWEN_BASE || 'https://dashscope.aliyuncs.com',
  chatModel: process.env.EXPO_PUBLIC_QWEN_CHAT_MODEL || 'qwen-plus',
  visionModel: process.env.EXPO_PUBLIC_QWEN_VISION_MODEL || 'qwen-vl-max',
  ttsModel: process.env.EXPO_PUBLIC_QWEN_TTS_MODEL || 'qwen-tts',
  asrModel: process.env.EXPO_PUBLIC_QWEN_ASR_MODEL || 'qwen3-asr-flash',
  // qwen3.5-omni-realtime series (NOT the old qwen3-omni-flash-realtime): required for semantic_vad
  // (语义打断) which rejects the assistant's own speaker echo / backchannel at the turn-detection
  // level. flash = 80 turns / 480s, ample for a check-in. Live-verified on the generic China host.
  realtimeModel: process.env.EXPO_PUBLIC_QWEN_REALTIME_MODEL || 'qwen3.5-omni-flash-realtime',
  realtimeUrl: process.env.EXPO_PUBLIC_QWEN_REALTIME_URL || 'wss://dashscope.aliyuncs.com/api-ws/v1/realtime',
  // Non-realtime omni model for the experimental single-call screening judgment (OMNI_JUDGMENT).
  omniModel: process.env.EXPO_PUBLIC_QWEN_OMNI_MODEL || 'qwen3-omni-flash',
  // voice profiles (match server/qwenConfig.mjs)
  defaultVoice: 'Cherry',
  englishVoice: 'Cherry',
  minnanVoice: 'Roy',
  cantoneseVoice: 'Kiki',
}

// Fallback nurse/patient ObjectIds for saving check-ins when the device isn't paired
// (testing without the caregiver app). Seed a matching NursePatientConfig with
// server/seed-demo-patient.mjs so the caregiver app can display them.
export const DEMO_IDS = {
  nurseId: process.env.EXPO_PUBLIC_DEMO_NURSE_ID || '64f0000000000000000000a1',
  patientId: process.env.EXPO_PUBLIC_DEMO_PATIENT_ID || '65f0000000000000000000b2',
}
