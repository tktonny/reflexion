// Conversation version switch + client-side Qwen config.
//
//   relay  = browser/client <-WS-> Node relay <-WS-> Qwen (server holds key). Web + native.
//   http   = on-device turn-based HTTP (ASR -> chat -> TTS). Web + native, no relay.
//   ws     = websocket-v0.0.0: native direct realtime WebSocket to Qwen (header auth). Native only; web falls back to relay.
//   webrtc = webrtc-v0.0.0: native WebRTC realtime to Qwen (SDP handshake + oai-events data channel).
//            Audio rides an RTP media track with BUILT-IN echo cancellation + noise reduction — the
//            hardware-grade fix for the mirror's speaker→mic echo. Native only; web falls back to relay.
//
// Set EXPO_PUBLIC_CONVERSATION_MODE to pick the version (default 'relay').

export type ConversationMode = 'relay' | 'http' | 'ws' | 'webrtc'

export const CONVERSATION_MODE: ConversationMode =
  (process.env.EXPO_PUBLIC_CONVERSATION_MODE as ConversationMode) || 'relay'

// Human-facing version names (transport-based). ws/webrtc are the realtime paths the product ships.
export const VERSION_LABELS: Record<ConversationMode, string> = {
  relay: 'relay-v0.0.0',
  http: 'http-v0.0.0',
  ws: 'websocket-v0.0.0',
  webrtc: 'webrtc-v0.0.0',
}
export const VERSION_LABEL = VERSION_LABELS[CONVERSATION_MODE]

// Initial conversation language when nothing is configured/paired. Chinese (Mandarin) by default;
// a paired patient's preferredLanguage and the /settings picker override it. Value is a hint string
// understood by normalizeLanguageKey (voice.ts).
export const DEFAULT_LANGUAGE = process.env.EXPO_PUBLIC_DEFAULT_LANGUAGE || 'mandarin'

// Experimental: let an omni model produce the screening judgment in ONE multimodal call, with an
// automatic fallback to the reliable two-stage qwen-vl-max + qwen-plus pipeline. Toggle with
// EXPO_PUBLIC_OMNI_JUDGMENT=false to force the two-stage path only.
export const OMNI_JUDGMENT = (process.env.EXPO_PUBLIC_OMNI_JUDGMENT ?? 'true') !== 'false'

// Client-reachable Qwen endpoint/model identifiers. Credentials always come from the authenticated
// `/api/v1/sessions/:sessionId/realtime-tickets` backend route and are never compiled into the APK.
export const QWEN = {
  // Production credentials are issued per backend session and never compiled into the APK.
  apiKey: '',
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
  // WebRTC realtime (webrtc-v0.0.0) needs a WORKSPACE-scoped MaaS host — the generic dashscope host is
  // WebSocket-only. Supply the workspace id (+ region) to reach it; otherwise a full URL override.
  //   https://{workspaceId}.{region}.maas.aliyuncs.com/api/v1/webrtc/realtime
  workspaceId: process.env.EXPO_PUBLIC_QWEN_WORKSPACE_ID || '',
  webrtcRegion: process.env.EXPO_PUBLIC_QWEN_WEBRTC_REGION || 'cn-beijing', // or 'ap-southeast-1' (Singapore)
  webrtcUrl: process.env.EXPO_PUBLIC_QWEN_WEBRTC_URL || '', // full override, wins over workspaceId/region
  // Non-realtime omni model for the experimental single-call screening judgment (OMNI_JUDGMENT).
  omniModel: process.env.EXPO_PUBLIC_QWEN_OMNI_MODEL || 'qwen3-omni-flash',
  // voice profiles (match server/qwenConfig.mjs)
  defaultVoice: 'Cherry',
  englishVoice: 'Cherry',
  minnanVoice: 'Roy',
  cantoneseVoice: 'Kiki',
}
