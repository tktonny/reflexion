// Qwen Omni Realtime config — ported 1:1 from REFLEXION clinic/configs/settings.py
// (the "April Qwen implementation"). Values are overridable via env.

const env = process.env

function str(name, fallback) {
  const v = env[name]
  return v === undefined || v === '' ? fallback : v
}
function num(name, fallback) {
  const v = env[name]
  const n = v === undefined || v === '' ? NaN : Number(v)
  return Number.isFinite(n) ? n : fallback
}

export const qwenConfig = {
  apiKey: env.QWEN_API_KEY || env.DASHSCOPE_API_KEY || null,
  realtimeUrl: str('REFLEXION_QWEN_OMNI_REALTIME_URL', 'wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime'),
  realtimeUrlChina: str('REFLEXION_QWEN_OMNI_REALTIME_URL_CHINA', 'wss://dashscope.aliyuncs.com/api-ws/v1/realtime'),
  realtimeModel: str('REFLEXION_QWEN_OMNI_REALTIME_MODEL', 'qwen3-omni-flash-realtime'),
  transcriptionModel: str('REFLEXION_QWEN_OMNI_REALTIME_TRANSCRIPTION_MODEL', 'gummy-realtime-v1'),
  defaultVoice: str('REFLEXION_QWEN_OMNI_REALTIME_DEFAULT_VOICE', 'Cherry'),
  englishVoice: str('REFLEXION_QWEN_OMNI_REALTIME_ENGLISH_VOICE', 'Cherry'),
  minnanVoice: str('REFLEXION_QWEN_OMNI_REALTIME_MINNAN_VOICE', 'Roy'),
  cantoneseVoice: str('REFLEXION_QWEN_OMNI_REALTIME_CANTONESE_VOICE', 'Kiki'),
  maxTokens: num('REFLEXION_QWEN_OMNI_REALTIME_MAX_TOKENS', 48),
  temperature: num('REFLEXION_QWEN_OMNI_REALTIME_TEMPERATURE', 0.25),
  topP: num('REFLEXION_QWEN_OMNI_REALTIME_TOP_P', 0.7),
  vadThreshold: num('REFLEXION_QWEN_OMNI_REALTIME_VAD_THRESHOLD', 0.1),
  vadPrefixPaddingMs: num('REFLEXION_QWEN_OMNI_REALTIME_VAD_PREFIX_PADDING_MS', 500),
  vadSilenceDurationMs: num('REFLEXION_QWEN_OMNI_REALTIME_VAD_SILENCE_DURATION_MS', 900),
  maxSessionSeconds: num('REFLEXION_REALTIME_MAX_SESSION_SECONDS', 90),
}

export const relayPort = num('REFLEXION_RELAY_PORT', 8787)
export const realtimeWsPath = '/api/clinic/realtime/ws'
