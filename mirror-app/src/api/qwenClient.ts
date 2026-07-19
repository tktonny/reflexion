// Direct DashScope/Qwen HTTP calls for the on-device turn-based version (v2 / Flavor B).
// Request/response shapes VERIFIED live against dashscope.aliyuncs.com (China region)
// via server/smoke-turnbased.mjs: chat=qwen-plus, tts=qwen-tts, asr=qwen3-asr-flash.

import { QWEN } from '../config/conversationMode'

export type QwenChatMessage = { role: 'system' | 'user' | 'assistant'; content: string }

function authHeaders(apiKey?: string) {
  const key = apiKey || QWEN.apiKey
  if (!key) throw new Error('missing Qwen key: set EXPO_PUBLIC_QWEN_API_KEY (kiosk/demo) or pass a token')
  return { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' }
}

/** LLM chat turn (OpenAI-compatible endpoint). Returns the assistant reply text. */
export async function qwenChat(
  messages: QwenChatMessage[],
  opts: { apiKey?: string; model?: string; maxTokens?: number; temperature?: number } = {},
): Promise<string> {
  const res = await fetch(`${QWEN.base}/compatible-mode/v1/chat/completions`, {
    method: 'POST',
    headers: authHeaders(opts.apiKey),
    body: JSON.stringify({
      model: opts.model || QWEN.chatModel,
      messages,
      max_tokens: opts.maxTokens ?? 120,
      temperature: opts.temperature ?? 0.4,
    }),
  })
  const body = await res.json()
  if (!res.ok) throw new Error(`qwen chat ${res.status}: ${JSON.stringify(body).slice(0, 200)}`)
  return String(body?.choices?.[0]?.message?.content ?? '').trim()
}

/** Text-to-speech. Returns base64 audio (preferred for web playback) and/or a URL. */
export async function qwenTTS(
  text: string,
  opts: { apiKey?: string; model?: string; voice?: string } = {},
): Promise<{ audioBase64: string | null; url: string | null; format: 'wav' }> {
  const res = await fetch(`${QWEN.base}/api/v1/services/aigc/multimodal-generation/generation`, {
    method: 'POST',
    headers: authHeaders(opts.apiKey),
    body: JSON.stringify({
      model: opts.model || QWEN.ttsModel,
      input: { text, voice: opts.voice || QWEN.defaultVoice },
    }),
  })
  const body = await res.json()
  if (!res.ok) throw new Error(`qwen tts ${res.status}: ${JSON.stringify(body).slice(0, 200)}`)
  const audio = body?.output?.audio ?? {}
  return { audioBase64: audio.data ?? null, url: audio.url ?? null, format: 'wav' }
}

/** Speech-to-text. `wavBase64` = base64 of a WAV (or other supported) audio clip. */
export async function qwenASR(
  wavBase64: string,
  opts: { apiKey?: string; model?: string } = {},
): Promise<string> {
  const res = await fetch(`${QWEN.base}/compatible-mode/v1/chat/completions`, {
    method: 'POST',
    headers: authHeaders(opts.apiKey),
    body: JSON.stringify({
      model: opts.model || QWEN.asrModel,
      messages: [
        { role: 'user', content: [{ type: 'input_audio', input_audio: { data: `data:;base64,${wavBase64}` } }] },
      ],
    }),
  })
  const body = await res.json()
  if (!res.ok) throw new Error(`qwen asr ${res.status}: ${JSON.stringify(body).slice(0, 200)}`)
  return String(body?.choices?.[0]?.message?.content ?? '').trim()
}
