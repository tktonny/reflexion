// Direct DashScope/Qwen HTTP calls for the on-device turn-based version (v2 / Flavor B).
// Request/response shapes VERIFIED live against dashscope.aliyuncs.com (China region)
// via server/smoke-turnbased.mjs: chat=qwen-plus, tts=qwen-tts, asr=qwen3-asr-flash.

import { QWEN } from '../config/conversationMode'
import { getBearer } from './qwenToken'

export type QwenChatMessage = { role: 'system' | 'user' | 'assistant'; content: string }
export type QwenContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } }

async function authHeaders(apiKey?: string) {
  const key = apiKey || (await getBearer())
  return { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' }
}

/**
 * Multimodal chat (text + images) for the video-batch screening. `parts` is the user message
 * content array (text + image_url data URLs). Uses the vision model (qwen-vl-max). Verified live
 * via server/smoke-vision.mjs. Returns the assistant reply text.
 */
export async function qwenVisionChat(
  system: string,
  parts: QwenContentPart[],
  opts: { apiKey?: string; model?: string; maxTokens?: number; temperature?: number } = {},
): Promise<string> {
  const res = await fetch(`${QWEN.base}/compatible-mode/v1/chat/completions`, {
    method: 'POST',
    headers: await authHeaders(opts.apiKey),
    body: JSON.stringify({
      model: opts.model || QWEN.visionModel,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: parts },
      ],
      max_tokens: opts.maxTokens ?? 700,
      temperature: opts.temperature ?? 0.2,
    }),
  })
  const body = await res.json()
  if (!res.ok) throw new Error(`qwen vision ${res.status}: ${JSON.stringify(body).slice(0, 200)}`)
  return String(body?.choices?.[0]?.message?.content ?? '').trim()
}

/** LLM chat turn (OpenAI-compatible endpoint). Returns the assistant reply text. */
export async function qwenChat(
  messages: QwenChatMessage[],
  opts: { apiKey?: string; model?: string; maxTokens?: number; temperature?: number } = {},
): Promise<string> {
  const res = await fetch(`${QWEN.base}/compatible-mode/v1/chat/completions`, {
    method: 'POST',
    headers: await authHeaders(opts.apiKey),
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
    headers: await authHeaders(opts.apiKey),
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

/**
 * Speech-to-text. `audioBase64` = base64 of an audio clip.
 * `format` (e.g. 'wav' | 'm4a' | 'mp3') is passed through when provided; web sends WAV
 * (no format, verified working), native sends m4a with format.
 */
export async function qwenASR(
  audioBase64: string,
  opts: { apiKey?: string; model?: string; format?: string } = {},
): Promise<string> {
  const inputAudio: Record<string, string> = { data: `data:;base64,${audioBase64}` }
  if (opts.format) inputAudio.format = opts.format
  const res = await fetch(`${QWEN.base}/compatible-mode/v1/chat/completions`, {
    method: 'POST',
    headers: await authHeaders(opts.apiKey),
    body: JSON.stringify({
      model: opts.model || QWEN.asrModel,
      messages: [
        { role: 'user', content: [{ type: 'input_audio', input_audio: inputAudio }] },
      ],
    }),
  })
  const body = await res.json()
  if (!res.ok) throw new Error(`qwen asr ${res.status}: ${JSON.stringify(body).slice(0, 200)}`)
  return String(body?.choices?.[0]?.message?.content ?? '').trim()
}
