// Realtime session.update builder (client TS port of relay.mjs buildLiveSessionUpdate).
// Shared by v3 (native direct WS). Values match server/qwenConfig.mjs.

import { QWEN } from '../config/conversationMode'
import { buildLiveInstructions, closingGoodbyeSentence } from './orchestrator'
import { realtimeVoiceForLanguageKey, type LanguageKey } from './voice'

export const REALTIME = {
  // ~one warm sentence + one short question; 48 was clipping natural replies mid-thought.
  maxTokens: 80,
  temperature: 0.25,
  topP: 0.7,
  // semantic_vad (qwen3.5-omni-realtime, 语义打断) uses conversational intent to gate turns, so the
  // assistant's own speaker echo / backchannel / room noise does NOT trigger a spurious user turn —
  // the API-level echo fix. threshold: doc default 0.5 (raise in noise, lower in a very quiet room).
  vadThreshold: 0.5,
  vadSilenceDurationMs: 800,
  transcriptionModel: 'gummy-realtime-v1',
}

function eventId(): string {
  // No Math.random dependency needed for correctness; a monotonic-ish id is fine.
  return `event_${Date.now().toString(36)}`
}

export function buildLiveSessionUpdate(
  patientId: string,
  language: string,
  opts: { voice: string; wrapUp?: boolean; languageKey?: LanguageKey; steer?: string; persona?: 'screening' | 'companion' },
): Record<string, unknown> {
  let instructions = buildLiveInstructions(patientId, language, {
    persona: opts.persona,
    ...(opts.steer ? { steer: opts.steer } : {}),
  })
  if (opts.wrapUp) {
    const goodbye = closingGoodbyeSentence(language)
    instructions +=
      '\nThe live capture is ending now. In your next reply, briefly thank the patient, ' +
      'say the conversation is ending, and end with exactly this goodbye sentence: ' +
      `"${goodbye}" The goodbye must be the final sentence. Do not ask another question after that goodbye.`
  }
  // qwen3.5-omni-realtime has its own voice list (rejects the qwen-tts voices carried on the profile).
  // Pick the language-appropriate realtime voice: 粤语->Kiki, 闽南->Joseph Chen, else a multilingual voice.
  const voice = opts.languageKey ? realtimeVoiceForLanguageKey(opts.languageKey) : REALTIME_VOICE_DEFAULT
  return {
    event_id: eventId(),
    type: 'session.update',
    session: {
      modalities: ['text', 'audio'],
      voice,
      instructions,
      max_tokens: REALTIME.maxTokens,
      temperature: REALTIME.temperature,
      top_p: REALTIME.topP,
      input_audio_format: 'pcm',
      output_audio_format: 'pcm',
      // semantic_vad = the API-level echo defense (qwen3.5-only). No prefix_padding_ms field for it.
      turn_detection: {
        type: 'semantic_vad',
        threshold: REALTIME.vadThreshold,
        silence_duration_ms: REALTIME.vadSilenceDurationMs,
        create_response: true,
        interrupt_response: false,
      },
      input_audio_transcription: { model: REALTIME.transcriptionModel },
    },
  }
}

const REALTIME_VOICE_DEFAULT = realtimeVoiceForLanguageKey('english')

/** Realtime WS URL. Our key is China-region (relay showed intl 401 → china OK). */
export function realtimeWsUrl(): string {
  return `${QWEN.realtimeUrl}?model=${QWEN.realtimeModel}`
}
