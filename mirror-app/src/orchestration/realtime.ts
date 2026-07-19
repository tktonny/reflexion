// Realtime session.update builder (client TS port of relay.mjs buildLiveSessionUpdate).
// Shared by v3 (native direct WS). Values match server/qwenConfig.mjs.

import { QWEN } from '../config/conversationMode'
import { buildLiveInstructions, closingGoodbyeSentence } from './orchestrator'

export const REALTIME = {
  maxTokens: 48,
  temperature: 0.25,
  topP: 0.7,
  vadThreshold: 0.1,
  vadPrefixPaddingMs: 500,
  vadSilenceDurationMs: 900,
  transcriptionModel: 'gummy-realtime-v1',
}

function eventId(): string {
  // No Math.random dependency needed for correctness; a monotonic-ish id is fine.
  return `event_${Date.now().toString(36)}`
}

export function buildLiveSessionUpdate(
  patientId: string,
  language: string,
  opts: { voice: string; wrapUp?: boolean },
): Record<string, unknown> {
  let instructions = buildLiveInstructions(patientId, language, {})
  if (opts.wrapUp) {
    const goodbye = closingGoodbyeSentence(language)
    instructions +=
      '\nThe live capture is ending now. In your next reply, briefly thank the patient, ' +
      'say the conversation is ending, and end with exactly this goodbye sentence: ' +
      `"${goodbye}" The goodbye must be the final sentence. Do not ask another question after that goodbye.`
  }
  return {
    event_id: eventId(),
    type: 'session.update',
    session: {
      modalities: ['text', 'audio'],
      voice: opts.voice,
      instructions,
      max_tokens: REALTIME.maxTokens,
      temperature: REALTIME.temperature,
      top_p: REALTIME.topP,
      input_audio_format: 'pcm',
      output_audio_format: 'pcm',
      turn_detection: {
        type: 'server_vad',
        threshold: REALTIME.vadThreshold,
        prefix_padding_ms: REALTIME.vadPrefixPaddingMs,
        silence_duration_ms: REALTIME.vadSilenceDurationMs,
        create_response: true,
        interrupt_response: false,
      },
      input_audio_transcription: { model: REALTIME.transcriptionModel },
    },
  }
}

/** Realtime WS URL. Our key is China-region (relay showed intl 401 → china OK). */
export function realtimeWsUrl(): string {
  return `${QWEN.realtimeUrl}?model=${QWEN.realtimeModel}`
}
