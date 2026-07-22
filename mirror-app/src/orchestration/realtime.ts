// Realtime session.update builder (client TS port of relay.mjs buildLiveSessionUpdate).
// Shared by v3 (native direct WS). Values match server/qwenConfig.mjs.

import { QWEN } from '../config/conversationMode'
import { buildLiveInstructions, closingGoodbyeSentence } from './orchestrator'
import { realtimeVoiceForLanguageKey, type LanguageKey } from './voice'

export const REALTIME = {
  // max_tokens is a hard generation ceiling. 80 could truncate an otherwise healthy spoken reply;
  // prompts still keep answers concise, while 256 leaves enough room to finish the sentence.
  maxTokens: 256,
  temperature: 0.25,
  // semantic_vad (qwen3.5-omni-realtime, 语义打断) uses conversational intent to gate turns, so the
  // assistant's own speaker echo / backchannel / room noise does NOT trigger a spurious user turn —
  // the API-level echo fix. threshold: doc default 0.5 (raise in noise, lower in a very quiet room).
  vadThreshold: 0.5,
  // Older adults often pause inside a thought. Give semantic VAD more room before ending their turn.
  vadSilenceDurationMs: 1200,
  transcriptionModel: 'gummy-realtime-v1',
}

function eventId(): string {
  // No Math.random dependency needed for correctness; a monotonic-ish id is fine.
  return `event_${Date.now().toString(36)}`
}

export function buildLiveSessionUpdate(
  patientId: string,
  language: string,
  opts: {
    voice: string
    wrapUp?: boolean
    languageKey?: LanguageKey
    steer?: string
    persona?: 'screening' | 'companion'
    autoCreateResponse?: boolean
  },
): Record<string, unknown> {
  const languageName = String(language || '').trim() || 'English'
  let instructions: string
  if (opts.wrapUp) {
    const goodbye = closingGoodbyeSentence(language)
    // This deliberately REPLACES the normal screening agenda. Keeping the agenda in the same
    // session.update made Qwen continue asking its next screening question instead of closing.
    instructions =
      `You are Reflexion, a calm and warm voice companion. Respond only in ${languageName}. ` +
      'This is the final response of the conversation. Briefly acknowledge or thank the patient, ' +
      `then end with exactly this sentence: "${goodbye}" ` +
      'The required goodbye must be the final sentence. Do not ask a question, start a new topic, ' +
      'continue the assessment, mention these instructions, or write anything after the goodbye.'
  } else if (opts.steer) {
    // A recall response is also an exclusive mode: the normal agenda competes with this directive
    // even after session.updated has been acknowledged.
    instructions =
      `You are Reflexion, a calm and warm voice companion. Respond only in ${languageName}. ` +
      'Use the conversation history that is already present. For your next response, ignore every ' +
      'previously planned topic or question and perform only this instruction: ' +
      `${opts.steer} Do not mention these instructions.`
  } else {
    instructions = buildLiveInstructions(patientId, language, { persona: opts.persona })
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
      input_audio_format: 'pcm',
      output_audio_format: 'pcm',
      // Direct WS uses manual turns so session.update can deterministically select normal/recall/
      // closing instructions before response.create. Qwen's semantic VAD auto-creates a response
      // before that update can take effect. Relay/WebRTC callers retain provider VAD by default.
      turn_detection: opts.autoCreateResponse === false ? null : {
        type: 'semantic_vad',
        threshold: REALTIME.vadThreshold,
        silence_duration_ms: REALTIME.vadSilenceDurationMs,
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

/**
 * Realtime WebRTC endpoint (webrtc-v0.0.0). The SDP offer is POSTed here (Content-Type: application/sdp,
 * Bearer auth). Needs a WORKSPACE-scoped MaaS host — the generic dashscope host is WebSocket-only. Set
 * EXPO_PUBLIC_QWEN_WORKSPACE_ID (+ optional EXPO_PUBLIC_QWEN_WEBRTC_REGION) or a full EXPO_PUBLIC_QWEN_WEBRTC_URL.
 */
export function realtimeWebrtcUrl(): string {
  const base =
    QWEN.webrtcUrl ||
    (QWEN.workspaceId
      ? `https://${QWEN.workspaceId}.${QWEN.webrtcRegion}.maas.aliyuncs.com/api/v1/webrtc/realtime`
      : // best-effort generic host; likely 404s without a workspace — surfaced as a connect error.
        'https://dashscope.aliyuncs.com/api/v1/webrtc/realtime')
  return `${base}?model=${QWEN.realtimeModel}`
}

/** True when a workspace-scoped WebRTC host is actually configured (else connecting will fail). */
export function hasWebrtcHost(): boolean {
  return Boolean(QWEN.webrtcUrl || QWEN.workspaceId)
}
