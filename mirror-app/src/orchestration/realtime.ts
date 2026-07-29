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

/** A human-readable current local date+time in the conversation's language, so Aria always knows the
 *  time/day without a tool round-trip. Computed on the device (its timezone) at each session.update. */
function currentLocalTimeLine(language: string): string {
  try {
    const locale = /mandarin|chinese|zh|粤|cantonese|闽|minnan/i.test(language) ? 'zh-CN' : 'en-US'
    return new Intl.DateTimeFormat(locale, { dateStyle: 'full', timeStyle: 'short' }).format(new Date())
  } catch {
    return new Date().toString()
  }
}

// Function tools Aria (companion) may call over the realtime WS for dynamic knowledge. The names map to
// the backend /sessions/:id/tool-invocations tool ids (which hold the provider keys and audit each call).
// NOT registered for screening (scripted). enable_search is intentionally never set — mutually exclusive.
export const REALTIME_TOOL_BACKEND: Record<string, string> = {
  get_weather: 'weather.get',
  web_search: 'web.search',
  list_medications: 'medication.list',
  upcoming_reminders: 'reminders.upcoming',
}
// Qwen omni realtime follows the OpenAI *Realtime* protocol, whose session.update tools are FLAT
// ({type,name,description,parameters}) — NOT the Chat-Completions nested {type,function:{...}} shape.
// The nested shape is rejected by the realtime server with an in-band error frame right after open,
// which (before this fix) knocked companion sessions — the only persona that sends tools — off omni
// and down to the turn-based fallback. Screening sends no tools, so it stayed on omni; that asymmetry
// was the tell. The incoming tool call is keyed by top-level `name`, so the response path is unchanged.
const COMPANION_TOOLS = [
  {
    type: 'function',
    name: 'get_weather',
    description: "Current weather and a short forecast for a city. Omit `city` for the patient's home area (already in your context).",
    parameters: { type: 'object', properties: { city: { type: 'string', description: 'City name, e.g. Tokyo.' } } },
  },
  {
    type: 'function',
    name: 'web_search',
    description: 'Search the web for current facts, news, or a general question you are unsure about.',
    parameters: { type: 'object', properties: { query: { type: 'string', description: 'The search query.' } }, required: ['query'] },
  },
  {
    type: 'function',
    name: 'list_medications',
    description: "List the patient's current medications and their schedules.",
    parameters: { type: 'object', properties: {} },
  },
  {
    type: 'function',
    name: 'upcoming_reminders',
    description: "List the patient's reminders or medications due in the next 24 hours.",
    parameters: { type: 'object', properties: {} },
  },
] as const

export function buildLiveSessionUpdate(
  patientId: string,
  language: string,
  opts: {
    voice: string
    wrapUp?: boolean
    languageKey?: LanguageKey
    persona?: 'screening' | 'companion'
    patientName?: string
    autoCreateResponse?: boolean
    /**
     * One-turn instruction APPENDED to the persona prompt — how the guided check-in tells Aria what to
     * say next (see guidedStageDirective). An earlier `steer` option replaced the instructions wholesale,
     * which also threw away the Singapore context, the clock and the patient memory; appending keeps all
     * of that and still pins the turn, because the flow machine — not the prompt — owns the agenda.
     */
    turnDirective?: string
    memory?: string[]
    /** Today's local weather, one short human line (from the device's ambient widget), if available. */
    weather?: string
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
  } else {
    instructions = buildLiveInstructions(patientId, language, {
      persona: opts.persona, patientName: opts.patientName, memory: opts.memory,
      now: currentLocalTimeLine(language), weather: opts.weather,
    })
    // Appended last so it outranks the prompt's own (hidden) agenda for this one turn.
    if (opts.turnDirective) instructions += `\n\n${opts.turnDirective}`
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
      // Direct WS uses manual turns so session.update can deterministically select normal/closing
      // instructions before response.create. Qwen's semantic VAD auto-creates a response
      // before that update can take effect. Relay/WebRTC callers retain provider VAD and MUST set
      // create_response:true explicitly — without it they'd rely on the API default and could stop
      // auto-responding (they don't send an explicit response.create like the manual WS path does).
      turn_detection: opts.autoCreateResponse === false ? null : {
        type: 'semantic_vad',
        threshold: REALTIME.vadThreshold,
        silence_duration_ms: REALTIME.vadSilenceDurationMs,
        create_response: true,
        interrupt_response: false,
      },
      input_audio_transcription: { model: REALTIME.transcriptionModel },
      // Companion (free chat) gets dynamic-knowledge tools; screening stays fully scripted.
      ...(opts.persona === 'companion' ? { tools: COMPANION_TOOLS } : {}),
    },
  }
}

const REALTIME_VOICE_DEFAULT = realtimeVoiceForLanguageKey('english')

/** Realtime WS URL. Prefer the region-adaptive endpoint + model the backend put in the session ticket
 *  (so a SEA device reaches Singapore, a CN device reaches dashscope); fall back to the build-time
 *  defaults when a ticket has not populated them yet. */
export function realtimeWsUrl(endpoint?: string, model?: string): string {
  return `${endpoint || QWEN.realtimeUrl}?model=${model || QWEN.realtimeModel}`
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
