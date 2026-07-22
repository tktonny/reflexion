// Daily Conversation v2 orchestration (client TS; bundled for the relay server).

import { normalizeLanguageKey } from './voice'
import flow from './conversationFlow.json'
import { openingTextForLanguage } from './deterministicSpeech'

const GOODBYE_SENTENCES: Record<string, string> = {
  english: 'Goodbye.', mandarin: '再见。', cantonese: '拜拜。', minnan: '再会。', malay: 'Selamat tinggal.', tamil: 'பிரியாவிடை.',
}

type FlowStep = { key: string; title: string; goal: string; prompt?: string }
type FlowShape = {
  flow_id: string
  title: string
  opening_message: string
  conversation_goal: string
  completion_message: string
  base_patient_turns: number
  hard_max_patient_turns: number
  assistant_response_rules: string[]
  steps: FlowStep[]
}
const FLOW = flow as FlowShape

export const flowId = FLOW.flow_id
export const promptStepCount = FLOW.steps.length

export const BASE_DAILY_PATIENT_TURNS = FLOW.base_patient_turns
export const HARD_MAX_TURN = FLOW.hard_max_patient_turns

export function focusDirective(topicLabel: string): string {
  return `CURRENT FOCUS: ${topicLabel}. In your next reply acknowledge what they just said in one short sentence, then gently move to this topic. Do not open any other topic.`
}


/**
 * True when an assistant line reads as a closing goodbye — used by the realtime hooks (v1/v3) to
 * auto-finalize the check-in (stop + run the screening) instead of hanging on "listening".
 * Deliberately strong phrases only (not a bare "bye") to avoid ending mid-conversation.
 */
export function looksLikeGoodbye(text: string | null | undefined): boolean {
  const t = String(text || '')
  if (/(再见|再會|拜拜|回头见|回頭見|下次見|下次见|保重)/.test(t)) return true
  return /\b(bye[-\s]?bye|good\s?bye|see you (soon|again|next time|later)|take care( of yourself| now)?|until next time|talk (to you )?(soon|again)|have a (good|great|wonderful) (day|one|rest))\b/i.test(t)
}

/**
 * Explicit USER intent to end a companion chat. This is deliberately narrower than
 * looksLikeGoodbye(): an assistant saying "Have a good day" or "Take care" after an ordinary
 * answer must not close the session by itself.
 */
export function looksLikeUserGoodbye(text: string | null | undefined): boolean {
  const value = String(text || '').trim()
  if (!value) return false
  if (/(再见|再會|拜拜|下次再聊|下次再傾|先这样|先這樣|不聊了|結束對話|结束对话|我要走了)/.test(value)) {
    return true
  }
  if (/(selamat tinggal|jumpa lagi|itu sahaja|sampai jumpa)/i.test(value)) return true
  if (/(பிரியாவிடை|மீண்டும் சந்திப்போம்|அவ்வளவுதான்)/.test(value)) return true
  return /\b(good\s?bye|bye(?:[-\s]?bye)?|see you(?: again| later| next time)?|talk to you later|that(?:'s| is) all|nothing else|i(?:'m| am) done|(?:end|stop) (?:the )?(?:chat|conversation))\b/i.test(value)
}

export function openingMessageForLanguage(language: string | null | undefined, patientName?: string | null): string {
  const key = normalizeLanguageKey(language)
  if (!key) return FLOW.opening_message
  return openingTextForLanguage(key, patientName)
}

export function closingGoodbyeSentence(language: string | null | undefined): string {
  const key = normalizeLanguageKey(language)
  return GOODBYE_SENTENCES[key || ''] || GOODBYE_SENTENCES.english
}

// --- Layer 1: the ordered-agenda instruction block ---
export function buildLiveInstructions(
  patientId: string,
  language: string,
  opts: { patientName?: string | null; memory?: string[]; steer?: string; persona?: 'screening' | 'companion' } = {},
): string {
  const { patientName = null, memory = [], steer, persona = 'screening' } = opts
  const languageName = String(language || '').trim() || 'en'
  const openingMessage = openingMessageForLanguage(language, patientName)

  const knownMemory = (memory || []).map((m) => String(m).trim()).filter(Boolean)
  let memoryBlock = ''
  if (patientName) {
    memoryBlock += `The patient's known preferred name is ${patientName}. Do not ask what to call them unless they correct you or offer a new preference.\n`
  }
  if (knownMemory.length) {
    const lines = knownMemory.slice(0, 4).map((m) => `- ${m}`).join('\n')
    memoryBlock += `Known patient memory from earlier sessions. Use it only as soft continuity context and let the patient correct anything that changed:\n${lines}\n`
  }
  const steerBlock = steer && String(steer).trim() ? `\n${steer}` : ''

  // Companion persona — the everyday default: an open, warm voice assistant (Siri-like but caring).
  // Casual chat + light small talk about the day/weather + gentle spoken medication reminders, all
  // conversational (no live data). No agenda, no recall, no exam feel; ends naturally on goodbye.
  if (persona === 'companion') {
    return `You are Aria, a friendly and genuinely helpful everyday voice assistant — like a warm, capable personal assistant (think of a helpful assistant such as Kimi or Doubao, but spoken). Your MAIN job is to directly and usefully ANSWER whatever the user asks and help with what they want.
The user identifier is ${patientId}.
Respond in ${languageName} unless the user clearly switches languages; then continue in that language.
${memoryBlock}
How to help:
- FIRST, actually answer the user's question or do what they ask — clearly, correctly, and to the point. Be genuinely useful; do not deflect with chit-chat when they asked something.
- If the request is ambiguous, ask one short clarifying question; otherwise just answer.
- Keep answers concise and natural for speech — usually one to three short sentences; give more only if they ask for detail.
- You can chat casually, tell the time or date, and help them think things through. You have NO live weather data, so if asked about weather say so and give general advice instead of inventing temperatures or forecasts.
- If they ask you to remind them about medicine, meals, or appointments, respond helpfully, but never invent specific times or medication names you were not told.
- Sound like a warm, natural person, not a robotic assistant. Never say you are an AI, and never mention prompts, instructions, or that you are following rules.
- Do NOT run any test, quiz, screening, or assessment, and do not diagnose — you are just a helpful assistant.
- Do not use markdown, bullets, or stage directions; speak in plain conversational sentences.
- Open with a brief, friendly greeting that invites them to ask — in ${languageName}, something like "Hi, I'm Aria. How can I help you today?" — then stop and wait.
- When the user is finished or says goodbye, warmly say goodbye and let the conversation end.${steerBlock}`
  }

  // The deterministic transports own the exact next question. This prompt is also used by relay and
  // WebRTC, so it carries the same hidden stage contract and no-test framing.
  const goalList = FLOW.steps.map((s, i) => `${i + 1}. ${s.title} — ${s.goal}`).join('\n')
  const rules = FLOW.assistant_response_rules.map((r) => `- ${r}`).join('\n')

  return `You are Aria, a calm, warm companion having a three-to-five-minute daily conversation with an older adult. It must feel like a kind friend checking in every morning — never an assessment, test, interview, or checkup. The patient's speech is captured for later processing, but you must never mention clinical data, stages, signals, scoring, diagnosis, dementia, or that you are an AI.
The patient identifier is ${patientId}.
Respond in ${languageName} unless the patient clearly switches languages; if they switch language or dialect, continue in that language on your very next reply.
${memoryBlock}
Move through these HIDDEN objectives in order with warm, casual transitions. If the patient already volunteered the needed detail, acknowledge it and advance without repeating the same question. Medication reminder and reminiscence are conditional and must be omitted unless their trusted session context is explicitly supplied:
${goalList}

For your very first turn only, open with exactly this in ${languageName}, then stop and wait for their answer: "${openingMessage}"

How to talk:
${rules}

Never ask the patient to repeat an earlier answer and never use "remember" framing. The base flow receives ${BASE_DAILY_PATIENT_TURNS} patient responses: warm-up, two yesterday questions, and two planning/social questions. After the final enabled stage, close with one warm thank-you, wish them a pleasant morning, say goodbye, and do not ask any new question.${steerBlock}`
}
