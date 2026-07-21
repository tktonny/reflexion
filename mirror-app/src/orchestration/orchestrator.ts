// Conversation orchestration (client TS; twin of server/orchestrator.mjs).
// Builds the LLM system prompt that drives the hidden 4-topic agenda.
// v2 redesign (hybrid): Layer 1 = strengthened ordered-agenda prompt (kills small-talk drift,
// mandates the recall step); Layer 2 = a deterministic turn-count "recall floor" (recallBudgetStep
// + RECALL/WRAPUP directives injected as `steer`) wired into each version so recall ALWAYS happens.

import { normalizeLanguageKey } from './voice'
import flow from './conversationFlow.json'

const OPENING_MESSAGES: Record<string, string> = {
  english: 'Hi, nice to meet you. What should I call you? And where are you right now?',
  mandarin: '你好，很高兴见到你。我该怎么称呼你？你现在在哪里？',
  cantonese: '你好，好高兴见到你。我应该点称呼你？你而家喺边度？',
  minnan: '你好，很欢喜见着你。我欲按怎称呼你？你这马佇佗位？',
  malay: 'Hai, gembira bertemu dengan anda. Saya patut panggil anda apa? Dan sekarang anda berada di mana?',
  tamil: 'வணக்கம், உங்களை சந்தித்ததில் மகிழ்ச்சி. நான் உங்களை எப்படி அழைக்கலாம்? நீங்கள் இப்போது எங்கே இருக்கிறீர்கள்?',
}

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
  assistant_response_rules: string[]
  steps: FlowStep[]
}
const FLOW = flow as FlowShape

export const flowId = FLOW.flow_id
export const promptStepCount = 4

// --- Layer 2: shared deterministic recall floor (single source of truth for all 3 versions) ---
// Kept deliberately short so the check-in doesn't drag: force the recall step by the 3rd patient
// turn, and hard-cap the whole conversation at 5 turns before wrapping up.
export const RECALL_DEADLINE_TURN = 3
export const HARD_MAX_TURN = 5

export const RECALL_DIRECTIVE =
  'PRIORITY RIGHT NOW: You have spent enough time on the earlier topics. In your very next reply, do the gentle recall step now: warmly bring back one specific thing the patient actually said earlier in this same conversation, name that real detail, and ask them in one short sentence to tell you about it again. Do not open any new topic and do not say goodbye yet; wait for their answer.'

export const WRAPUP_DIRECTIVE =
  'PRIORITY RIGHT NOW: The recall step is finished. In your next reply, briefly and warmly thank the patient and end with one short goodbye sentence. Do not ask another question after the goodbye.'

export function focusDirective(topicLabel: string): string {
  return `CURRENT FOCUS: ${topicLabel}. In your next reply acknowledge what they just said in one short sentence, then gently move to this topic. Do not open any other topic.`
}

export type RecallBudgetState = { turnCount: number; recallProbeIssued: boolean; recallAnswered: boolean }
export type RecallBudgetAction = { action: 'none' | 'force_recall' | 'wrap_up' }

/** Evaluate once per COMPLETED patient turn. Fires from the turn counter regardless of model behaviour. */
export function recallBudgetStep(s: RecallBudgetState): RecallBudgetAction {
  if (s.recallProbeIssued && !s.recallAnswered) return { action: 'wrap_up' }
  if (!s.recallProbeIssued && s.turnCount >= RECALL_DEADLINE_TURN) return { action: 'force_recall' }
  if (s.turnCount >= HARD_MAX_TURN) return { action: s.recallProbeIssued ? 'wrap_up' : 'force_recall' }
  return { action: 'none' }
}

/** Heuristic: did the assistant reply look like a recall probe? (soft dedupe so the floor stays dormant when Layer 1 works) */
export function looksLikeRecallProbe(text: string | null | undefined): boolean {
  return /mention|earlier|told me|talk(ed)? about|said (before|earlier)|brought up|think back|remember/i.test(String(text || ''))
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

export function openingMessageForLanguage(language: string | null | undefined): string {
  const key = normalizeLanguageKey(language)
  if (!key) return FLOW.opening_message
  return OPENING_MESSAGES[key] || FLOW.opening_message
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
  const openingMessage = openingMessageForLanguage(language)

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

  // Goal-driven, NON-scripted design (ported from platform_April's realtime_conversation_flow.json).
  // The stages are HIDDEN objectives reached through natural small talk — deliberately NOT a rigid
  // ordered agenda to recite, which is what made the conversation robotic and repetitive.
  const goalList = FLOW.steps.map((s, i) => `${i + 1}. ${s.title} — ${s.goal}`).join('\n')
  const rules = FLOW.assistant_response_rules.map((r) => `- ${r}`).join('\n')

  return `You are Reflexion, a calm, warm companion having a short, natural daily check-in with an older adult. It should feel like a friendly chat with a kind person — never an assessment, test, interview, or checkup. Never mention stages, topics, scoring, or that you are an AI.
The patient identifier is ${patientId}.
Respond in ${languageName} unless the patient clearly switches languages; if they switch language or dialect, continue in that language on your very next reply.
${memoryBlock}
Your goal is to gently reach these HIDDEN objectives through natural conversation — treat them as things to get to warmly and casually, NOT a checklist to recite and NOT a fixed order to march through:
${goalList}

For your very first turn only, open with exactly this in ${languageName}, then stop and wait for their answer: "${openingMessage}"

How to talk:
${rules}

Near the end, do the wrap-up recall gently: warmly bring back one real thing the patient actually mentioned earlier in this same chat and invite them to say a little more about it — refer only to a real detail they gave, never an invented one, and never make it feel like a memory test. Accept whatever they recall, full, partial, or none, with warmth. Once they have responded to that, close naturally: one short, warm thank-you and a brief goodbye sentence, and do not ask any new question after that.${steerBlock}`
}
