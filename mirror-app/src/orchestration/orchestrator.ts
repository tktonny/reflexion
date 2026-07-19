// Conversation orchestration (client TS port of server/orchestrator.mjs).
// Builds the hidden staged-plan instruction block used as the LLM system prompt.
// Verified end-to-end against Qwen via server/smoke-turnloop.mjs.

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

const NO_MARKDOWN_RULE = 'Do not use markdown, asterisks, underscores, bullets, numbered lists, or stage directions. Speak in plain conversational sentences only.'
const NO_PREMATURE_GOODBYE_RULE = 'Do not say goodbye, do not say the session is ending, and do not act like you are closing the conversation unless you are explicitly told that the live capture is ending now.'

type FlowStep = {
  key: string; title: string; goal: string; prompt: string; rationale: string
  guided_transition?: string; exit_when: string[]; max_follow_ups: number
}
type Flow = {
  flow_id: string; title: string; opening_message: string; conversation_goal: string
  completion_rule: string; assistant_response_rules: string[]; steps: FlowStep[]
}
const FLOW = flow as Flow

export const flowId = FLOW.flow_id
export const promptStepCount = FLOW.steps.length

export function openingMessageForLanguage(language: string | null | undefined): string {
  const key = normalizeLanguageKey(language)
  if (!key) return FLOW.opening_message
  return OPENING_MESSAGES[key] || FLOW.opening_message
}

export function closingGoodbyeSentence(language: string | null | undefined): string {
  const key = normalizeLanguageKey(language)
  return GOODBYE_SENTENCES[key || ''] || GOODBYE_SENTENCES.english
}

function formatStageBlock(index: number, step: FlowStep): string {
  const exitRules = (step.exit_when || []).map((r) => `  - ${r}`).join('\n') || '  - Move on after one answer.'
  const transitionLine = step.guided_transition ? `  Natural bridge: ${step.guided_transition}\n` : ''
  return (
    `${index}. ${step.title} (${step.key})\n` +
    `  Goal: ${step.goal}\n` +
    transitionLine +
    `  Primary prompt: ${step.prompt}\n` +
    `  Rationale: ${step.rationale}\n` +
    `  Exit when:\n${exitRules}\n` +
    `  If needed, ask at most ${step.max_follow_ups} gentle clarification question(s) before moving on.`
  )
}

// MVP: no on-device identity yet, so patientName/memory default to empty (first-time patient).
export function buildLiveInstructions(
  patientId: string,
  language: string,
  opts: { patientName?: string | null; memory?: string[] } = {},
): string {
  const { patientName = null, memory = [] } = opts
  const languageName = String(language || '').trim() || 'en'
  const openingMessage = openingMessageForLanguage(language)

  let responseRules = FLOW.assistant_response_rules && FLOW.assistant_response_rules.length
    ? [...FLOW.assistant_response_rules]
    : [
        'Sound like a calm, warm human guide rather than a robotic assistant.',
        'Treat the stage plan as hidden objectives, not lines to recite.',
        'Keep replies extremely brief: usually one short sentence, never more than one short sentence plus one short question.',
        'Do not diagnose, score risk, or discuss dementia probability during the live conversation.',
        'After the final stage is complete, thank the patient and say the session is complete.',
      ]
  if (!responseRules.includes(NO_MARKDOWN_RULE)) responseRules = [...responseRules, NO_MARKDOWN_RULE]
  if (!responseRules.includes(NO_PREMATURE_GOODBYE_RULE)) responseRules = [...responseRules, NO_PREMATURE_GOODBYE_RULE]

  const stageBlocks = FLOW.steps.map((step, i) => formatStageBlock(i + 1, step)).join('\n\n')
  const ruleLines = responseRules.map((r) => `- ${r}`).join('\n')

  const knownMemory = (memory || []).map((m) => String(m).trim()).filter(Boolean)
  let memoryBlock = ''
  if (patientName) {
    memoryBlock += `The patient's known preferred name is ${patientName}. Do not ask what to call them unless they correct you or offer a new preference.\n`
  }
  if (knownMemory.length) {
    const lines = knownMemory.slice(0, 4).map((m) => `- ${m}`).join('\n')
    memoryBlock += `Known patient memory from earlier sessions. Use it only as soft continuity context and let the patient correct anything that changed:\n${lines}\n`
  }

  return (
    'You are Reflexion, a calm and natural conversation guide conducting a short clinical intake.\n' +
    `The patient identifier is ${patientId}.\n` +
    `Respond in ${languageName} unless the patient clearly switches languages.\n` +
    'If the patient switches languages or dialects, immediately continue in that language on the next turn.\n' +
    memoryBlock +
    `Conversation flow: ${FLOW.title}.\n` +
    `Conversation goal: ${FLOW.conversation_goal}\n` +
    `Completion rule: ${FLOW.completion_rule}\n` +
    'The stage plan below is hidden guidance. The patient should feel like they are in a natural human conversation, not a checklist.\n' +
    'Never mention stage names, scoring, assessment logic, or that you are an AI.\n' +
    `For your first turn only, say exactly this opening in ${languageName}: "${openingMessage}"\n` +
    "After the opening question, stop and wait for the patient's answer.\n" +
    'Live response rules:\n' +
    `${ruleLines}\n` +
    'Staged plan:\n' +
    `${stageBlocks}`
  )
}
