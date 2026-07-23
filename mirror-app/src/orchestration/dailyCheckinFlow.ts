// Per-question deterministic state machine for the Daily Check-in (implementation baseline §7 Phase 1
// #1; doc "Daily Conversation Flow"). The LLM produces natural speech; THIS drives correctness — fixed
// order, per-question required/skip/timeout/reprompt rules, an explicit answered condition, and a
// completion condition. Pure and clock-injected so it is fully unit-testable and can be exercised
// headlessly (e.g. via the phase1 audio injector) on the emulator without a physical device.

import {
  type DailyConversationPlan,
  createDailyConversationPlan,
  screeningQuestionForTurn,
} from './deterministicSpeech'
import type { LanguageKey } from './voice'

export type CheckinStage = 'warm_up' | 'yesterday_recall' | 'present_planning' | 'medication_reminder' | 'reminiscence'
export type QuestionState = 'pending' | 'asked' | 'answered' | 'skipped'

export type CheckinQuestion = {
  questionId: string
  order: number
  stage: CheckinStage
  cognitiveSignals: string[]
  prompt: string
  required: boolean
  maxReprompts: number
  timeoutMs: number
  skippable: boolean
  minWordsToAnswer: number
}

// Stable per-order question ids for the 4 base recall/planning questions (warm-up is the opening line,
// not part of this scripted list). The medication + reminiscence tail get their own ids.
const BASE_QUESTION_IDS = ['yesterday_dinner', 'yesterday_sleep', 'today_plan', 'week_visit'] as const

const STAGE_SIGNALS: Record<CheckinStage, string[]> = {
  warm_up: ['mood', 'speech_initiation', 'response_latency'],
  yesterday_recall: ['episodic_memory', 'temporal_orientation', 'narrative_coherence'],
  present_planning: ['executive_function', 'prospective_memory', 'social_connectedness'],
  medication_reminder: ['memory', 'caregiver_adjunct'],
  reminiscence: ['semantic_memory', 'language_richness', 'lexical_diversity', 'speech_fluency'],
}

const DEFAULTS = { maxReprompts: 1, timeoutMs: 9_000, minWordsToAnswer: 3 }

// Word count that tolerates CJK (a whole Mandarin sentence is not "one word").
function wordCount(text: string): number {
  const trimmed = text.trim()
  if (!trimmed) return 0
  const cjk = (trimmed.match(/[㐀-鿿぀-ヿ가-힯]/g) || []).length
  const rest = trimmed.replace(/[㐀-鿿぀-ヿ가-힯]/g, ' ').trim()
  return cjk + (rest ? rest.split(/\s+/).filter(Boolean).length : 0)
}

/** Builds the ordered, structured question script for a session's language + plan. Single source of
 *  truth for question identity/metadata; the prompt text reuses the existing localized copy. */
export function buildDailyCheckinScript(language: LanguageKey, plan: DailyConversationPlan): CheckinQuestion[] {
  // Iterate the real question generator until it stops, so the order (incl. the medication/reminiscence
  // tail position) always matches screeningQuestionForTurn exactly and can never drift.
  const script: CheckinQuestion[] = []
  let optionalSeen = 0
  for (let turn = 1; turn <= 12; turn += 1) {
    const prompt = screeningQuestionForTurn(language, turn, plan)
    if (!prompt) break
    if (turn <= BASE_QUESTION_IDS.length) {
      const stage: CheckinStage = turn <= 2 ? 'yesterday_recall' : 'present_planning'
      script.push({
        questionId: BASE_QUESTION_IDS[turn - 1], order: turn, stage, cognitiveSignals: STAGE_SIGNALS[stage],
        prompt, required: false, maxReprompts: DEFAULTS.maxReprompts, timeoutMs: DEFAULTS.timeoutMs,
        skippable: true, minWordsToAnswer: DEFAULTS.minWordsToAnswer,
      })
      continue
    }
    // Optional tail: medication (if scheduled) is always the first optional slot, then reminiscence.
    const isMedication = plan.medicationReminder && optionalSeen === 0
    optionalSeen += 1
    if (isMedication) {
      script.push({
        questionId: 'medication', order: turn, stage: 'medication_reminder', cognitiveSignals: STAGE_SIGNALS.medication_reminder,
        // Medication is the top-WTP question — it should get an answer; allow one extra reprompt and no skip.
        prompt, required: true, maxReprompts: 2, timeoutMs: 12_000, skippable: false, minWordsToAnswer: 1,
      })
    } else {
      script.push({
        questionId: 'reminiscence', order: turn, stage: 'reminiscence', cognitiveSignals: STAGE_SIGNALS.reminiscence,
        prompt, required: false, maxReprompts: DEFAULTS.maxReprompts, timeoutMs: DEFAULTS.timeoutMs,
        skippable: true, minWordsToAnswer: DEFAULTS.minWordsToAnswer,
      })
    }
  }
  return script
}

export type CheckinFlowSnapshot = {
  questionId: string | null
  order: number
  askedCount: number
  answeredCount: number
  skippedCount: number
  complete: boolean
}

/** Deterministic driver over a script. The hook feeds it answer/reprompt/timeout events; it decides
 *  reprompt-vs-skip-vs-advance and when the flow is complete. It never generates speech itself. */
export function createDailyCheckinFlow(script: CheckinQuestion[]) {
  const states: QuestionState[] = script.map(() => 'pending')
  const reprompts: number[] = script.map(() => 0)
  let index = 0

  function advance() {
    while (index < script.length && (states[index] === 'answered' || states[index] === 'skipped')) index += 1
  }

  return {
    script,
    /** The question to ask now, or null when the flow is complete. */
    current(): CheckinQuestion | null {
      advance()
      return index < script.length ? script[index] : null
    },
    markAsked() {
      advance()
      if (index < script.length && states[index] === 'pending') states[index] = 'asked'
    },
    /** Record the user's answer. Returns 'answered' if it satisfied the completion condition (advances),
     *  or 'insufficient' if it did not (the caller may reprompt). */
    recordAnswer(text: string): 'answered' | 'insufficient' {
      advance()
      if (index >= script.length) return 'answered'
      const question = script[index]
      if (wordCount(text) >= question.minWordsToAnswer) {
        states[index] = 'answered'
        index += 1
        advance()
        return 'answered'
      }
      return 'insufficient'
    },
    /** Record a gentle reprompt or a no-response timeout. Returns the next action for the current
     *  question: 'reprompt' (ask once more), or 'skip'/'complete' when the reprompt budget is spent. */
    recordRepromptOrTimeout(): 'reprompt' | 'skip' | 'complete' {
      advance()
      if (index >= script.length) return 'complete'
      const question = script[index]
      reprompts[index] += 1
      if (reprompts[index] <= question.maxReprompts) return 'reprompt'
      // Budget spent: skip if allowed, else mark answered-as-skipped so a required question can't wedge.
      states[index] = 'skipped'
      index += 1
      advance()
      return index >= script.length ? 'complete' : 'skip'
    },
    isComplete(): boolean {
      advance()
      return index >= script.length
    },
    repromptsFor(questionId: string): number {
      const i = script.findIndex((question) => question.questionId === questionId)
      return i >= 0 ? reprompts[i] : 0
    },
    snapshot(): CheckinFlowSnapshot {
      advance()
      const currentQuestion = index < script.length ? script[index] : null
      return {
        questionId: currentQuestion?.questionId ?? null,
        order: currentQuestion?.order ?? script.length + 1,
        askedCount: states.filter((state) => state !== 'pending').length,
        answeredCount: states.filter((state) => state === 'answered').length,
        skippedCount: states.filter((state) => state === 'skipped').length,
        complete: index >= script.length,
      }
    },
  }
}

export type DailyCheckinFlow = ReturnType<typeof createDailyCheckinFlow>

/** Convenience: build the flow for a session in one call. */
export function createSessionCheckinFlow(language: LanguageKey, plan = createDailyConversationPlan({ reminiscenceWeekdays: [] })) {
  return createDailyCheckinFlow(buildDailyCheckinScript(language, plan))
}
