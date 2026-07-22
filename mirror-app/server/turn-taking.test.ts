import assert from 'node:assert/strict'
import test from 'node:test'

import {
  choosePostPlaybackAction,
  playbackDrainDecision,
  replayTurnTaking,
  type TurnTakingEvent,
  type TurnTakingPhase,
} from '../src/orchestration/turnTaking'
import {
  BASE_DAILY_PATIENT_TURNS,
  looksLikeUserGoodbye,
  openingMessageForLanguage,
} from '../src/orchestration/orchestrator'
import { buildLiveSessionUpdate } from '../src/orchestration/realtime'
import { createEnergyVad } from '../src/orchestration/energyVad'
import {
  closingTextForLanguage,
  companionClosingTextForLanguage,
  createDailyConversationPlan,
  dailyConversationMetadataForPatientTurn,
  dailyConversationPatientTurns,
  qwenWavToPcm24kChunks,
  SCREENING_TOTAL_QUESTIONS,
  screeningQuestionForTurn,
} from '../src/orchestration/deterministicSpeech'
import { isCognitiveAssessmentEligible } from '../src/orchestration/conversationPurpose'
import {
  playAndWaitForCompletion,
  type PlaybackCompletionPlayer,
  type PlaybackCompletionStatus,
} from '../src/orchestration/playbackCompletion'
import { acquireConversationRuntime } from '../src/orchestration/conversationRuntime'
import { createPushToTalkGesture } from '../src/orchestration/pushToTalkGesture'
import { secureQwenAssetUrl } from '../src/orchestration/networkSecurity'

const ready: TurnTakingEvent[] = [
  { type: 'connect_started' },
  { type: 'session_configuring' },
  { type: 'session_ready' },
]
const normalResponse: TurnTakingEvent[] = [
  { type: 'response_requested' },
  { type: 'response_created' },
  { type: 'audio_delta' },
  { type: 'response_done' },
  { type: 'playback_drained' },
]

type ReplayCase = {
  name: string
  events: TurnTakingEvent[]
  phase: TurnTakingPhase
  muted: boolean
  violations?: number
}

const replays: ReplayCase[] = [
  { name: 'connect remains muted', events: [{ type: 'connect_started' }], phase: 'connecting', muted: true },
  { name: 'configuration remains muted', events: ready.slice(0, 2), phase: 'configuring', muted: true },
  { name: 'ready opens capture', events: ready, phase: 'listening', muted: false },
  { name: 'user starts speaking', events: [...ready, { type: 'user_speech_started' }], phase: 'user_speaking', muted: false },
  { name: 'user stops speaking', events: [...ready, { type: 'user_speech_started' }, { type: 'user_speech_stopped' }], phase: 'thinking', muted: false },
  { name: 'response request owns capture', events: [...ready, { type: 'response_requested' }], phase: 'assistant_generating', muted: true },
  { name: 'audio keeps capture muted', events: [...ready, ...normalResponse.slice(0, 3)], phase: 'assistant_playing', muted: true },
  { name: 'response done is not playback done', events: [...ready, ...normalResponse.slice(0, 4)], phase: 'playback_guard', muted: true },
  { name: 'playback drained still guards capture', events: [...ready, ...normalResponse], phase: 'playback_guard', muted: true },
  { name: 'guard completion reopens capture', events: [...ready, ...normalResponse, { type: 'mic_reopened' }], phase: 'listening', muted: false },
  { name: 'text-only response drains safely', events: [...ready, { type: 'response_requested' }, { type: 'response_created' }, { type: 'response_done' }, { type: 'playback_drained' }], phase: 'playback_guard', muted: true },
  { name: 'closing response finishes muted', events: [...ready, { type: 'close_requested' }, { type: 'response_requested', closing: true }, { type: 'response_created' }, { type: 'audio_delta' }, { type: 'response_done' }, { type: 'playback_drained' }, { type: 'finished' }], phase: 'ended', muted: true },
  { name: 'failure stays muted', events: [...ready, { type: 'failed', reason: 'network' }], phase: 'error', muted: true },
  { name: 'reset returns to idle', events: [...ready, ...normalResponse, { type: 'reset' }], phase: 'idle', muted: true },
  { name: 'early mic reopen is rejected', events: [...ready, ...normalResponse.slice(0, 4), { type: 'mic_reopened' }], phase: 'playback_guard', muted: true, violations: 1 },
  { name: 'second response is rejected', events: [...ready, { type: 'response_requested' }, { type: 'response_requested' }], phase: 'assistant_generating', muted: true, violations: 1 },
  { name: 'audio without response is rejected', events: [...ready, { type: 'audio_delta' }], phase: 'listening', muted: false, violations: 1 },
  { name: 'duplicate response done is rejected', events: [...ready, ...normalResponse.slice(0, 4), { type: 'response_done' }], phase: 'playback_guard', muted: true, violations: 1 },
  { name: 'duplicate playback drain is rejected', events: [...ready, ...normalResponse, { type: 'playback_drained' }], phase: 'playback_guard', muted: true, violations: 1 },
  { name: 'speech during playback is rejected', events: [...ready, ...normalResponse.slice(0, 3), { type: 'user_speech_started' }], phase: 'assistant_playing', muted: true, violations: 1 },
  { name: 'explicit barge-in transfers the audio turn to the user', events: [...ready, ...normalResponse.slice(0, 3), { type: 'assistant_interrupted' }], phase: 'user_speaking', muted: false },
]

test('21 deterministic lifecycle replays preserve turn ownership', () => {
  assert.equal(replays.length, 21)
  for (const replay of replays) {
    const state = replayTurnTaking(replay.events)
    assert.equal(state.phase, replay.phase, replay.name)
    assert.equal(state.captureMuted, replay.muted, replay.name)
    assert.equal(state.violations.length, replay.violations ?? 0, replay.name)
  }
})

test('post-playback policy preserves screening order and graceful closing', () => {
  const base = {
    persona: 'screening' as const,
    closingResponse: false,
    manualCloseRequested: false,
    spontaneousGoodbye: false,
    dailyFlowComplete: false,
  }

  assert.equal(choosePostPlaybackAction(base), 'listen')
  assert.equal(choosePostPlaybackAction({ ...base, dailyFlowComplete: true }), 'request_goodbye')
  assert.equal(choosePostPlaybackAction({ ...base, manualCloseRequested: true }), 'request_goodbye')
  assert.equal(choosePostPlaybackAction({ ...base, closingResponse: true }), 'finish')
  assert.equal(choosePostPlaybackAction({ ...base, persona: 'companion', spontaneousGoodbye: true }), 'finish')
  assert.equal(choosePostPlaybackAction({ ...base, persona: 'companion', dailyFlowComplete: true }), 'listen')
})

test('daily conversation follows warm-up, yesterday recall, planning, then closes', () => {
  const plan = createDailyConversationPlan({ patientName: 'Margaret', now: new Date('2026-07-22T08:00:00Z'), reminiscenceWeekdays: [] })
  assert.equal(BASE_DAILY_PATIENT_TURNS, 5)
  assert.equal(SCREENING_TOTAL_QUESTIONS, 5)
  assert.equal(dailyConversationPatientTurns(plan), 5)
  assert.equal(openingMessageForLanguage('english', plan.patientName), "Good morning Margaret, it's lovely to see you. How are you feeling today?")
  assert.match(screeningQuestionForTurn('english', 1, plan) ?? '', /dinner yesterday/i)
  assert.match(screeningQuestionForTurn('english', 2, plan) ?? '', /sleep well last night/i)
  assert.match(screeningQuestionForTurn('english', 3, plan) ?? '', /planning to do today/i)
  assert.match(screeningQuestionForTurn('english', 4, plan) ?? '', /visiting you this week/i)
  assert.equal(screeningQuestionForTurn('english', 5, plan), null)
  const allSpeech = [openingMessageForLanguage('english', plan.patientName), 1, 2, 3, 4]
    .map((item) => typeof item === 'number' ? screeningQuestionForTurn('english', item, plan) : item)
    .join(' ')
  assert.doesNotMatch(allSpeech, /\bremember\b|right or wrong|clinic|assessment/i)
})

test('medication and twice-weekly reminiscence are conditional and ordered', () => {
  const medication = { occurrenceId: 'occ-1', displayText: 'afternoon heart tablet', scheduledAt: '2026-07-21T06:00:00.000Z' }
  const tuesday = createDailyConversationPlan({ patientName: 'Margaret', now: new Date('2026-07-21T08:00:00Z'), medicationReminder: medication })
  const wednesday = createDailyConversationPlan({ patientName: 'Margaret', now: new Date('2026-07-22T08:00:00Z'), medicationReminder: medication })
  const friday = createDailyConversationPlan({ patientName: 'Margaret', now: new Date('2026-07-24T08:00:00Z') })

  assert.equal(tuesday.includeReminiscence, true)
  assert.equal(wednesday.includeReminiscence, false)
  assert.equal(friday.includeReminiscence, true)
  assert.equal(dailyConversationPatientTurns(tuesday), 7)
  assert.match(screeningQuestionForTurn('english', 5, tuesday) ?? '', /caregiver has scheduled/i)
  assert.match(screeningQuestionForTurn('english', 5, tuesday) ?? '', /afternoon heart tablet/i)
  assert.doesNotMatch(screeningQuestionForTurn('english', 5, tuesday) ?? '', /dose|mg|milligram/i)
  assert.match(screeningQuestionForTurn('english', 6, tuesday) ?? '', /holiday you loved/i)
  assert.equal(screeningQuestionForTurn('english', 7, tuesday), null)
  assert.match(screeningQuestionForTurn('english', 5, friday) ?? '', /favourite food as a child/i)
})

test('patient speech turns carry protocol stage and signal tags', () => {
  const plan = createDailyConversationPlan({
    now: new Date('2026-07-21T08:00:00Z'),
    medicationReminder: { occurrenceId: 'occ-1', displayText: 'tablet', scheduledAt: '2026-07-21T08:00:00Z' },
  })
  assert.deepEqual(dailyConversationMetadataForPatientTurn(1, plan), {
    protocolStage: 'warm_up', cognitiveSignals: ['mood', 'speech_initiation', 'response_latency'],
  })
  assert.equal(dailyConversationMetadataForPatientTurn(2, plan).protocolStage, 'yesterday_recall')
  assert.equal(dailyConversationMetadataForPatientTurn(4, plan).protocolStage, 'present_planning')
  assert.equal(dailyConversationMetadataForPatientTurn(6, plan).protocolStage, 'medication_reminder')
  assert.equal(dailyConversationMetadataForPatientTurn(7, plan).protocolStage, 'reminiscence')
})

test('playback timeout never masquerades as a safe drain', () => {
  assert.deepEqual(playbackDrainDecision(0, 100, 40, 25_000), { drained: true, timedOut: false })
  assert.deepEqual(playbackDrainDecision(41, 100, 40, 25_000), { drained: false, timedOut: false })
  assert.deepEqual(playbackDrainDecision(-1, 100, 40, 25_000), { drained: true, timedOut: false })
  assert.deepEqual(playbackDrainDecision(500, 25_000, 40, 25_000), { drained: false, timedOut: true })
})

test('rejected empty or echo input returns ownership without creating a response', () => {
  const state = replayTurnTaking([
    ...ready,
    { type: 'user_speech_started' },
    { type: 'user_speech_stopped' },
    { type: 'response_requested' },
    { type: 'input_rejected' },
    { type: 'mic_reopened' },
  ])
  assert.equal(state.phase, 'listening')
  assert.equal(state.captureMuted, false)
  assert.equal(state.responseInFlight, false)
  assert.equal(state.violations.length, 0)
})

test('daily flow prompt and closing update cannot compete', () => {
  const normal = buildLiveSessionUpdate('patient', 'Mandarin', {
    voice: 'ignored', languageKey: 'mandarin', persona: 'screening', autoCreateResponse: false,
  }) as any
  const closing = buildLiveSessionUpdate('patient', 'Mandarin', {
    voice: 'ignored', languageKey: 'mandarin', persona: 'screening',
    wrapUp: true, autoCreateResponse: false,
  }) as any

  assert.match(normal.session.instructions, /three-to-five-minute daily conversation/)
  assert.match(normal.session.instructions, /Never ask the patient to repeat an earlier answer/)
  assert.equal(normal.session.turn_detection, null)
  assert.doesNotMatch(closing.session.instructions, /HIDDEN objectives/)
  assert.match(closing.session.instructions, /end with exactly this sentence: "再见。"/)
  assert.match(closing.session.instructions, /Do not ask a question/)
})

test('local energy VAD waits through short pauses and stops after sustained silence', () => {
  const vad = createEnergyVad()
  const frame = (sample: number) => new Int16Array(1600).fill(sample)

  for (let index = 0; index < 5; index += 1) assert.equal(vad.feed(frame(50)).event, null)
  assert.equal(vad.feed(frame(3000)).event, null)
  assert.equal(vad.feed(frame(3000)).event, 'speech_started')
  for (let index = 0; index < 8; index += 1) assert.equal(vad.feed(frame(0)).event, null)
  assert.equal(vad.feed(frame(3000)).event, null, 'a brief older-adult pause must not end the turn')
  for (let index = 0; index < 11; index += 1) assert.equal(vad.feed(frame(0)).event, null)
  assert.equal(vad.feed(frame(0)).event, 'speech_stopped')
})

test('deterministic positive close cannot open a new screening topic', () => {
  assert.equal(
    closingTextForLanguage('english'),
    'That sounds lovely, thank you so much for chatting with me today. Enjoy your morning! Goodbye.',
  )
  assert.equal(closingTextForLanguage('mandarin'), '听起来真不错，非常感谢你今天和我聊天。祝你今天过得愉快，再见。')
  assert.doesNotMatch(closingTextForLanguage('english'), /\?/)
})

test('companion closes only from explicit user intent and is never cognitively scored', () => {
  assert.equal(looksLikeUserGoodbye('Thanks, that is all for today.'), true)
  assert.equal(looksLikeUserGoodbye('再见，下次再聊。'), true)
  assert.equal(looksLikeUserGoodbye('How can I have a good day?'), false)
  assert.equal(looksLikeUserGoodbye('Please take care to explain my medicine schedule.'), false)
  assert.equal(
    companionClosingTextForLanguage('english'),
    "You're welcome. It was nice talking with you. Goodbye.",
  )
  const patientTurn = [{ id: 'u1', role: 'user' as const, text: 'Hello' }]
  assert.equal(isCognitiveAssessmentEligible('companion', patientTurn), false)
  assert.equal(isCognitiveAssessmentEligible('screening', []), false)
  assert.equal(isCognitiveAssessmentEligible('screening', patientTurn), true)
})

test('release audio upgrades only trusted Qwen asset URLs to HTTPS', () => {
  assert.equal(
    secureQwenAssetUrl('http://dashscope-result-wlcb.oss-cn-wulanchabu.aliyuncs.com/audio.wav'),
    'https://dashscope-result-wlcb.oss-cn-wulanchabu.aliyuncs.com/audio.wav',
  )
  assert.equal(
    secureQwenAssetUrl('https://dashscope-result-wlcb.oss-cn-wulanchabu.aliyuncs.com/audio.wav'),
    'https://dashscope-result-wlcb.oss-cn-wulanchabu.aliyuncs.com/audio.wav',
  )
  assert.equal(secureQwenAssetUrl('http://example.com/audio.wav'), null)
  assert.equal(secureQwenAssetUrl('not-a-url'), null)
})

test('Qwen 24 kHz mono WAV is converted to native PCM playback chunks', () => {
  const payload = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8])
  const wav = new Uint8Array(44 + payload.length)
  const view = new DataView(wav.buffer)
  const write = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) wav[offset + index] = value.charCodeAt(index)
  }
  write(0, 'RIFF'); view.setUint32(4, 36 + payload.length, true); write(8, 'WAVE')
  write(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true)
  view.setUint16(22, 1, true); view.setUint32(24, 24_000, true); view.setUint16(34, 16, true)
  write(36, 'data'); view.setUint32(40, payload.length, true); wav.set(payload, 44)

  const chunks = qwenWavToPcm24kChunks(wav, 4)
  assert.deepEqual(chunks.map((chunk) => Buffer.from(chunk, 'base64')), [
    Buffer.from([1, 2, 3, 4]), Buffer.from([5, 6, 7, 8]),
  ])
})

test('fallback playback waits for native didJustFinish before resolving', async () => {
  let listener: ((status: PlaybackCompletionStatus) => void) | null = null
  let listenerRemoved = false
  let played = false
  const player: PlaybackCompletionPlayer = {
    addListener: (_event, nextListener) => {
      listener = nextListener
      return { remove: () => { listenerRemoved = true } }
    },
    play: () => { played = true },
  }

  let resolved = false
  const completion = playAndWaitForCompletion(player, 1000).then(() => { resolved = true })
  assert.equal(played, true)
  listener?.({ currentTime: 1, duration: 3, isLoaded: true, playing: true })
  await Promise.resolve()
  assert.equal(resolved, false)
  listener?.({ currentTime: 3, didJustFinish: true, duration: 3, isLoaded: true, playing: false })
  await completion
  assert.equal(resolved, true)
  assert.equal(listenerRemoved, true)
})

test('fallback playback errors instead of treating failure as a completed goodbye', async () => {
  let listener: ((status: PlaybackCompletionStatus) => void) | null = null
  const player: PlaybackCompletionPlayer = {
    addListener: (_event, nextListener) => {
      listener = nextListener
      return { remove: () => {} }
    },
    play: () => {},
  }
  const completion = playAndWaitForCompletion(player, 1000)
  listener?.({ error: 'decoder failed' })
  await assert.rejects(completion, /decoder failed/)
})

test('a new conversation lease terminates the old owner and stale release cannot stop the new one', () => {
  let firstTerminated = 0
  let secondTerminated = 0
  const first = acquireConversationRuntime('first-test-owner', () => { firstTerminated += 1 })
  assert.equal(first.isCurrent(), true)

  const second = acquireConversationRuntime('second-test-owner', () => { secondTerminated += 1 })
  assert.equal(firstTerminated, 1)
  assert.equal(first.isCurrent(), false)
  assert.equal(second.isCurrent(), true)

  first.release()
  assert.equal(second.isCurrent(), true, 'a stale cleanup must not release the current owner')
  second.release()
  assert.equal(second.isCurrent(), false)
  assert.equal(secondTerminated, 0)
})

test('superseding a conversation aborts pending native playback immediately', async () => {
  let listenerRemoved = false
  const player: PlaybackCompletionPlayer = {
    addListener: () => ({ remove: () => { listenerRemoved = true } }),
    play: () => {},
  }
  const controller = new AbortController()
  const completion = playAndWaitForCompletion(player, 10_000, controller.signal)
  controller.abort()
  await assert.rejects(completion, /newer conversation/)
  assert.equal(listenerRemoved, true)
})

test('push-to-talk sends once on release and cancels a release during recorder preparation', () => {
  const gesture = createPushToTalkGesture()

  const first = gesture.begin()
  assert.equal(typeof first, 'number')
  assert.equal(gesture.getState(), 'preparing')
  assert.equal(gesture.release(), 'cancelled')
  assert.equal(gesture.ready(first!), false, 'late native prepare must not start recording after release')
  assert.equal(gesture.getState(), 'idle')

  const second = gesture.begin()
  assert.equal(gesture.begin(), null, 'duplicate press-in must be idempotent')
  assert.equal(gesture.ready(second!), true)
  assert.equal(gesture.getState(), 'recording')
  assert.equal(gesture.release(), 'send')
  assert.equal(gesture.release(), 'ignored', 'duplicate press-out must not submit twice')
})

test('push-to-talk discards an accidental tap instead of sending empty audio', () => {
  let now = 1_000
  const gesture = createPushToTalkGesture({ minimumRecordingMs: 250, now: () => now })
  const token = gesture.begin()
  assert.equal(gesture.ready(token!), true)
  now += 80
  assert.equal(gesture.release(), 'discard')

  const heldToken = gesture.begin()
  assert.equal(gesture.ready(heldToken!), true)
  now += 800
  assert.equal(gesture.release(), 'send')
})
