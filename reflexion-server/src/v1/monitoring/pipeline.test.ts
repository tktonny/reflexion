import assert from 'node:assert/strict'
import test from 'node:test'
import { evaluateSessionQuality } from './pipeline.js'

// The numbers in these cases are real: they come from `quality_assessments` in production on 2026-07-26,
// where the single-tier gate rejected 5 of 8 daily check-ins. A rejected session contributes nothing to the
// operational baseline, which needs 12 usable ones before a caregiver sees any status colour at all.

/** Three patient turns, 30 word-like segments — comfortably over the ideal bar. */
const adequateTurns = [
  { role: 'patient', text: '今天早上我吃了早餐，然后在小区里散步。' },
  { role: 'assistant', text: '后来呢？' },
  { role: 'patient', text: '中午女儿打电话来，我们聊了周末的安排。' },
  { role: 'assistant', text: '今天心情怎么样？' },
  { role: 'patient', text: '心情不错，下午还准备给阳台的花浇水。' },
]

/** Three patient turns, 13 segments — a real but brief answer, between the floor and the ideal bar. */
const briefTurns = [
  { role: 'patient', text: '今天还好，早上吃了粥。' },
  { role: 'assistant', text: '有人来看你吗？' },
  { role: 'patient', text: '女儿有打电话来。' },
  { role: 'assistant', text: '下午打算做什么？' },
  { role: 'patient', text: '下午想去楼下走走。' },
]

/** Three patient turns, 5 segments — monosyllabic, nothing to measure. */
const monosyllabicTurns = [
  { role: 'patient', text: '还好啦。' },
  { role: 'assistant', text: '吃了吗？' },
  { role: 'patient', text: '吃了粥。' },
  { role: 'assistant', text: '还有别的吗？' },
  { role: 'patient', text: '没什么。' },
]

test('a check-in 29 milliseconds under the ideal duration is kept, not discarded', () => {
  // Production: 4 turns, 26 tokens, 14_971 ms — rejected outright, while a session with 22 tokens passed.
  const result = evaluateSessionQuality({ acquisition: { patientSpeechMs: 14_971 } }, adequateTurns)
  assert.equal(result.verdict, 'include_with_caveats', 'a near-miss on one signal must not discard the session')
  assert.deepEqual(result.flags, ['PATIENT_SPEECH_DURATION_INSUFFICIENT'])
})

test('an ample transcript that ran short on the clock is kept', () => {
  // Production: 3 turns, 25 tokens, 13_626 ms — rejected on duration alone despite ample content.
  const result = evaluateSessionQuality({ acquisition: { patientSpeechMs: 13_626 } }, adequateTurns)
  assert.equal(result.verdict, 'include_with_caveats')
})

test('a brief but real answer is kept with a caveat', () => {
  const result = evaluateSessionQuality({ acquisition: { patientSpeechMs: 10_782 } }, briefTurns)
  assert.equal(result.verdict, 'include_with_caveats')
  assert.deepEqual(result.flags.sort(), ['PATIENT_SPEECH_DURATION_INSUFFICIENT', 'TRANSCRIPT_COVERAGE_INSUFFICIENT'])
})

test('below the floor the session is still discarded', () => {
  // Production: 3 turns, 9 tokens, 6_355 ms. Short enough that there is genuinely nothing to measure.
  const result = evaluateSessionQuality({ acquisition: { patientSpeechMs: 6_355 } }, monosyllabicTurns)
  assert.equal(result.verdict, 'repeat_requested')
  assert.ok(result.flags.includes('TRANSCRIPT_COVERAGE_UNUSABLE'))
  assert.ok(result.flags.includes('PATIENT_SPEECH_DURATION_UNUSABLE'))
})

test('each signal contributes at most one flag, so a far miss is not penalised twice', () => {
  const result = evaluateSessionQuality({ acquisition: { patientSpeechMs: 1_000 } }, monosyllabicTurns)
  assert.ok(!result.flags.includes('TRANSCRIPT_COVERAGE_INSUFFICIENT'),
    'the floor flag replaces the ideal-bar flag rather than joining it')
  assert.ok(!result.flags.includes('PATIENT_SPEECH_DURATION_INSUFFICIENT'))
  assert.equal(result.flags.length, 2)
})

test('a session where the person never spoke is discarded', () => {
  const result = evaluateSessionQuality({ acquisition: {} }, [
    { role: 'assistant', text: '早安，今天感觉怎么样？' },
    { role: 'assistant', text: '还在吗？' },
  ])
  assert.equal(result.verdict, 'repeat_requested')
  assert.ok(result.flags.includes('PATIENT_TURNS_INSUFFICIENT'))
  assert.equal(result.scores.patientTurns, 0)
})

test('missing speech duration is a caveat rather than excluding an otherwise usable transcript', () => {
  const result = evaluateSessionQuality({ acquisition: {} }, adequateTurns)
  assert.equal(result.verdict, 'include_with_caveats')
  assert.ok(result.flags.includes('PATIENT_SPEECH_DURATION_UNAVAILABLE'))
  assert.ok(!result.flags.includes('PATIENT_SPEECH_DURATION_INSUFFICIENT'))
})

test('a measured sample under the floor still requests a repeat', () => {
  const result = evaluateSessionQuality({ acquisition: { patientSpeechMs: 5_000 } }, adequateTurns)
  assert.equal(result.verdict, 'repeat_requested')
  assert.ok(result.flags.includes('PATIENT_SPEECH_DURATION_UNUSABLE'))
})

test('a full check-in passes with no flags at all', () => {
  const result = evaluateSessionQuality({ acquisition: { patientSpeechMs: 18_998 } }, adequateTurns)
  assert.equal(result.verdict, 'include')
  assert.deepEqual(result.flags, [])
  assert.equal(result.scores.overall, 1)
})

test('a language mismatch and a talkative caregiver keep their existing severities', () => {
  const mismatch = evaluateSessionQuality({ acquisition: { patientSpeechMs: 18_998, languageMismatch: true } }, adequateTurns)
  assert.equal(mismatch.verdict, 'repeat_requested', 'the wrong language is still not analysable')

  const talkative = evaluateSessionQuality({ acquisition: { patientSpeechMs: 18_998, caregiverSpeechRatio: 0.5 } }, adequateTurns)
  assert.equal(talkative.verdict, 'include_with_caveats', 'someone else answering is a caveat, not a rejection')
  assert.deepEqual(talkative.flags, ['CAREGIVER_SPEECH_HIGH'])
})

test('the production sample that motivated this now yields 7 of 8 usable', () => {
  // Every daily_checkin quality_assessment from production on 2026-07-25/26, as (turns, tokens, ms).
  // Reproduced through the real function by picking a fixture with the matching token tier.
  const sample: [typeof adequateTurns, number][] = [
    [adequateTurns, 18_998], [adequateTurns, 16_014], [adequateTurns, 15_508],
    [adequateTurns, 14_971], [adequateTurns, 13_626], [briefTurns, 10_782],
    [briefTurns, 9_058], [monosyllabicTurns, 6_355],
  ]
  const usable = sample
    .map(([turns, ms]) => evaluateSessionQuality({ acquisition: { patientSpeechMs: ms } }, turns))
    .filter((result) => result.verdict !== 'repeat_requested')
  assert.equal(usable.length, 7, 'only the monosyllabic 6-second session should be discarded')
})
