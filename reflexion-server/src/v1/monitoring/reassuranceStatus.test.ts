import assert from 'node:assert/strict'
import { test } from 'node:test'
import { evaluateReassuranceStatus, type ReassuranceBaseline, type ReassuranceInputs } from './reassuranceStatus.js'

const agg = (median: number, mad = median * 0.1, sampleCount = 10) => ({ median, mad, sampleCount })

const healthyBaseline: ReassuranceBaseline = {
  patientSpeechMs: agg(180_000),
  patientTurns: agg(6),
  medianResponseLatencyMs: agg(1_200),
  sessionStartMinuteOfDay: agg(8 * 60, 15),
  weeklyRate: 6,
}

const base = (over: Partial<ReassuranceInputs> = {}): ReassuranceInputs => ({
  baselineComplete: true,
  baselineSessionCount: 10,
  completedToday: true,
  today: { patientSpeechMs: 185_000, patientTurns: 6, medianResponseLatencyMs: 1_150, sessionStartMinuteOfDay: 8 * 60 + 10 },
  baseline: healthyBaseline,
  missedStreak: 0,
  weeklyCompletedCount: 6,
  technicalState: 'ok',
  awayActive: false,
  manualFlag: null,
  ...over,
})

test('establishing baseline is reassuring, never red, even with a single missed day', () => {
  const r = evaluateReassuranceStatus(base({ baselineComplete: false, completedToday: false, today: null, baseline: null, missedStreak: 1 }))
  assert.equal(r.status, 'establishing')
  assert.equal(r.primaryReason, 'LEARNING_PERSONAL_ROUTINE')
})

test('missed-streak is Day-1 active even during establishing (3 days -> needs_attention)', () => {
  const r = evaluateReassuranceStatus(base({ baselineComplete: false, completedToday: false, today: null, baseline: null, missedStreak: 3 }))
  assert.equal(r.status, 'needs_attention')
  assert.equal(r.primaryReason, 'CHECKIN_MISSED_3_DAYS')
})

test('stable completed day -> doing_well', () => {
  const r = evaluateReassuranceStatus(base())
  assert.equal(r.status, 'doing_well')
  assert.equal(r.primaryReason, 'DAILY_PATTERN_ON_TRACK')
  assert.ok(r.secondaryReasons.includes('CHECKIN_COMPLETED_TODAY'))
})

test('M4 speech ratio 0.65 -> worth_checking (amber), single day', () => {
  const r = evaluateReassuranceStatus(base({ today: { patientSpeechMs: 180_000 * 0.65, patientTurns: 6, medianResponseLatencyMs: 1_150 } }))
  assert.equal(r.status, 'worth_checking')
  assert.equal(r.primaryReason, 'SPOKE_LESS_THAN_USUAL')
})

test('M4 red only fires with persistence (3 consecutive below-red)', () => {
  const low = { today: { patientSpeechMs: 180_000 * 0.4, patientTurns: 6, medianResponseLatencyMs: 1_150 } }
  const single = evaluateReassuranceStatus(base(low))
  assert.equal(single.status, 'worth_checking', 'one low day is amber, not red')
  const persistent = evaluateReassuranceStatus(base({ ...low, persistence: { speechBelowRed: 2 } }))
  assert.equal(persistent.status, 'needs_attention')
  assert.equal(persistent.primaryReason, 'SPOKE_LESS_THAN_USUAL')
})

test('streak thresholds: 2 -> amber, 3 -> red', () => {
  assert.equal(evaluateReassuranceStatus(base({ completedToday: false, missedStreak: 2 })).primaryReason, 'CHECKIN_MISSED_REPEATEDLY')
  assert.equal(evaluateReassuranceStatus(base({ completedToday: false, missedStreak: 3 })).status, 'needs_attention')
})

test('away period suppresses the missed-streak reason', () => {
  const r = evaluateReassuranceStatus(base({ completedToday: false, missedStreak: 5, awayActive: true }))
  assert.notEqual(r.status, 'needs_attention')
  assert.ok(r.secondaryReasons.includes('AWAY_PERIOD_ACTIVE'))
})

test('guardrail suppresses M4 when baseline is too small to be meaningful', () => {
  const tinyBaseline = { ...healthyBaseline, patientSpeechMs: agg(10_000) } // below 30s guardrail
  const r = evaluateReassuranceStatus(base({ baseline: tinyBaseline, today: { patientSpeechMs: 3_000, patientTurns: 6, medianResponseLatencyMs: 1_150 } }))
  assert.equal(r.status, 'doing_well', 'a low ratio against a meaningless baseline must not fire')
  assert.ok(r.metricEvaluations.some((m) => m.metric === 'M4_speech' && m.verdict === 'neutral'))
})

test('reason priority: significant manual flag outranks 3-day missed streak', () => {
  const r = evaluateReassuranceStatus(base({ completedToday: false, missedStreak: 3, manualFlag: 'needs_attention' }))
  assert.equal(r.status, 'needs_attention')
  assert.equal(r.primaryReason, 'CAREGIVER_FLAG_NEEDS_ATTENTION')
  assert.ok(r.secondaryReasons.includes('CHECKIN_MISSED_3_DAYS'))
})

test('device unreachable with no completion today -> amber technical, not decline', () => {
  const r = evaluateReassuranceStatus(base({ completedToday: false, technicalState: 'unreachable' }))
  assert.equal(r.status, 'worth_checking')
  assert.equal(r.primaryReason, 'DEVICE_UNREACHABLE')
})

test('M7 latency ratio > 1.5 -> amber slower to respond', () => {
  const r = evaluateReassuranceStatus(base({ today: { patientSpeechMs: 185_000, patientTurns: 6, medianResponseLatencyMs: 1_200 * 1.8 } }))
  assert.equal(r.status, 'worth_checking')
  assert.equal(r.primaryReason, 'SLOWER_TO_RESPOND')
})

test('M5 weekly engagement below 0.8 -> amber', () => {
  const r = evaluateReassuranceStatus(base({ weeklyCompletedCount: 3 })) // 3 / max(6,3)=0.5
  assert.equal(r.status, 'worth_checking')
  assert.ok(r.secondaryReasons.includes('WEEKLY_ENGAGEMENT_DOWN') || r.primaryReason === 'WEEKLY_ENGAGEMENT_DOWN')
})
