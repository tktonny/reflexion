import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createDailyConversationPlan } from './deterministicSpeech'
import { buildDailyCheckinScript, createDailyCheckinFlow, createSessionCheckinFlow } from './dailyCheckinFlow'

const withMedAndReminiscence = createDailyConversationPlan({
  patientName: 'Margaret',
  medicationReminder: { occurrenceId: 'occ_1', displayText: 'morning tablets', scheduledAt: '2026-07-23T01:00:00Z' },
  reminiscenceWeekdays: [new Date('2026-07-23T00:00:00Z').getDay()], // force reminiscence on
  now: new Date('2026-07-23T00:00:00Z'),
})

test('script has stable ids, fixed order, and the medication + reminiscence tail', () => {
  const script = buildDailyCheckinScript('english', withMedAndReminiscence)
  assert.deepEqual(script.map((q) => q.questionId), ['yesterday_dinner', 'yesterday_sleep', 'today_plan', 'week_visit', 'medication', 'reminiscence'])
  assert.deepEqual(script.map((q) => q.order), [1, 2, 3, 4, 5, 6])
  const med = script.find((q) => q.questionId === 'medication')!
  assert.equal(med.required, true)
  assert.equal(med.skippable, false)
  assert.equal(med.stage, 'medication_reminder')
})

test('answers in fixed order and completes only after the last question', () => {
  const flow = createSessionCheckinFlow('english', withMedAndReminiscence)
  const seen: string[] = []
  for (let guard = 0; guard < 20 && !flow.isComplete(); guard += 1) {
    const q = flow.current()!
    seen.push(q.questionId)
    flow.markAsked()
    assert.equal(flow.recordAnswer('a proper full sentence answer'), 'answered')
  }
  assert.deepEqual(seen, ['yesterday_dinner', 'yesterday_sleep', 'today_plan', 'week_visit', 'medication', 'reminiscence'])
  assert.equal(flow.isComplete(), true)
  assert.equal(flow.current(), null)
})

test('a too-short answer is insufficient and does not advance', () => {
  const flow = createSessionCheckinFlow('english', createDailyConversationPlan({ reminiscenceWeekdays: [] }))
  const first = flow.current()!
  assert.equal(flow.recordAnswer('ok'), 'insufficient') // < 3 words
  assert.equal(flow.current()!.questionId, first.questionId, 'stays on the same question')
  assert.equal(flow.recordAnswer('yes I had fish and rice'), 'answered')
  assert.notEqual(flow.current()!.questionId, first.questionId, 'advances after a real answer')
})

test('reprompt budget: one reprompt then skip for a conversational question', () => {
  const flow = createSessionCheckinFlow('english', createDailyConversationPlan({ reminiscenceWeekdays: [] }))
  const q = flow.current()!
  assert.equal(flow.recordRepromptOrTimeout(), 'reprompt') // 1st (maxReprompts=1)
  assert.equal(flow.current()!.questionId, q.questionId)
  assert.equal(flow.recordRepromptOrTimeout(), 'skip') // budget spent -> skip
  assert.notEqual(flow.current()?.questionId, q.questionId)
  assert.equal(flow.snapshot().skippedCount, 1)
})

test('a required non-skippable question cannot wedge the flow', () => {
  const medOnly = createDailyConversationPlan({
    medicationReminder: { occurrenceId: 'o', displayText: 'pills', scheduledAt: '2026-07-23T01:00:00Z' },
    reminiscenceWeekdays: [],
  })
  const flow = createSessionCheckinFlow('english', medOnly)
  // Drive to the medication question.
  for (let i = 0; i < 4; i += 1) flow.recordAnswer('a full proper answer here')
  const med = flow.current()!
  assert.equal(med.questionId, 'medication')
  // Exhaust its (larger) reprompt budget; it must eventually complete, never loop forever.
  let action = ''
  for (let i = 0; i < 10 && !flow.isComplete(); i += 1) action = flow.recordRepromptOrTimeout()
  assert.equal(flow.isComplete(), true)
  assert.ok(action === 'complete' || action === 'skip')
})

test('CJK answers count characters, not whitespace tokens', () => {
  const flow = createSessionCheckinFlow('mandarin', createDailyConversationPlan({ reminiscenceWeekdays: [] }))
  assert.equal(flow.recordAnswer('我昨天吃了鱼'), 'answered') // 6 CJK chars >= 3
})
