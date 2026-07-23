import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createSessionTelemetry } from './sessionTelemetry'

test('recorder aggregates speech, latency, reprompts across a two-turn conversation', () => {
  const t = createSessionTelemetry()
  t.reset()
  // Aria turn 1
  t.onAriaResponseCreated(0)
  t.onAriaFirstAudio(100)
  t.onAriaResponseDone(2000)
  t.onAriaPlaybackFinished(2100) // ariaSpeechMs += 2000, lastAriaEnd = 2100
  // Patient turn 1
  t.onUserSpeechStart(2600)
  t.onUserTurn('warm_up', 5600) // speechMs 3000, latency 2600-2100 = 500
  t.onReprompt()
  // Aria turn 2
  t.onAriaResponseCreated(6000)
  t.onAriaFirstAudio(6100)
  t.onAriaPlaybackFinished(8100) // ariaSpeechMs += 2000, lastAriaEnd = 8100
  // Patient turn 2
  t.onUserSpeechStart(8300)
  t.onUserTurn('yesterday_recall', 10300) // speechMs 2000, latency 8300-8100 = 200

  const snap = t.snapshot(0, 10300)
  assert.equal(snap.patientTurns, 2)
  assert.equal(snap.ariaTurns, 2)
  assert.equal(snap.patientSpeechMs, 5000)
  assert.equal(snap.ariaSpeechMs, 4000)
  assert.equal(snap.repromptCount, 1)
  assert.equal(snap.medianResponseLatencyMs, 350) // median([200,500]) rounded
  assert.equal(snap.turns.length, 4)
  const firstPatient = snap.turns.find((turn) => turn.turnId === 'p1')
  assert.equal(firstPatient?.questionId, 'warm_up')
  assert.equal(firstPatient?.responseLatencyMs, 500)
})

test('a no-response turn (no prior Aria playback) contributes no latency', () => {
  const t = createSessionTelemetry()
  t.reset()
  t.onUserSpeechStart(1000)
  t.onUserTurn(undefined, 3000) // no lastAriaPlaybackEnd yet -> latency null
  const snap = t.snapshot(0, 3000)
  assert.equal(snap.medianResponseLatencyMs, null)
  assert.equal(snap.turns[0].responseLatencyMs, null)
  assert.equal(snap.patientSpeechMs, 2000)
})

test('reset clears all accumulators', () => {
  const t = createSessionTelemetry()
  t.onUserSpeechStart(0); t.onUserTurn('x', 1000); t.onReprompt()
  t.reset()
  const snap = t.snapshot(0, 0)
  assert.equal(snap.patientTurns, 0)
  assert.equal(snap.repromptCount, 0)
  assert.equal(snap.turns.length, 0)
})
