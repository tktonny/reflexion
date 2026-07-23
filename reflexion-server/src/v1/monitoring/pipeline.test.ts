import assert from 'node:assert/strict'
import test from 'node:test'
import { evaluateSessionQuality } from './pipeline.js'

const adequateTurns = [
  { role: 'patient', text: '今天早上我吃了早餐，然后在小区里散步。' },
  { role: 'assistant', text: '后来呢？' },
  { role: 'patient', text: '中午女儿打电话来，我们聊了周末的安排。' },
  { role: 'assistant', text: '今天心情怎么样？' },
  { role: 'patient', text: '心情不错，下午还准备给阳台的花浇水。' },
]

test('missing speech duration is a caveat rather than excluding an otherwise usable transcript', () => {
  const result = evaluateSessionQuality({ acquisition: {} }, adequateTurns)
  assert.equal(result.verdict, 'include_with_caveats')
  assert.ok(result.flags.includes('PATIENT_SPEECH_DURATION_UNAVAILABLE'))
  assert.ok(!result.flags.includes('PATIENT_SPEECH_DURATION_INSUFFICIENT'))
})

test('a measured but too-short sample still requests a repeat', () => {
  const result = evaluateSessionQuality({ acquisition: { patientSpeechMs: 5_000 } }, adequateTurns)
  assert.equal(result.verdict, 'repeat_requested')
  assert.ok(result.flags.includes('PATIENT_SPEECH_DURATION_INSUFFICIENT'))
})
