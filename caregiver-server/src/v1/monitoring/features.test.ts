import assert from 'node:assert/strict'
import test from 'node:test'
import { extractStructuredFeatures, flattenFeatureGroups } from './features.js'

test('feature extraction uses measured speech and real transcript only', () => {
  const turns = [
    { role: 'assistant', text: 'How are you?', startedAt: new Date('2026-01-01T00:00:00Z'), endedAt: new Date('2026-01-01T00:00:02Z') },
    { role: 'patient', text: '那个，我今天很好', startedAt: new Date('2026-01-01T00:00:03Z'), endedAt: new Date('2026-01-01T00:00:05Z') },
    { role: 'assistant', text: 'Tell me more.', endedAt: new Date('2026-01-01T00:00:06Z') },
    { role: 'patient', text: 'I went walking, um, walking outside.', startedAt: new Date('2026-01-01T00:00:08Z') },
  ]
  const result = extractStructuredFeatures(turns, { patientSpeechMs: 30_000, interruptionCount: 2 })
  const flat = result.flattened
  assert.equal(flat['interaction.patientTurns'], 2)
  assert.equal(flat['interaction.interruptionCount'], 2)
  assert.equal(flat['speechAcoustic.responseLatencyMs'], 1500)
  assert.ok((flat['speechAcoustic.speechRate'] || 0) > 0)
  assert.equal(flat['speechLanguage.wordFindingMarkers'], 2)
  assert.deepEqual(flattenFeatureGroups(result.featureGroups), flat)
})

test('missing measurements remain absent instead of becoming zero', () => {
  const result = extractStructuredFeatures([{ role: 'assistant', text: 'hello' }])
  assert.equal(result.flattened['speechAcoustic.speechRate'], undefined)
  assert.equal(result.flattened['interaction.patientTurns'], undefined)
  assert.deepEqual(flattenFeatureGroups({ bad: null, array: [], text: 'x', group: { valid: 1, missing: null } }), { 'group.valid': 1 })
})
