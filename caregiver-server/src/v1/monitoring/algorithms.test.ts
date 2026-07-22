import assert from 'node:assert/strict'
import test from 'node:test'
import { anomalyBand, buildStructuredBaseline, cosineDistance, ewma, mad, median, normalizedCentroid, operationalEligibility, researchEligibility, robustZ, scoreStructured } from './algorithms.js'

test('robust scalar helpers preserve medians, MAD and missingness', () => {
  assert.equal(median([]), undefined)
  assert.equal(median([3, 1, 2]), 2)
  assert.equal(median([4, 1, 2, 3]), 2.5)
  assert.equal(mad([], undefined), undefined)
  assert.equal(mad([1, 2, 3], 2), 1)
  const baseline = buildStructuredBaseline([
    { 'speechAcoustic.speechRate': 2, 'interaction.patientTurns': 7 },
    { 'speechAcoustic.speechRate': 3 },
    { 'speechAcoustic.speechRate': 4, 'interaction.patientTurns': 5 },
  ])
  assert.deepEqual(baseline['speechAcoustic.speechRate'], { median: 3, mad: 1, sampleCount: 3, missingRate: 0, minimum: 2, maximum: 4 })
  assert.ok(Math.abs(baseline['interaction.patientTurns'].missingRate - 1 / 3) < 1e-12)
  assert.equal(Math.round(robustZ(5, baseline['speechAcoustic.speechRate']) * 1000) / 1000, 1.349)
})

test('structured scoring applies registered direction and reason codes', () => {
  const baseline = buildStructuredBaseline(Array.from({ length: 8 }, (_, index) => ({
    'speechAcoustic.speechRate': 3 + (index % 3) * 0.1,
    'speechAcoustic.responseLatencyMs': 800 + (index % 3) * 50,
    'speechLanguage.lexicalDiversity': 0.5 + (index % 3) * 0.02,
    'speechLanguage.wordFindingMarkers': index % 3,
    'interaction.patientTurns': 7 + (index % 3),
    'interaction.interruptionCount': index % 2,
  })))
  const score = scoreStructured({
    'speechAcoustic.speechRate': 2,
    'speechAcoustic.responseLatencyMs': 1500,
    'speechLanguage.lexicalDiversity': 0.2,
    'speechLanguage.wordFindingMarkers': 5,
    'interaction.patientTurns': 3,
    'interaction.interruptionCount': 4,
  }, baseline)
  assert.ok((score.value || 0) > 0.5)
  assert.equal(score.observedFeatures, 6)
  assert.ok(score.reasons.some((reason) => reason.code === 'SPEECH_ACOUSTIC_SPEECH_RATE_DECREASE'))
  assert.equal(scoreStructured({}, baseline).value, undefined)
})

test('operational and research eligibility enforce distinct windows', () => {
  const now = new Date('2026-07-22T12:00:00Z')
  const mature = [...Array.from({ length: 7 }, (_, index) => new Date(now.getTime() - index * 2 * 86_400_000)), new Date(now.getTime() - 20 * 86_400_000)]
  assert.equal(operationalEligibility(mature, now).eligible, true)
  assert.equal(operationalEligibility(mature.map(() => now), now).eligible, false)
  assert.equal(operationalEligibility([], now).completedSessions, 0)

  const research = Array.from({ length: 28 }, (_, index) => new Date(Date.UTC(2026, 5, 1 + index)))
  assert.equal(researchEligibility(research, 1).eligible, true)
  assert.equal(researchEligibility(research.slice(0, 11), 1).eligible, false)
  assert.equal(researchEligibility([], 0).distinctWeeks, 0)
})

test('time-series helpers and bands are deterministic', () => {
  assert.equal(ewma([], 0.1), undefined)
  assert.equal(ewma([10, 20], 0.1), 11)
  assert.equal(anomalyBand(undefined, 1), 'insufficient_data')
  assert.equal(anomalyBand(0.9, 0.4), 'insufficient_data')
  assert.equal(anomalyBand(0.9, 1, 2), 'priority_review')
  assert.equal(anomalyBand(0.7, 1, 2), 'review_recommended')
  assert.equal(anomalyBand(0.4, 1), 'watch')
  assert.equal(anomalyBand(0.1, 1), 'within_personal_range')
})

test('cosine distance and normalized centroid reject incompatible input', () => {
  assert.equal(cosineDistance([], []), undefined)
  assert.equal(cosineDistance([1], [1, 2]), undefined)
  assert.equal(cosineDistance([0], [1]), undefined)
  assert.equal(cosineDistance([1, 0], [1, 0]), 0)
  assert.equal(normalizedCentroid([]), undefined)
  assert.equal(normalizedCentroid([[1], [1, 2]]), undefined)
  assert.equal(normalizedCentroid([[0], [0]]), undefined)
  assert.deepEqual(normalizedCentroid([[1, 0], [0, 1]])?.map((value) => Math.round(value * 1000) / 1000), [0.707, 0.707])
})
