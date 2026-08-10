import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'

// SHADOW ISOLATION (implementation baseline §4, doc §7 "out of backend scope"): the deterministic
// caregiver status must NEVER be driven by research/longitudinal signals. This is enforced
// structurally, not by convention — if a future edit makes the caregiver-status path read an anomaly
// score, the longitudinal baseline, or an embedding, this test fails. Guarding the engine keeps the
// consumer product (green/amber/red reassurance) provably separate from the clinical research shadow.

const here = fileURLToPath(new URL('.', import.meta.url))
const FORBIDDEN = ['anomalyScores', 'baselineModels', 'featureEmbeddings', 'anomaly_scores', 'baseline_models', 'feature_embeddings', 'monitoringWindows']

function sliceFunction(source: string, startMarker: string, endMarker: string) {
  const start = source.indexOf(startMarker)
  assert.notEqual(start, -1, `expected to find ${startMarker}`)
  const end = source.indexOf(endMarker, start + startMarker.length)
  assert.notEqual(end, -1, `expected to find ${endMarker} after ${startMarker}`)
  return source.slice(start, end)
}

test('computeCaregiverStatus reads no research/anomaly collections', () => {
  const source = readFileSync(`${here}../routes/monitoring.ts`, 'utf8')
  // The caregiver-status path spans the function plus its private helpers, up to the serializers.
  const region = sliceFunction(source, 'export async function computeCaregiverStatus', 'function serializeReviewCase')
  for (const token of FORBIDDEN) {
    assert.ok(!region.includes(token), `caregiver status path must not reference "${token}" — shadow isolation violated`)
  }
  // Positive control: it must still read the operational signals it depends on.
  for (const allowed of ['operationalBaselines', 'manualFlags', 'awayPeriods', 'dailyStatuses']) {
    assert.ok(region.includes(allowed), `expected caregiver status to read ${allowed}`)
  }
})

test('the pure reassurance evaluator imports nothing from the research pipeline', () => {
  const source = readFileSync(`${here}reassuranceStatus.ts`, 'utf8')
  for (const token of ['pipeline', 'embeddings', 'anomaly', 'baselineModels']) {
    assert.ok(!source.includes(`from './${token}`), `reassuranceStatus must not import ${token}`)
  }
})
