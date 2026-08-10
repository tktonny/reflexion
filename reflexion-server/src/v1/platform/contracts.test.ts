import assert from 'node:assert/strict'
import test from 'node:test'

import { reviewCaseNotificationDedupeKey } from '../notifications/service.js'
import { serializeProcessingStatus } from '../routes/sessions.js'

test('processing status exposes a stable queued state without internal details', () => {
  assert.deepEqual(serializeProcessingStatus({
    _id: 'ses_1', operationId: 'op_1', state: 'ingesting', createdAt: new Date('2026-07-22T00:00:00Z'),
  }), {
    sessionId: 'ses_1', operationId: 'op_1', state: 'queued', stage: 'queued', retryable: false,
    result: null, updatedAt: '2026-07-22T00:00:00.000Z',
  })
})

test('processing status preserves safe completed outcome fields', () => {
  const value = serializeProcessingStatus({
    _id: 'ses_2', operationId: 'op_2', state: 'review_pending', updatedAt: new Date('2026-07-22T01:00:00Z'),
    processingSummary: { stage: 'complete', inclusion: 'include', anomalyState: 'review_recommended', privateVector: [1, 2] },
  })
  assert.equal(value.state, 'completed')
  assert.deepEqual(value.result, {
    outcome: 'review_pending', inclusion: 'include', monitoringUse: undefined, anomalyState: 'review_recommended',
  })
  assert.equal('privateVector' in (value.result || {}), false)
})

test('review-case notification key is deterministic for delivery deduplication', () => {
  assert.equal(reviewCaseNotificationDedupeKey('case_abc'), 'review_case:case_abc')
})
