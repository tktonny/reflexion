import assert from 'node:assert/strict'
import test from 'node:test'
import { zonedDateTimeToUtc } from './reminderScheduler.js'

test('zoned reminder conversion respects fixed and daylight-saving offsets', () => {
  assert.equal(zonedDateTimeToUtc(2026, 7, 22, 9, 30, 'Asia/Shanghai').toISOString(), '2026-07-22T01:30:00.000Z')
  assert.equal(zonedDateTimeToUtc(2026, 1, 15, 9, 0, 'America/New_York').toISOString(), '2026-01-15T14:00:00.000Z')
  assert.equal(zonedDateTimeToUtc(2026, 7, 15, 9, 0, 'America/New_York').toISOString(), '2026-07-15T13:00:00.000Z')
})
