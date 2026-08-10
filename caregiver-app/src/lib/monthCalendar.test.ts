import assert from 'node:assert/strict';
import test from 'node:test';

import { buildMonthCalendar, monthLabel, shiftMonth } from './monthCalendar';

test('month navigation crosses year boundaries in both directions', () => {
  assert.equal(shiftMonth('2026-12', 1), '2027-01');
  assert.equal(shiftMonth('2027-01', -1), '2026-12');
  assert.equal(shiftMonth('2026-08', 3), '2026-11');
  assert.equal(shiftMonth('2026-08', -8), '2025-12');
});

test('calendar cells keep Monday-first weekday positioning', () => {
  const cells = buildMonthCalendar('2025-05');
  assert.equal(cells.length, 35);
  assert.deepEqual(cells.slice(0, 4).map((cell) => [cell.date, cell.inMonth]), [
    ['2025-04-28', false],
    ['2025-04-29', false],
    ['2025-04-30', false],
    ['2025-05-01', true],
  ]);
  assert.equal(cells.filter((cell) => cell.inMonth).length, 31);
});

test('calendar handles leap years and exposes a stable month label', () => {
  assert.equal(buildMonthCalendar('2024-02').filter((cell) => cell.inMonth).length, 29);
  assert.match(monthLabel('2026-08'), /August 2026/);
});
