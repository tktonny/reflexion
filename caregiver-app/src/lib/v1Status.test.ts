import assert from 'node:assert/strict';
import test from 'node:test';

import {
  NEUTRAL_STATUS_COLOR,
  STATUS_META,
  firstName,
  formatLastInteraction,
  getBaselineProgressText,
  getReasonText,
  getStatusLabel,
  getTechnicalNote,
  type V1Status,
} from './v1Status';

// This module decides every word and colour a caregiver reads about how their loved one is doing, so the
// product rules in CLAUDE.md are asserted here rather than left to review: nothing clinical, nothing
// alarming for a patient who is still establishing a baseline, and device trouble always framed as a
// connection issue rather than news about the person.

const ALL_STATUSES: V1Status[] = ['establishing', 'doing_well', 'worth_checking', 'needs_attention'];

test('every status has a colour, a dot and a label', () => {
  for (const status of ALL_STATUSES) {
    const meta = STATUS_META[status];
    assert.ok(meta, `${status} is missing from STATUS_META`);
    assert.match(meta.color, /^#[0-9A-Fa-f]{6}$/);
    assert.match(meta.dot, /^#[0-9A-Fa-f]{6}$/);
    assert.ok(meta.label.length > 0);
  }
});

test('establishing is never rendered in the alarm colour', () => {
  // The rule: a patient whose baseline is still being learned must never be shown as needing attention.
  assert.notEqual(STATUS_META.establishing.dot, STATUS_META.needs_attention.dot);
  assert.notEqual(STATUS_META.establishing.color, STATUS_META.needs_attention.color);
  assert.notEqual(NEUTRAL_STATUS_COLOR, STATUS_META.needs_attention.dot);
});

test('no status label or reason reads clinically', () => {
  const clinical = /\b(normal|abnormal|cognitive|dementia|impair|diagnos|decline|risk|score|symptom)\b/i;
  for (const status of ALL_STATUSES) {
    assert.doesNotMatch(STATUS_META[status].label, clinical, `${status} label`);
    assert.doesNotMatch(getStatusLabel(status, 'Mei Ling Tan'), clinical, `${status} personalised label`);
  }
});

test('establishing is described as learning a routine, using the first name only', () => {
  assert.equal(getStatusLabel('establishing', 'Mei Ling Tan'), "Learning Mei's routine");
  assert.equal(getStatusLabel('establishing', ''), 'Learning their routine');
  assert.equal(getStatusLabel('establishing', '   '), 'Learning their routine');
});

test('firstName takes the first token and tolerates junk', () => {
  assert.equal(firstName('Mei Ling Tan'), 'Mei');
  assert.equal(firstName('  Ah   Kow '), 'Ah');
  assert.equal(firstName(''), '');
  assert.equal(firstName(null), '');
  assert.equal(firstName(undefined), '');
});

test('every reason code the server can send has warm English', () => {
  // Kept in step with the reason codes in reflexion-server (baseline §4). A code with no mapping falls
  // through to a reassuring default, which is safe — but a MISSING mapping means the caregiver is told
  // "everything looks steady" while the server flagged something, so the list is asserted explicitly.
  const codes = [
    'LEARNING_PERSONAL_ROUTINE', 'DAILY_PATTERN_ON_TRACK', 'CHECKIN_COMPLETED_TODAY',
    'CHECKIN_MISSED_TODAY', 'CHECKIN_MISSED_REPEATEDLY', 'CHECKIN_MISSED_3_DAYS',
    'CHECKIN_OUTSIDE_USUAL_WINDOW', 'WEEKLY_ENGAGEMENT_DOWN', 'SPOKE_LESS_THAN_USUAL',
    'FEWER_RESPONSES', 'SLOWER_TO_RESPOND', 'DEVICE_UNREACHABLE', 'AWAY_PERIOD_ACTIVE',
    'CAREGIVER_FLAG_WORTH_CHECKING', 'CAREGIVER_FLAG_NEEDS_ATTENTION',
  ];
  const fallback = getReasonText('SOMETHING_NEW_FROM_THE_SERVER');
  for (const code of codes) {
    const text = getReasonText(code, 'Mei Ling Tan');
    assert.notEqual(text, fallback, `${code} has no mapping and silently reads as "everything is fine"`);
    assert.doesNotMatch(text, /\b(cognitive|dementia|diagnos|decline|impair|score)\b/i, code);
    assert.doesNotMatch(text, /Ling|Tan/, `${code} should use the first name only`);
  }
});

test('an unknown reason code degrades to something calm rather than blank', () => {
  const text = getReasonText('TOTALLY_UNKNOWN');
  assert.ok(text.length > 0);
  assert.equal(getReasonText(null), text);
  assert.equal(getReasonText(undefined), text);
});

test('a device problem is attributed to the mirror, not to the person', () => {
  const unreachable = getTechnicalNote('unreachable');
  assert.ok(unreachable);
  assert.match(unreachable, /mirror/i);
  assert.match(unreachable, /not a change in how they are doing/i);

  assert.ok(getTechnicalNote('possible_issue'));
  // A healthy device says nothing at all — silence is the reassuring state.
  assert.equal(getTechnicalNote('ok'), null);
  assert.equal(getTechnicalNote('unknown'), null);
});

test('baseline progress reads as a count, never as a deficit', () => {
  assert.equal(
    getBaselineProgressText({ completedSessions: 3, requiredSessions: 7, windowDays: 14 }),
    '3 of 7 sessions recorded',
  );
  // Defensive: the read model may not carry progress yet.
  assert.equal(
    getBaselineProgressText(undefined as unknown as Parameters<typeof getBaselineProgressText>[0]),
    '0 of 7 sessions recorded',
  );
});

test('last interaction is phrased relative to today', () => {
  const now = new Date();
  const at = (dayOffset: number, hour = 8, minute = 15) => {
    const d = new Date(now.getFullYear(), now.getMonth(), now.getDate() - dayOffset, hour, minute);
    return d.toISOString();
  };

  assert.match(formatLastInteraction(at(0)), /^Today, /);
  assert.match(formatLastInteraction(at(1)), /^Yesterday, /);
  assert.equal(formatLastInteraction(at(3)), '3 days ago');
});

test('a missing or unparseable timestamp never renders as "Invalid Date"', () => {
  assert.equal(formatLastInteraction(null), 'No check-in yet');
  assert.equal(formatLastInteraction(undefined), 'No check-in yet');
  assert.equal(formatLastInteraction(''), 'No check-in yet');
  assert.equal(formatLastInteraction('not-a-date'), 'No check-in yet');
});

test('a timestamp from later today still reads as today, not as a negative day count', () => {
  const later = new Date();
  later.setHours(later.getHours() + 3);
  assert.match(formatLastInteraction(later.toISOString()), /^Today, /);
});
