import assert from 'node:assert/strict';
import test from 'node:test';

import { parsePatientAge } from './helpers';

// parsePatientAge exists because `Number('') === 0` and 0 is an integer, so the previous
// `Number.isInteger(Number(value))` guard accepted a cleared age field and sent `age: 0` to the server.

test('a cleared or blank age is rejected rather than saved as zero', () => {
  assert.equal(parsePatientAge(''), null);
  assert.equal(parsePatientAge('   '), null);
  assert.equal(parsePatientAge('0'), null);
});

test('a plausible whole age is accepted', () => {
  assert.equal(parsePatientAge('78'), 78);
  assert.equal(parsePatientAge(' 78 '), 78);
  assert.equal(parsePatientAge('1'), 1);
  assert.equal(parsePatientAge('130'), 130);
});

test('anything that is not a whole age in range is rejected', () => {
  for (const value of ['seventy', '78.5', '-5', '131', '1e3', 'NaN', '12abc', '٧٨']) {
    assert.equal(parsePatientAge(value), null, `${value} should not parse as an age`);
  }
});

test('the client bound matches what the server will accept', () => {
  // reflexion-server/src/routes/nurse-patient-config/settings.ts requires an integer 1-130. Keeping the two
  // in step is what makes the client-side message trustworthy instead of a guess ahead of a 400.
  assert.equal(parsePatientAge('131'), null);
  assert.equal(parsePatientAge('130'), 130);
});
