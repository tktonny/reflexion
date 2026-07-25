import assert from 'node:assert/strict';
import test from 'node:test';

import { LegacyApiError } from './apiClient';
import { MIN_PASSWORD_LENGTH, passwordResetMessage, passwordResetRequestMessage, signInMessage } from './authMessages';
import { V1ApiError } from './v1Errors';

// Regression guard for a real leak. Sign-in built its visible error straight from err.message, and
// apiClient composes those from the wire, so the app's front door could display — and, once the box became
// an accessibility live region, read aloud — text like:
//   "Expected JSON from /api/auth/sign-in, received 502: <html><head><title>502 Bad Gateway..."
// Every message below must be caregiver-facing prose, never a server string.

const originalWarn = console.warn;
test.before(() => { console.warn = () => {}; });
test.after(() => { console.warn = originalWarn; });

const LEAKY = [
  new LegacyApiError('Expected JSON from /api/auth/sign-in, received 502: <html> <head><title>502 Bad Gateway', 502),
  new LegacyApiError('Request failed with 404', 404),
  new LegacyApiError('<html><body>nginx</body></html>', 500),
  new TypeError('Network request failed'),
  'a bare string thrown from somewhere',
  null,
  undefined,
];

const ALL = [signInMessage, passwordResetRequestMessage, passwordResetMessage];

test('no server text, markup, status code or path ever reaches the caregiver', () => {
  for (const build of ALL) {
    for (const error of LEAKY) {
      const message = build(error);
      assert.doesNotMatch(message, /</, `${build.name} leaked markup for ${String(error)}`);
      assert.doesNotMatch(message, /\/api\//, `${build.name} leaked a path`);
      assert.doesNotMatch(message, /\b(50[0-9]|40[0-9]|429)\b/, `${build.name} leaked a status code`);
      assert.doesNotMatch(message, /Expected JSON|Request failed|nginx|Gateway/i, `${build.name} leaked wire text`);
      assert.ok(message.length > 20, `${build.name} produced something too terse to act on`);
      assert.match(message, /[.!]$/, `${build.name} should read as a sentence`);
    }
  }
});

test('bad credentials are distinguished from an unreachable server', () => {
  const rejected = signInMessage(new LegacyApiError('Invalid email or password.', 401));
  const unreachable = signInMessage(new TypeError('Network request failed'));

  assert.match(rejected, /email and password/i);
  assert.notEqual(rejected, unreachable);
  assert.match(unreachable, /connection/i);
});

test('a missing field and rate limiting each get their own advice', () => {
  assert.match(signInMessage(new LegacyApiError('Email and password are required.', 400)), /enter both/i);
  assert.match(signInMessage(new LegacyApiError('slow down', 429)), /wait a moment/i);
});

test('an expired reset link is explained rather than blamed on the connection', () => {
  const expired = passwordResetMessage(new LegacyApiError('Reset token is invalid.', 400));
  assert.match(expired, /expired|no longer valid/i);
  assert.match(expired, /request a new one/i);
});

test('nothing reads clinically or blames the person', () => {
  const clinical = /\b(cognitive|dementia|diagnos|decline|impair|invalid user|your fault)\b/i;
  for (const build of ALL) {
    for (const error of LEAKY) {
      assert.doesNotMatch(build(error), clinical);
    }
  }
});

test('the raw error is still logged for whoever has to debug it', () => {
  const logged: unknown[] = [];
  console.warn = (...args: unknown[]) => { logged.push(args); };
  signInMessage(new LegacyApiError('Expected JSON from /api/auth/sign-in, received 502', 502));
  console.warn = () => {};

  assert.equal(logged.length, 1);
  assert.match(JSON.stringify(logged[0]), /502/, 'the detail must survive somewhere');
});

test('a rejected v1 password reads as bad credentials, not as a connection problem', () => {
  // v1 is the primary login now. Classifying only LegacyApiError meant every v1 401 fell through to
  // "we could not reach Reflexion" — sending the caregiver to check a connection that was fine.
  const rejected = signInMessage(new V1ApiError('Email or password is incorrect.', 401, 'INVALID_CREDENTIALS'));
  assert.match(rejected, /email and password/i);
  assert.doesNotMatch(rejected, /connection/i);

  const rateLimited = signInMessage(new V1ApiError('slow down', 429, 'RATE_LIMITED'));
  assert.match(rateLimited, /wait a moment/i);

  // A v1 5xx or a network failure still reads as a connection problem, which is what it is.
  assert.match(signInMessage(new V1ApiError('boom', 502, 'BAD_GATEWAY')), /connection/i);
  assert.match(signInMessage(new TypeError('Network request failed')), /connection/i);
});

test('a v1 error never leaks its own message either', () => {
  const leaky = new V1ApiError('Expected JSON from /api/v1/auth/sessions (received 502).', 502);
  for (const build of [signInMessage, passwordResetRequestMessage, passwordResetMessage]) {
    const message = build(leaky);
    assert.doesNotMatch(message, /\/api\//);
    assert.doesNotMatch(message, /Expected JSON|502/);
  }
});

test('the password floor matches what the server enforces', () => {
  // v1 identity.ts rejects anything shorter than 12. The form used to accept 8, so a 9-character password
  // passed validation and was then refused by the server.
  assert.equal(MIN_PASSWORD_LENGTH, 12);
});
