import assert from 'node:assert/strict';
import test from 'node:test';

import { normalizePhone, validateCreateAccount, validateEmail, validatePasswordPair, validateSignIn } from './authValidation';

test('new account validation gives field-level guidance for missing and malformed values', () => {
  const errors = validateCreateAccount({ name: '', email: 'not-an-email', countryCode: '+65', phoneNumber: '', password: 'short', repeatPassword: 'different' });
  assert.equal(errors.name, 'Enter your name.');
  assert.equal(errors.email, 'Enter a valid email address.');
  assert.match(errors.password || '', /at least 12/);
  assert.equal(errors.repeatPassword, 'The passwords do not match.');
});

test('a valid twelve-character password and separate phone parts pass client validation', () => {
  const errors = validateCreateAccount({ name: 'Chloe', email: 'chloe@example.com', countryCode: '+65', phoneNumber: '9000 1234', password: 'twelve-char!', repeatPassword: 'twelve-char!' });
  assert.deepEqual(errors, {});
  assert.equal(normalizePhone('+65', '9000 1234'), '+6590001234');
});

test('sign-in does not apply the new-password length policy to legacy accounts', () => {
  assert.deepEqual(validateSignIn('legacy@example.com', 'short', 'email'), {});
});

test('email and phone validation distinguish empty from malformed input', () => {
  assert.equal(validateEmail(''), 'Enter your email address.');
  assert.equal(validateEmail('missing-at.example.com'), 'Enter a valid email address.');
  assert.equal(validateEmail('valid@example.com'), undefined);
  assert.match(validateSignIn('+65', 'password', 'phone').identifier || '', /valid phone/);
});

test('password confirmation reports both the canonical policy and mismatch', () => {
  const errors = validatePasswordPair('short', 'other');
  assert.match(errors.password || '', /at least 12/);
  assert.equal(errors.repeatPassword, 'The passwords do not match.');
});
