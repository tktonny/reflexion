import assert from 'node:assert/strict'
import test from 'node:test'
import { hashSecret, hmac, openSecret, requireServerSecret, sealSecret, sha256, verifySecret } from './crypto.js'
import { issueAccessToken, verifyAccessToken } from './tokens.js'

const secret = 'test-secret-that-is-definitely-longer-than-thirty-two-characters'

test('secret hashing, HMAC and sealed storage do not expose plaintext', () => {
  assert.equal(sha256('a').length, 64)
  assert.equal(hmac('pairing-code', secret), hmac('pairing-code', secret))
  const hash = hashSecret('refresh')
  assert.equal(verifySecret('refresh', hash), true)
  assert.equal(verifySecret('wrong', hash), false)
  assert.equal(verifySecret('refresh', 'broken'), false)
  const sealed = sealSecret('one-time-ticket', secret)
  assert.equal(sealed.includes('one-time-ticket'), false)
  assert.equal(openSecret(sealed, secret), 'one-time-ticket')
  assert.throws(() => openSecret('broken', secret))
})

test('server secret validation rejects missing and short secrets', () => {
  const previous = process.env.TEST_SERVER_SECRET
  delete process.env.TEST_SERVER_SECRET
  assert.throws(() => requireServerSecret('TEST_SERVER_SECRET'))
  process.env.TEST_SERVER_SECRET = 'short'
  assert.throws(() => requireServerSecret('TEST_SERVER_SECRET'))
  process.env.TEST_SERVER_SECRET = secret
  assert.equal(requireServerSecret('TEST_SERVER_SECRET'), secret)
  if (previous === undefined) delete process.env.TEST_SERVER_SECRET
  else process.env.TEST_SERVER_SECRET = previous
})

test('JWT tokens validate issuer, kind, signature and expiry', () => {
  const token = issueAccessToken({ sub: 'dev_1', kind: 'device', did: 'dev_1', tid: 'ten_1', pid: 'pat_1', cid: 'cred_1' }, 60, secret)
  const claims = verifyAccessToken(token, ['device'], secret)
  assert.equal(claims.did, 'dev_1')
  assert.throws(() => verifyAccessToken(token, ['human'], secret))
  assert.throws(() => verifyAccessToken(`${token}x`, ['device'], secret))
  assert.throws(() => verifyAccessToken('broken', ['device'], secret))
  const expired = issueAccessToken({ sub: 'dev_1', kind: 'device' }, -1, secret)
  assert.throws(() => verifyAccessToken(expired, ['device'], secret))
})
