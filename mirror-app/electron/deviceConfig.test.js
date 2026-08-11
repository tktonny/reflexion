const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { test } = require('node:test')

const { claimedDeviceId, readDeviceConfig, resolveApiBase, resolveBootstrapToken } = require('./deviceConfig')
const { DEFAULT_API_BASE } = require('./apiProxy')

function scratch(contents) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'reflexion-config-'))
  if (contents !== undefined) fs.writeFileSync(path.join(dir, 'device-config.json'), contents)
  return dir
}

// A shape-valid bootstrap token: header.payload.signature, payload carrying a did. Not signed — the shape
// check is all the main process does, because only the backend can verify a signature.
function fakeToken(did = 'dev_abc123') {
  const b64 = (value) => Buffer.from(JSON.stringify(value)).toString('base64url')
  return `${b64({ alg: 'HS256', typ: 'JWT' })}.${b64({ did, kind: 'bootstrap' })}.c2lnbmF0dXJl`
}

test('a missing config file is "no configuration", not a crash', () => {
  assert.deepEqual(readDeviceConfig(scratch()), {})
})

test('a corrupt config file degrades instead of taking the appliance down', () => {
  // A mirror in a home must still boot with a stray comma in its config; throwing here would be a black screen.
  assert.deepEqual(readDeviceConfig(scratch('{ "apiBase": "https://x.test",, }')), {})
})

test('a config file that is valid JSON but not an object is ignored', () => {
  assert.deepEqual(readDeviceConfig(scratch('"just a string"')), {})
})

test('apiBase precedence: env beats file beats the production default', () => {
  const dir = scratch(JSON.stringify({ apiBase: 'https://file.test' }))
  assert.equal(resolveApiBase(dir, { REFLEXION_API_BASE: 'https://env.test' }), 'https://env.test')
  assert.equal(resolveApiBase(dir, {}), 'https://file.test')
  assert.equal(resolveApiBase(scratch(), {}), DEFAULT_API_BASE)
})

test('an unprovisioned unit reports no token rather than failing', () => {
  assert.equal(resolveBootstrapToken(scratch(), {}), null)
})

test('the bootstrap token is read from the config file', () => {
  const token = fakeToken()
  assert.equal(resolveBootstrapToken(scratch(JSON.stringify({ bootstrapToken: token })), {}), token)
})

test('the launch env overrides the config file', () => {
  const fromEnv = fakeToken('dev_env')
  const dir = scratch(JSON.stringify({ bootstrapToken: fakeToken('dev_file') }))
  assert.equal(resolveBootstrapToken(dir, { REFLEXION_BOOTSTRAP_TOKEN: fromEnv }), fromEnv)
})

test('a token that is not JWT-shaped is rejected, and does not mask a good one in the file', () => {
  // The ordinary operator mistakes: a pasted filename, a truncated copy, an empty value. Catching them at
  // startup turns an opaque 401-at-pairing into a log line naming the source.
  const good = fakeToken('dev_file')
  const dir = scratch(JSON.stringify({ bootstrapToken: good }))
  for (const bad of ['bootstrap-token.txt', 'eyJhbGciOiJIUzI1NiJ9', '', '   ']) {
    assert.equal(resolveBootstrapToken(dir, { REFLEXION_BOOTSTRAP_TOKEN: bad }), good)
  }
})

test('whitespace around a copied token is tolerated', () => {
  const token = fakeToken()
  assert.equal(resolveBootstrapToken(scratch(), { REFLEXION_BOOTSTRAP_TOKEN: `\n  ${token}\t ` }), token)
})

test('claimedDeviceId reads did for logging and never throws on junk', () => {
  assert.equal(claimedDeviceId(fakeToken('dev_xyz')), 'dev_xyz')
  assert.equal(claimedDeviceId('a.b.c'), null)
  assert.equal(claimedDeviceId('not-a-token'), null)
})
