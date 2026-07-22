import assert from 'node:assert/strict'

const base = argument('base') || process.env.API_BASE?.trim()
if (!base) throw new Error('Provide --base=https://api.example.com or set API_BASE.')

const origin = new URL(base)
if (origin.protocol !== 'https:' && !['localhost', '127.0.0.1'].includes(origin.hostname)) {
  throw new Error('The deployed API must use HTTPS.')
}
const apiBase = origin.toString().replace(/\/$/, '')

const health = await fetch(`${apiBase}/health`, { signal: AbortSignal.timeout(10_000) })
assert.equal(health.status, 200, 'GET /health must return 200.')
assert.deepEqual(await health.json(), { ok: true })
assert.equal(health.headers.get('x-powered-by'), null, 'Express fingerprinting must be disabled.')
assert.ok(health.headers.get('x-request-id'), 'Health response must include X-Request-Id.')
assert.equal(health.headers.get('x-content-type-options'), 'nosniff')

const unauthenticated = await fetch(`${apiBase}/api/v1/me`, { signal: AbortSignal.timeout(10_000) })
assert.equal(unauthenticated.status, 401, 'GET /api/v1/me must reach v1 auth, not return the old 404.')
const unauthorizedBody = await unauthenticated.json() as any
assert.equal(unauthorizedBody?.error?.code, 'UNAUTHORIZED')
assert.ok(unauthorizedBody?.meta?.requestId)

const missing = await fetch(`${apiBase}/api/v1/deployment-smoke-missing`, { signal: AbortSignal.timeout(10_000) })
assert.equal(missing.status, 404)
const missingBody = await missing.json() as any
assert.equal(missingBody?.error?.code, 'ROUTE_NOT_FOUND')
assert.ok(missingBody?.meta?.requestId)

console.log(JSON.stringify({
  ok: true,
  apiBase,
  checks: ['health', 'security_headers', 'v1_auth_boundary', 'v1_error_envelope'],
}, null, 2))

function argument(name: string) {
  const prefix = `--${name}=`
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length)
}
