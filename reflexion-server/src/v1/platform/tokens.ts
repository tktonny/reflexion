import { createHmac, timingSafeEqual } from 'node:crypto'
import { ApiError } from './errors.js'
import { requireServerSecret } from './crypto.js'

export type TokenKind = 'human' | 'device' | 'bootstrap'

export type AccessClaims = {
  iss: 'reflexion'
  aud: 'reflexion-api'
  sub: string
  kind: TokenKind
  tid?: string
  uid?: string
  did?: string
  pid?: string
  sid?: string
  cid?: string
  roles?: string[]
  scopes?: string[]
  serialHash?: string
  iat: number
  exp: number
}

type IssueClaims = Omit<AccessClaims, 'iss' | 'aud' | 'iat' | 'exp'>

function encode(value: unknown) {
  return Buffer.from(JSON.stringify(value)).toString('base64url')
}

function signingSecret(explicit?: string) {
  return explicit || requireServerSecret('JWT_SECRET')
}

export function issueAccessToken(claims: IssueClaims, ttlSeconds: number, secret?: string) {
  const now = Math.floor(Date.now() / 1000)
  const header = encode({ alg: 'HS256', typ: 'JWT' })
  const payload = encode({
    ...claims,
    iss: 'reflexion',
    aud: 'reflexion-api',
    iat: now,
    exp: now + ttlSeconds,
  } satisfies AccessClaims)
  const signature = createHmac('sha256', signingSecret(secret)).update(`${header}.${payload}`).digest('base64url')
  return `${header}.${payload}.${signature}`
}

export function verifyAccessToken(token: string, expectedKinds?: TokenKind[], secret?: string): AccessClaims {
  const [header, payload, signature] = token.split('.')
  if (!header || !payload || !signature) throw invalidToken()
  const expected = createHmac('sha256', signingSecret(secret)).update(`${header}.${payload}`).digest()
  const actual = Buffer.from(signature, 'base64url')
  if (expected.length !== actual.length || !timingSafeEqual(expected, actual)) throw invalidToken()

  let claims: AccessClaims
  try {
    claims = JSON.parse(Buffer.from(payload, 'base64url').toString('utf8')) as AccessClaims
  } catch {
    throw invalidToken()
  }
  const now = Math.floor(Date.now() / 1000)
  if (claims.iss !== 'reflexion' || claims.aud !== 'reflexion-api' || !claims.sub || claims.exp <= now) {
    throw invalidToken()
  }
  if (expectedKinds && !expectedKinds.includes(claims.kind)) throw invalidToken()
  return claims
}

function invalidToken() {
  return new ApiError(401, 'INVALID_TOKEN', 'The access token is invalid or expired.')
}
