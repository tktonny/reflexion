export type BootstrapCredentialClaims = {
  deviceId: string
  expiresAt: number
}

export function validateBootstrapCredential(token: string, nowSeconds = Math.floor(Date.now() / 1000)): BootstrapCredentialClaims {
  const normalized = token.trim()
  const parts = normalized.split('.')
  if (parts.length !== 3 || parts.some((part) => !part)) throw new Error('bootstrap_token_malformed')

  let claims: Record<string, unknown>
  try {
    const payload = parts[1].replace(/-/g, '+').replace(/_/g, '/')
    const padded = payload.padEnd(Math.ceil(payload.length / 4) * 4, '=')
    claims = JSON.parse(atob(padded)) as Record<string, unknown>
  } catch {
    throw new Error('bootstrap_token_malformed')
  }

  if (claims.iss !== 'reflexion' || claims.aud !== 'reflexion-api' || claims.kind !== 'bootstrap') {
    throw new Error('bootstrap_token_wrong_type')
  }
  if (typeof claims.did !== 'string' || !claims.did.trim()) throw new Error('bootstrap_token_device_missing')
  if (typeof claims.exp !== 'number' || !Number.isFinite(claims.exp) || claims.exp <= nowSeconds) {
    throw new Error('bootstrap_token_expired')
  }
  if (!Array.isArray(claims.scopes) || !claims.scopes.includes('device:pair')) {
    throw new Error('bootstrap_token_scope_missing')
  }

  return { deviceId: claims.did, expiresAt: claims.exp }
}
