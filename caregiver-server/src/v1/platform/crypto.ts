import { createCipheriv, createDecipheriv, createHash, createHmac, randomBytes, timingSafeEqual } from 'node:crypto'

export function sha256(value: string | Buffer) {
  return createHash('sha256').update(value).digest('hex')
}

export function hmac(value: string, secret = requireServerSecret('PAIRING_PEPPER')) {
  return createHmac('sha256', secret).update(value).digest('hex')
}

export function hashSecret(secret: string) {
  const salt = randomBytes(16).toString('hex')
  const digest = createHash('sha256').update(`${salt}:${secret}`).digest('hex')
  return `sha256$${salt}$${digest}`
}

export function verifySecret(secret: string, stored: string) {
  const [scheme, salt, expected] = stored.split('$')
  if (scheme !== 'sha256' || !salt || !expected) return false
  const actual = createHash('sha256').update(`${salt}:${secret}`).digest('hex')
  const actualBytes = Buffer.from(actual)
  const expectedBytes = Buffer.from(expected)
  return actualBytes.length === expectedBytes.length && timingSafeEqual(actualBytes, expectedBytes)
}

export function requireServerSecret(name: string) {
  const value = process.env[name]?.trim()
  if (!value || value.length < 32) {
    throw new Error(`${name} must be configured with at least 32 characters.`)
  }
  return value
}

export function sealSecret(value: string, secret = requireServerSecret('CREDENTIAL_ENCRYPTION_KEY')) {
  const key = createHash('sha256').update(secret).digest()
  const iv = randomBytes(12)
  const cipher = createCipheriv('aes-256-gcm', key, iv)
  const encrypted = Buffer.concat([cipher.update(value, 'utf8'), cipher.final()])
  const tag = cipher.getAuthTag()
  return [iv, tag, encrypted].map((part) => part.toString('base64url')).join('.')
}

export function openSecret(value: string, secret = requireServerSecret('CREDENTIAL_ENCRYPTION_KEY')) {
  const [ivText, tagText, encryptedText] = value.split('.')
  if (!ivText || !tagText || !encryptedText) throw new Error('Invalid sealed secret.')
  const key = createHash('sha256').update(secret).digest()
  const decipher = createDecipheriv('aes-256-gcm', key, Buffer.from(ivText, 'base64url'))
  decipher.setAuthTag(Buffer.from(tagText, 'base64url'))
  return Buffer.concat([
    decipher.update(Buffer.from(encryptedText, 'base64url')),
    decipher.final(),
  ]).toString('utf8')
}
