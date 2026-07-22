import { pbkdf2Sync, randomBytes, timingSafeEqual } from 'node:crypto'

export function hashPassword(password: string) {
  const salt = randomBytes(16).toString('hex')
  const hash = pbkdf2Sync(password, salt, 120000, 32, 'sha256').toString('hex')
  return `pbkdf2_sha256$120000$${salt}$${hash}`
}

export function verifyPassword(password: string, storedHash: string) {
  const [scheme, iterationText, salt, expectedHash] = storedHash.split('$')
  const iterations = Number(iterationText)
  if (scheme !== 'pbkdf2_sha256' || !Number.isInteger(iterations) || !salt || !expectedHash) {
    return false
  }

  const actualHash = pbkdf2Sync(password, salt, iterations, 32, 'sha256')
  const expected = Buffer.from(expectedHash, 'hex')
  return actualHash.length === expected.length && timingSafeEqual(actualHash, expected)
}
