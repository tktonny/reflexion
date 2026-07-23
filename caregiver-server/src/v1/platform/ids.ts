import { randomBytes, randomUUID } from 'node:crypto'

export type IdPrefix =
  | 'ten' | 'usr' | 'pat' | 'rel' | 'con' | 'enr'
  | 'dev' | 'pair' | 'asg' | 'cred' | 'cfg'
  | 'ses' | 'evt' | 'art' | 'op' | 'run'
  | 'plan' | 'rem' | 'task' | 'away' | 'flag'
  | 'base' | 'score' | 'win' | 'case' | 'disp'
  | 'auth' | 'idem' | 'audit' | 'notif' | 'day'

export function newId(prefix: IdPrefix) {
  return `${prefix}_${randomUUID().replaceAll('-', '')}`
}

export function randomSecret(bytes = 32) {
  return randomBytes(bytes).toString('base64url')
}

export function randomPairingCode() {
  const value = randomBytes(4).readUInt32BE() % 1_000_000
  return String(value).padStart(6, '0')
}
