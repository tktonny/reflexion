import 'dotenv/config'
import { closeMongo, getDb } from '../lib/mongo.js'
import { collections } from '../v1/platform/collections.js'
import { sha256 } from '../v1/platform/crypto.js'
import { newId } from '../v1/platform/ids.js'
import { issueAccessToken } from '../v1/platform/tokens.js'

function argument(name: string) {
  const prefix = `--${name}=`
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length)
}

// A bootstrap token is DEVICE-BOUND: `bootstrapClaims` re-reads the devices row by (did, serialHash) on
// every call, so a token is only usable against the database and JWT_SECRET that minted it. A token
// provisioned on a laptop therefore 401s in production — the row does not exist there and the signature
// does not verify. Production tokens must be minted by running this script ON the production server.
const serial = (argument('serial') || process.env.PROVISION_DEVICE_SERIAL || '').trim()
// The plaintext serial is not always available: a unit already in the field reports only its hash (and
// the device id it was first provisioned with). Accept those directly so the SAME identity can be
// re-issued against production instead of inventing a second one for the same physical mirror.
const serialHashArgument = (argument('serial-hash') || '').trim().toLowerCase()
const deviceIdArgument = (argument('device-id') || '').trim()
const hardwareRevision = (argument('hardware') || 'unknown').trim()
// Without this every unit reads as "Reflexion Mirror" in the caregiver app (serializeDevice's fallback),
// which is wrong the moment a fleet contains something that is not a mirror.
const displayName = (argument('display-name') || '').trim()
const softwareVersion = (argument('software') || 'uninstalled').trim()
// 30 days is right for a unit shipping straight to a home. A test fleet that is provisioned once and
// re-flashed for weeks needs longer, otherwise pairing silently starts failing mid-pilot.
const ttlDays = Number(argument('ttl-days') || 30)

if (!serial && !serialHashArgument) throw new Error('Provide --serial=<unique hardware serial> or --serial-hash=<sha256 of it>.')
if (serial && serialHashArgument) throw new Error('Provide either --serial or --serial-hash, not both.')
if (serialHashArgument && !/^[0-9a-f]{64}$/.test(serialHashArgument)) {
  throw new Error(`--serial-hash must be 64 lowercase hex characters (a sha256 digest); received ${serialHashArgument.length}.`)
}
if (!Number.isFinite(ttlDays) || ttlDays <= 0 || ttlDays > 365) throw new Error('--ttl-days must be between 1 and 365.')

try {
  const db = await getDb()
  const serialHash = serialHashArgument || sha256(serial)
  // Match on the device id when one was supplied, so a mirror keeps the identity it already displays;
  // otherwise the serial hash is the natural key and a fresh id is minted on first sight.
  const filter = deviceIdArgument ? { _id: deviceIdArgument } : { serialHash }
  const candidateId = deviceIdArgument || newId('dev')
  const device = await db.collection<any>(collections.devices).findOneAndUpdate(filter, { $set: {
    hardwareRevision, softwareVersion, status: 'provisioned', serialHash, updatedAt: new Date(),
    ...(displayName ? { displayName } : {}),
  }, $setOnInsert: { _id: candidateId, createdAt: new Date() } }, { upsert: true, returnDocument: 'after' })
  if (!device) throw new Error('Unable to provision device.')
  const bootstrapToken = issueAccessToken({
    sub: String(device._id), kind: 'bootstrap', did: String(device._id), serialHash,
    roles: ['device_bootstrap'], scopes: ['device:pair'],
  }, ttlDays * 24 * 60 * 60)
  console.log(JSON.stringify({ deviceId: device._id, serialHash, displayName: device.displayName || null, bootstrapToken, expiresInDays: ttlDays }, null, 2))
} finally {
  await closeMongo()
}
