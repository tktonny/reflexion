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

const serial = (argument('serial') || process.env.PROVISION_DEVICE_SERIAL || '').trim()
const hardwareRevision = (argument('hardware') || 'unknown').trim()
const softwareVersion = (argument('software') || 'uninstalled').trim()
if (!serial) throw new Error('Provide --serial=<unique hardware serial>.')

try {
  const db = await getDb()
  const serialHash = sha256(serial)
  const candidateId = newId('dev')
  const device = await db.collection<any>(collections.devices).findOneAndUpdate({ serialHash }, { $set: {
    hardwareRevision, softwareVersion, status: 'provisioned', updatedAt: new Date(),
  }, $setOnInsert: { _id: candidateId, serialHash, createdAt: new Date() } }, { upsert: true, returnDocument: 'after' })
  if (!device) throw new Error('Unable to provision device.')
  const bootstrapToken = issueAccessToken({
    sub: String(device._id), kind: 'bootstrap', did: String(device._id), serialHash,
    roles: ['device_bootstrap'], scopes: ['device:pair'],
  }, 30 * 24 * 60 * 60)
  console.log(JSON.stringify({ deviceId: device._id, bootstrapToken, expiresInDays: 30 }, null, 2))
} finally {
  await closeMongo()
}
