import 'dotenv/config'
import { hashPassword } from '../lib/password.js'
import { closeMongo, getDb } from '../lib/mongo.js'
import { collections } from '../v1/platform/collections.js'
import { newId } from '../v1/platform/ids.js'
import { ensureV1Indexes } from '../v1/platform/indexes.js'

function argument(name: string) {
  const prefix = `--${name}=`
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length)
}

const email = (argument('email') || process.env.BOOTSTRAP_ADMIN_EMAIL || '').trim().toLowerCase()
const password = argument('password') || process.env.BOOTSTRAP_ADMIN_PASSWORD || ''
const name = (argument('name') || process.env.BOOTSTRAP_ADMIN_NAME || 'Reflexion Admin').trim()
const tenantName = (argument('tenant') || process.env.BOOTSTRAP_TENANT_NAME || 'Reflexion').trim()

if (!email || !password || password.length < 12) {
  throw new Error('Provide --email and a --password of at least 12 characters (or BOOTSTRAP_ADMIN_EMAIL/PASSWORD).')
}

try {
  const db = await getDb()
  await ensureV1Indexes(db)
  let tenant = await db.collection<any>(collections.tenants).findOne({ name: tenantName, status: 'active' })
  if (!tenant) {
    tenant = { _id: newId('ten'), name: tenantName, status: 'active', region: process.env.DATA_REGION || 'unspecified', policyVersion: 'v1', createdAt: new Date() }
    await db.collection<any>(collections.tenants).insertOne(tenant)
  }
  const userId = newId('usr')
  const result = await db.collection<any>(collections.users).findOneAndUpdate({
    tenantId: tenant._id,
    emailNormalized: email,
  }, { $set: {
    name, email, emailNormalized: email, passwordHash: hashPassword(password), status: 'active',
    authSubject: `local:${email}`, roles: ['tenant_admin', 'provider', 'caregiver'],
    scopes: ['patient:read', 'patient:write', 'device:assign', 'care_plan:read', 'care_plan:write', 'monitoring:read', 'review:read', 'review:write'],
    updatedAt: new Date(),
  }, $setOnInsert: { _id: userId, tenantId: tenant._id, createdAt: new Date() } }, { upsert: true, returnDocument: 'after' })
  console.log(JSON.stringify({ tenantId: tenant._id, userId: result?._id, email }, null, 2))
} finally {
  await closeMongo()
}
