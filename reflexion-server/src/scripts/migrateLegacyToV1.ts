// One-time (idempotent) migration: NursePatientConfig -> v1 tenants/users/patients/care_relationships.
// v1 _ids reuse the legacy ObjectId hex, so the legacy API keeps returning the same ids (see
// LEGACY_V1_ADAPTER.md). Safe to re-run. Does NOT migrate historical Conversations.
//   npm run build && npm run migrate:legacy-v1
import 'dotenv/config'
import { closeMongo, getDb } from '../lib/mongo.js'
import { NURSE_CONFIG_COLLECTION } from '../lib/constants.js'
import { ensureV1Patient, ensureV1TenantUser } from '../lib/legacyV1Bridge.js'

async function main() {
  const db = await getDb()
  const configs = await db.collection<any>(NURSE_CONFIG_COLLECTION).find({}).toArray()
  let nurses = 0
  let patients = 0
  for (const config of configs) {
    if (typeof config?._id?.toHexString !== 'function') continue
    const { tenantId, userId } = await ensureV1TenantUser(db, config)
    const list = Array.isArray(config.patients) ? config.patients : []
    for (const patient of list) {
      if (typeof patient?._id?.toHexString !== 'function') continue
      await ensureV1Patient(db, tenantId, userId, patient)
      patients += 1
    }
    nurses += 1
    console.log(`✓ nurse ${userId} → ${tenantId} (${list.length} patients)`)
  }
  console.log(`DONE: ${nurses} nurses, ${patients} patients → v1 (db=${process.env.MONGODB_DB || 'ref'})`)
  await closeMongo()
}

main().catch((error) => {
  console.error('migration failed:', error)
  process.exit(1)
})
