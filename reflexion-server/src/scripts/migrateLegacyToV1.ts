// One-time (idempotent) migration: NursePatientConfig -> v1 tenants/users/patients/care_relationships,
// plus a marked backfill of the check-in consent every patient needs (see ensureCheckInConsent) — without it
// POST /sessions refuses a daily_checkin with 403 CONSENT_REQUIRED, so no check-in can run at all.
// v1 _ids reuse the legacy ObjectId hex, so the legacy API keeps returning the same ids (see
// LEGACY_V1_ADAPTER.md). Safe to re-run. Does NOT migrate historical Conversations.
//   npm run build && npm run migrate:legacy-v1
import 'dotenv/config'
import { closeMongo, getDb } from '../lib/mongo.js'
import { NURSE_CONFIG_COLLECTION } from '../lib/constants.js'
import { BACKFILL_CONSENT_DOCUMENT_VERSION, ensureCheckInConsent, ensureV1Patient, ensureV1TenantUser } from '../lib/legacyV1Bridge.js'

async function main() {
  const db = await getDb()
  const configs = await db.collection<any>(NURSE_CONFIG_COLLECTION).find({}).toArray()
  let nurses = 0
  let patients = 0
  let consentsCreated = 0
  for (const config of configs) {
    if (typeof config?._id?.toHexString !== 'function') continue
    const { tenantId, userId } = await ensureV1TenantUser(db, config)
    const list = Array.isArray(config.patients) ? config.patients : []
    for (const patient of list) {
      if (typeof patient?._id?.toHexString !== 'function') continue
      await ensureV1Patient(db, tenantId, userId, patient)
      // ensureV1Patient already backfills this, but it is asserted here too so the script's own count is
      // honest for patients that existed in v1 before this run.
      if (await ensureCheckInConsent(db, { tenantId, patientId: patient._id.toHexString(), actorId: userId }) === 'created') {
        consentsCreated += 1
      }
      patients += 1
    }
    nurses += 1
    console.log(`✓ nurse ${userId} → ${tenantId} (${list.length} patients)`)
  }
  console.log(`DONE: ${nurses} nurses, ${patients} patients → v1 (db=${process.env.MONGODB_DB || 'ref'})`)
  if (consentsCreated) {
    console.log(
      `     ${consentsCreated} check-in consent(s) BACKFILLED as '${BACKFILL_CONSENT_DOCUMENT_VERSION}'.\n` +
      "     These were inferred, not asked for — nothing has ever collected consent during onboarding.\n" +
      `     Audit or revoke them with: db.consents.find({ documentVersion: '${BACKFILL_CONSENT_DOCUMENT_VERSION}' })`,
    )
  }
  await closeMongo()
}

main().catch((error) => {
  console.error('migration failed:', error)
  process.exit(1)
})
