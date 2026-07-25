#!/usr/bin/env node
/*
 * READ-ONLY pre-flight for the caregiver v1 migration. Writes nothing, ever.
 *
 * Answers the three questions you cannot answer from the code:
 *   1. Has migrate:legacy-v1 already run? (are there hex-keyed v1 users matching NursePatientConfig ids)
 *   2. Which emails have duplicate v1 users, and would the dedupe collapse the RIGHT one?
 *   3. What is still missing that would keep a daily check-in from running? (consent, patients, relationships)
 *
 * Plain CommonJS on purpose: `npm run build` OOMs on the server, so this must run without one.
 *
 *   cd /path/to/reflexion-server && node --env-file=.env scripts/diagnose-v1-migration-state.cjs
 *
 * No password hashes, tokens or transcript text are printed. Emails ARE printed, since identifying which
 * account needs attention is the point; run it where that is acceptable.
 */
const { MongoClient, ObjectId } = require('mongodb')

const MONGODB_URI = process.env.MONGODB_URI
const MONGODB_DB = process.env.MONGODB_DB || 'ref'

if (!MONGODB_URI) {
  console.error('MONGODB_URI is not set. Run with: node --env-file=.env scripts/diagnose-v1-migration-state.cjs')
  process.exit(1)
}

const HEX_24 = /^[0-9a-f]{24}$/i
const CHECKIN_CONSENT_PURPOSE = 'home_cognitive_monitoring'
const heading = (text) => console.log(`\n${text}\n${'-'.repeat(text.length)}`)

async function main() {
  const client = new MongoClient(MONGODB_URI)
  await client.connect()
  const db = client.db(MONGODB_DB)
  console.log(`database: ${MONGODB_DB}   (READ-ONLY — this script writes nothing)`)

  const nurses = await db.collection('NursePatientConfig')
    .find({}, { projection: { name: 1, email: 1, passwordHash: 1, patients: 1 } }).toArray()
  const users = await db.collection('users')
    .find({}, { projection: { tenantId: 1, email: 1, emailNormalized: 1, roles: 1, status: 1, passwordHash: 1, notificationPreferences: 1, phoneNumber: 1, createdAt: 1 } }).toArray()

  heading('1. Has migrate:legacy-v1 run?')
  const userIds = new Set(users.map((user) => String(user._id)))
  const legacyPatientIds = nurses.flatMap((nurse) => (nurse.patients || [])
    .map((patient) => patient?._id && String(patient._id)).filter(Boolean))
  const bridgedNurses = nurses.filter((nurse) => userIds.has(String(nurse._id)))
  const v1Patients = await db.collection('patients')
    .find({}, { projection: { tenantId: 1, displayName: 1, profile: 1 } }).toArray()
  const v1PatientIds = new Set(v1Patients.map((patient) => String(patient._id)))
  const migratedPatients = legacyPatientIds.filter((id) => v1PatientIds.has(id))

  console.log(`legacy caregivers (NursePatientConfig): ${nurses.length}`)
  console.log(`  of those with a hex-keyed v1 user:    ${bridgedNurses.length}`)
  console.log(`legacy patients:                        ${legacyPatientIds.length}`)
  console.log(`  of those present in v1 patients:      ${migratedPatients.length}`)
  const withProfile = v1Patients.filter((patient) => patient.profile && Object.keys(patient.profile).length).length
  console.log(`v1 patients carrying a \`profile\`:       ${withProfile}  (0 = the profile-carrying migration has NOT run)`)
  console.log(
    bridgedNurses.length === 0 ? '=> VERDICT: never run, and no caregiver has signed in since the bridge shipped.'
      : bridgedNurses.length < nurses.length ? `=> VERDICT: partial — ${nurses.length - bridgedNurses.length} caregiver(s) have no v1 user. Those accounts CANNOT sign in to v1 yet.`
        : withProfile === 0 ? '=> VERDICT: an older migration/bridge ran, but NOT the profile-carrying one. Re-run it.'
          : '=> VERDICT: looks migrated. Re-running is idempotent and will fill any gaps.',
  )

  heading('2. Duplicate v1 users per email (what the dedupe would collapse)')
  const byEmail = new Map()
  for (const user of users) {
    const key = String(user.emailNormalized || user.email || '').trim().toLowerCase()
    if (!key) continue
    if (!byEmail.has(key)) byEmail.set(key, [])
    byEmail.get(key).push(user)
  }
  const duplicates = [...byEmail.entries()].filter(([, rows]) => rows.filter((row) => row.status === 'active').length > 1)
  const missingEmailNormalized = users.filter((user) => user.status === 'active' && !user.emailNormalized)

  if (!duplicates.length) console.log('none — every active email maps to exactly one v1 user.')
  for (const [email, rows] of duplicates) {
    console.log(`\n  ${email}`)
    const legacyNurse = nurses.find((nurse) => String(nurse.email || '').trim().toLowerCase() === email)
    for (const row of rows) {
      const id = String(row._id)
      const isCanonical = legacyNurse && id === String(legacyNurse._id)
      const privileged = (row.roles || []).some((role) => role === 'operator' || role === 'provider')
      const inSync = legacyNurse && row.passwordHash && legacyNurse.passwordHash
        ? row.passwordHash === legacyNurse.passwordHash : null
      const fate = row.status !== 'active' ? 'already archived'
        : isCanonical ? 'KEPT (canonical — its _id is the legacy nurse id)'
          : privileged ? 'KEPT (operator/provider — deliberately spared)'
            : 'WOULD BE ARCHIVED'
      console.log(`    ${id}  tenant=${row.tenantId}  roles=[${(row.roles || []).join(',')}]  status=${row.status}`)
      console.log(`      password matches legacy: ${inSync === null ? 'unknown' : inSync}   -> ${fate}`)
    }
    if (!legacyNurse) {
      console.log('    !! no NursePatientConfig with this email — nothing here is "canonical", so the dedupe')
      console.log('       will not fire and v1 login will resolve by verifying the password. Worth a look.')
    }
  }
  if (missingEmailNormalized.length) {
    console.log(`\n  !! ${missingEmailNormalized.length} active user(s) have NO emailNormalized — they can never sign in to v1:`)
    for (const user of missingEmailNormalized.slice(0, 10)) console.log(`     ${String(user._id)}  ${user.email || '(no email)'}`)
  }

  heading('3. What still blocks a daily check-in')
  const relationships = await db.collection('care_relationships')
    .find({ status: 'active' }, { projection: { patientId: 1, userId: 1, scopes: 1 } }).toArray()
  const consents = await db.collection('consents')
    .find({ purpose: CHECKIN_CONSENT_PURPOSE, status: 'granted' }, { projection: { patientId: 1, documentVersion: 1, withdrawnAt: 1 } }).toArray()
  const consentedPatients = new Set(consents.filter((consent) => !consent.withdrawnAt).map((consent) => String(consent.patientId)))
  const relatedPatients = new Set(relationships.map((relationship) => String(relationship.patientId)))

  console.log(`v1 patients:                            ${v1Patients.length}`)
  console.log(`  with an active care relationship:     ${v1Patients.filter((patient) => relatedPatients.has(String(patient._id))).length}`)
  console.log(`  with a granted check-in consent:      ${v1Patients.filter((patient) => consentedPatients.has(String(patient._id))).length}`)
  const blocked = v1Patients.filter((patient) => !consentedPatients.has(String(patient._id)))
  console.log(`  WITHOUT consent (check-ins refused):  ${blocked.length}`)
  for (const patient of blocked.slice(0, 15)) {
    console.log(`     ${String(patient._id)}  ${patient.displayName || '(no name)'}`)
  }
  const backfilled = consents.filter((consent) => consent.documentVersion === 'legacy-onboarding-backfill-v1').length
  if (backfilled) console.log(`  (${backfilled} consent(s) are already marked as backfills)`)

  heading('4. Collections the migration will touch')
  for (const name of ['users', 'tenants', 'patients', 'care_relationships', 'consents', 'care_plans', 'notification_devices']) {
    const count = await db.collection(name).countDocuments().catch(() => 'n/a')
    console.log(`  ${name.padEnd(22)} ${count}`)
  }

  console.log('\nSuggested order:')
  console.log('  1. mongodump the users, consents, patients and care_plans collections (the dedupe changes status).')
  console.log('  2. npm run db:indexes')
  console.log('  3. npm run migrate:legacy-v1   (idempotent; prints how many consents it backfilled)')
  console.log('  4. Re-run this script — section 3 should show 0 patients without consent.')

  await client.close()
}

main().catch((error) => {
  console.error('diagnosis failed:', error)
  process.exit(1)
})
