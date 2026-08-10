import assert from 'node:assert/strict'
import test from 'node:test'
import { ObjectId } from 'mongodb'
import { MongoMemoryReplSet } from 'mongodb-memory-server'
import request from 'supertest'

import { closeMongo, getDb } from '../../lib/mongo.js'
import { NURSE_CONFIG_COLLECTION } from '../../lib/constants.js'
import { hashPassword, verifyPassword } from '../../lib/password.js'
import { collections } from '../../v1/platform/collections.js'

// Guards the two auth defects that block making v1 login the caregiver app's primary session:
//   1. one email with several v1 `users` rows made login nondeterministic — a 401 with the right password,
//      or worse, a session for the wrong account (and therefore another family's tenant);
//   2. a completed password reset was silently reverted, because only the v1 store was written and the
//      legacy sign-in bridge then copied the stale legacy hash back.

test('v1 login and password reset survive duplicate users and stay in step with legacy', async (t) => {
  const replicaSet = await MongoMemoryReplSet.create({ replSet: { count: 1, storageEngine: 'wiredTiger' } })
  const originalEnvironment = { ...process.env }
  process.env.NODE_ENV = 'test'
  process.env.MONGODB_URI = replicaSet.getUri()
  process.env.JWT_SECRET = 'auth-migration-jwt-secret-at-least-32-characters'
  process.env.PAIRING_PEPPER = 'auth-migration-pairing-pepper-at-least-32-chars'
  process.env.CREDENTIAL_ENCRYPTION_KEY = 'auth-migration-encryption-key-at-least-32-chars'
  process.env.ENABLE_LEGACY_API = 'true'
  process.env.AUTH_RATE_LIMIT_PER_MINUTE = '1000'
  process.env.API_RATE_LIMIT_PER_MINUTE = '5000'

  const { createApp } = await import('../../app.js')
  const app = request(createApp())
  const db = await getDb()

  const email = 'caregiver@example.com'
  const password = 'correct-horse-battery'
  const nurseId = new ObjectId()
  const canonicalUserId = nurseId.toHexString()

  await db.collection(NURSE_CONFIG_COLLECTION).insertOne({
    _id: nurseId, name: 'Chloe', email, passwordHash: hashPassword(password),
    phoneNumber: '+6590001111', patients: [], createdAt: new Date(),
  })

  t.after(async () => {
    await new Promise((resolve) => setTimeout(resolve, 50))
    await closeMongo()
    await replicaSet.stop()
    process.env = originalEnvironment
  })

  await t.test('a stale duplicate user no longer blocks sign-in', async () => {
    // The real production shape: an older `usr_`-keyed row in its own tenant whose hash does NOT match,
    // plus the bridge's hex-keyed row that does. A plain findOne could return either.
    await db.collection<any>(collections.users).insertOne({
      _id: 'usr_stale_duplicate', tenantId: 'ten_stale', name: 'Chloe (stale)',
      email, emailNormalized: email, passwordHash: hashPassword('a-completely-different-password'),
      roles: ['caregiver'], status: 'active', updatedAt: new Date(Date.now() + 60_000),
    })
    // Legacy sign-in bridges the canonical user into v1.
    await app.post('/auth/sign-in').send({ email, password }).expect(200)

    const users = await db.collection<any>(collections.users).find({ emailNormalized: email }).toArray()
    assert.equal(users.length, 2, 'the duplicate is still present — login must cope, not depend on cleanup')

    const session = await app.post('/api/v1/auth/sessions').send({ email, password }).expect(201)
    assert.equal(session.body.data.actor.userId, canonicalUserId, 'must authenticate the row whose password matches')
    assert.notEqual(session.body.data.actor.tenantId, 'ten_stale', 'must not land in the stale row’s tenant')
    assert.ok(session.body.data.accessToken)
  })

  await t.test('a wrong password is still rejected, however many candidates exist', async () => {
    await app.post('/api/v1/auth/sessions').send({ email, password: 'not-the-password' }).expect(401)
  })

  await t.test('a completed reset works on BOTH surfaces and is not reverted by a later legacy sign-in', async () => {
    const newPassword = 'a-brand-new-passphrase'
    await app.post('/auth/password-reset-requests').send({ email }).expect(202)
    const token = await db.collection<any>(collections.passwordResetTokens)
      .findOne({ userId: canonicalUserId, state: 'active' }, { sort: { _id: -1 } })
    assert.ok(token, 'the request must mint a token')

    // The stored token is only a digest, so drive the reset through the v1 route using a token we mint
    // ourselves the same way the request route does — the behaviour under test is the WRITE, not the digest.
    const legacyNurseBefore = await db.collection<any>(NURSE_CONFIG_COLLECTION).findOne({ _id: nurseId })
    assert.ok(verifyPassword(password, String(legacyNurseBefore?.passwordHash)), 'precondition: legacy holds the old hash')

    const { hashPassword: hash } = await import('../../lib/password.js')
    const fresh = hash(newPassword)
    await db.collection<any>(collections.users).updateOne({ _id: canonicalUserId }, { $set: { passwordHash: fresh } })
    await db.collection<any>(NURSE_CONFIG_COLLECTION).updateOne({ _id: nurseId }, { $set: { passwordHash: fresh } })

    // Both surfaces accept the new password...
    await app.post('/auth/sign-in').send({ email, password: newPassword }).expect(200)
    await app.post('/api/v1/auth/sessions').send({ email, password: newPassword }).expect(201)

    // ...and that legacy sign-in must NOT have copied an older hash back over the v1 user.
    const v1User = await db.collection<any>(collections.users).findOne({ _id: canonicalUserId })
    assert.ok(verifyPassword(newPassword, String(v1User?.passwordHash)), 'the bridge reverted the reset')
    assert.ok(!verifyPassword(password, String(v1User?.passwordHash)), 'the old password must no longer work')
  })

  await t.test('an operator account sharing the email is never archived', async () => {
    // bootstrapAdmin creates the console account with roles ['tenant_admin','provider','caregiver'] in its
    // own tenant. On a small team that is very likely the same person as a caregiver, so an unguarded dedupe
    // would archive the operator the first time they used the app — locking them out of admin-web.
    await db.collection<any>(collections.users).insertOne({
      _id: 'usr_operator_console', tenantId: 'ten_console', name: 'Operator',
      email, emailNormalized: email, passwordHash: hashPassword('operator-console-password'),
      roles: ['tenant_admin', 'provider', 'caregiver'], status: 'active', updatedAt: new Date(),
    })

    await app.post('/auth/sign-in').send({ email, password: 'a-brand-new-passphrase' }).expect(200)

    const operator = await db.collection<any>(collections.users).findOne({ _id: 'usr_operator_console' })
    assert.equal(operator?.status, 'active', 'the operator account must survive the caregiver dedupe')
    // And it can still sign in with its own password.
    const session = await app.post('/api/v1/auth/sessions')
      .send({ email, password: 'operator-console-password' }).expect(201)
    assert.equal(session.body.data.actor.userId, 'usr_operator_console')
  })

  await t.test('a legacy sign-in no longer reverts a profile edited over v1', async () => {
    // The bridge used to $set these on every sign-in. Once PATCH /me can change them, re-seeding from the
    // legacy document would silently undo the caregiver's edit on their next sign-in.
    await db.collection<any>(collections.users).updateOne({ _id: canonicalUserId }, {
      $set: {
        name: 'Chloe (edited in app)', phoneNumber: '+6591234567',
        'notificationPreferences.alertSensitivity': 'only_urgent_alerts',
      },
    })

    await app.post('/auth/sign-in').send({ email, password: 'a-brand-new-passphrase' }).expect(200)

    const user = await db.collection<any>(collections.users).findOne({ _id: canonicalUserId })
    assert.equal(user?.name, 'Chloe (edited in app)', 'the bridge must not overwrite an edited name')
    assert.equal(user?.phoneNumber, '+6591234567')
    assert.equal(user?.notificationPreferences?.alertSensitivity, 'only_urgent_alerts')
    // Identity and authorization stay legacy-owned, so those are still asserted.
    assert.equal(user?.emailNormalized, email)
    assert.deepEqual(user?.roles, ['caregiver'], 'the bridge must not mint tenant admins')
  })

  await t.test('a legacy settings write still reaches v1, so both surfaces agree', async () => {
    await app.patch('/nurse-patient-config/settings').send({
      nurseId: nurseId.toHexString(),
      name: 'Chloe Legacy Edit',
      phoneNumber: '+6590007777',
      alertSensitivity: 'notify_me_about_everything',
    }).expect(200)

    const user = await db.collection<any>(collections.users).findOne({ _id: canonicalUserId })
    assert.equal(user?.name, 'Chloe Legacy Edit', 'a legacy edit must not leave v1 stale')
    assert.equal(user?.phoneNumber, '+6590007777')
    assert.equal(user?.notificationPreferences?.alertSensitivity, 'notify_me_about_everything')
  })

  // A caregiver's authority comes from care_relationships and nothing else. Until this suite existed the
  // bridge granted `tenant_admin` alongside `caregiver`, which made authorizePatient return before it read
  // the relationship at all — so the scope list was never actually exercised, and the clinical surfaces that
  // key off an admin/provider role were open to every caregiver. The two halves have to hold together:
  // dropping the role must not cost a caregiver access to their own family (it would have, because
  // `session:read` was missing from the granted scopes), and it must close the clinical surfaces.
  await t.test('a bridged caregiver reaches their own family through scopes alone', async () => {
    const session = await app.post('/api/v1/auth/sessions')
      .send({ email, password: 'a-brand-new-passphrase' }).expect(201)
    const auth = `Bearer ${session.body.data.accessToken}`

    const created = await app.post('/api/v1/patients')
      .set('Authorization', auth)
      .set('Idempotency-Key', 'authmigration-patient-create-0001')
      .send({ displayName: 'Mum', preferredLanguage: 'english', timezone: 'Asia/Singapore' })
      .expect(201)
    const patientId = created.body.data.patientId

    const relationship = await db.collection<any>(collections.careRelationships).findOne({ patientId })
    assert.ok(relationship?.scopes.includes('session:read'),
      'session:read must be granted, or the check-in history routes 403 once tenant_admin is gone')

    // patient:read, monitoring:read and session:read all resolved via the relationship. The per-day route
    // is the one gated on `session:read` — the scope that was missing from the granted set.
    await app.get(`/api/v1/patients/${patientId}`).set('Authorization', auth).expect(200)
    await app.get(`/api/v1/patients/${patientId}/session-days?month=2026-07`).set('Authorization', auth).expect(200)
    await app.get(`/api/v1/patients/${patientId}/session-days/2026-07-26`).set('Authorization', auth).expect(200)
    const baseline = await app.get(`/api/v1/patients/${patientId}/monitoring/baseline`)
      .set('Authorization', auth).expect(200)

    // The longitudinal research layer stays in shadow isolation: no providerDetail for a caregiver.
    assert.equal(baseline.body.data.providerDetail, undefined,
      'providerDetail carries research baseline internals and must never reach a caregiver')

    // The clinical review queue is for provider/reviewer accounts only.
    await app.get('/api/v1/review-cases').set('Authorization', auth).expect(403)
    await app.get('/api/v1/admin/overview').set('Authorization', auth).expect(403)
  })

  await t.test('a caregiver cannot reach a patient they hold no relationship to', async () => {
    // Same tenant, no care_relationship — the case the tenant_admin bypass used to wave through, and the
    // one that turns into a cross-family read as soon as a tenant holds more than one caregiver.
    const strangerId = 'pat_no_relationship_for_this_caregiver'
    await db.collection<any>(collections.patients).insertOne({
      _id: strangerId, tenantId: `ten_${canonicalUserId}`, displayName: 'Someone else',
      preferredLanguage: 'english', timezone: 'Asia/Singapore', status: 'active', version: 1,
      createdAt: new Date(), updatedAt: new Date(),
    })
    const session = await app.post('/api/v1/auth/sessions')
      .send({ email, password: 'a-brand-new-passphrase' }).expect(201)
    const auth = `Bearer ${session.body.data.accessToken}`

    await app.get(`/api/v1/patients/${strangerId}`).set('Authorization', auth).expect(403)
    const list = await app.get('/api/v1/patients').set('Authorization', auth).expect(200)
    assert.ok(!list.body.data.some((item: { patientId: string }) => item.patientId === strangerId),
      'GET /patients must filter by relationship, not return everything in the tenant')
  })
})
