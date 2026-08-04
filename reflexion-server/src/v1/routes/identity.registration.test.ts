import assert from 'node:assert/strict'
import test from 'node:test'
import { MongoMemoryReplSet } from 'mongodb-memory-server'
import request from 'supertest'

import { closeMongo, getDb } from '../../lib/mongo.js'
import { collections } from '../platform/collections.js'
import { openSecret } from '../platform/crypto.js'

// v1 could not create a user over HTTP at all before this — `users` was written only by the CLI bootstrap,
// the legacy sign-in bridge and the migration script — which is why caregiver accounts had to be created
// through the legacy API. These cover the new registration route and the profile read/write the settings
// screen needs, both preconditions for the app leaving legacy behind.

test('a caregiver can register, verify, sign in and manage their own profile entirely over v1', async (t) => {
  const replicaSet = await MongoMemoryReplSet.create({ replSet: { count: 1, storageEngine: 'wiredTiger' } })
  const originalEnvironment = { ...process.env }
  process.env.NODE_ENV = 'test'
  process.env.MONGODB_URI = replicaSet.getUri()
  process.env.JWT_SECRET = 'registration-jwt-secret-at-least-32-characters'
  process.env.PAIRING_PEPPER = 'registration-pairing-pepper-at-least-32-characters'
  process.env.CREDENTIAL_ENCRYPTION_KEY = 'registration-encryption-key-at-least-32-characters'
  process.env.AUTH_RATE_LIMIT_PER_MINUTE = '1000'
  process.env.API_RATE_LIMIT_PER_MINUTE = '5000'
  process.env.AUTH_EMAIL_VERIFICATION_REQUIRED = 'true'
  process.env.EMAIL_PROVIDER = 'postmark'
  process.env.POSTMARK_SERVER_TOKEN = 'test-postmark-token'
  process.env.EMAIL_FROM = 'Reflexion <test@example.com>'

  const { createApp } = await import('../../app.js')
  const app = request(createApp())
  const db = await getDb()

  t.after(async () => {
    await new Promise((resolve) => setTimeout(resolve, 50))
    await closeMongo()
    await replicaSet.stop()
    process.env = originalEnvironment
  })

  const email = 'newcaregiver@example.com'
  const password = 'a-long-enough-passphrase'
  let accessToken = ''
  let userId = ''

  let verificationCode = ''

  await t.test('registration creates a pending account and verification event', async () => {
    const created = await app.post('/api/v1/auth/registrations')
      .send({ name: 'Wei Ling', email: '  NewCaregiver@Example.com ', password, phoneNumber: '+6590001234', relationshipToElderly: 'parent' })
      .expect(202)

    assert.deepEqual(created.body.data, { state: 'verification_pending', email })

    const stored = await db.collection<any>(collections.users).findOne({ emailNormalized: email })
    assert.equal(stored?.emailNormalized, email, 'without emailNormalized the account could never sign in again')
    assert.equal(stored?.status, 'pending_verification')
    assert.equal(stored?.emailVerifiedAt, null)
    assert.ok(stored?.passwordHash)
    const tenant = await db.collection<any>(collections.tenants).findOne({ _id: stored?.tenantId })
    assert.equal(tenant?.status, 'active')
    const progress = await db.collection<any>(collections.setupProgress).findOne({ userId: stored?._id })
    assert.equal(progress?.categories['research-participation'], 'not-started')
    const event = await db.collection<any>(collections.outboxEvents).findOne({ eventType: 'account_verification.requested', aggregateId: stored?._id })
    assert.ok(event?.payload?.sealedCode)
    verificationCode = openSecret(String(event.payload.sealedCode))

    await app.get('/api/v1/me').expect(401)
  })

  await t.test('verification activates the account and is single-use', async () => {
    const verified = await app.post('/api/v1/auth/account-verifications').send({ email, code: verificationCode }).expect(201)
    const { actor, accessToken: token, refreshToken } = verified.body.data
    assert.ok(token && refreshToken)
    assert.match(actor.userId, /^usr_/)
    assert.match(actor.tenantId, /^ten_/)
    assert.equal(actor.email, email, 'email is normalised')
    assert.deepEqual(actor.roles, ['caregiver'])
    accessToken = token
    userId = actor.userId
    const stored = await db.collection<any>(collections.users).findOne({ _id: userId })
    assert.equal(stored?.status, 'active')
    assert.ok(stored?.emailVerifiedAt)
    await app.post('/api/v1/auth/account-verifications').send({ email, code: verificationCode }).expect(400)
  })

  await t.test('the new account can sign in through the normal v1 route', async () => {
    const session = await app.post('/api/v1/auth/sessions').send({ email, password }).expect(201)
    assert.equal(session.body.data.actor.userId, userId)
    const phoneSession = await app.post('/api/v1/auth/sessions').send({ identifier: '+6590001234', password }).expect(201)
    assert.equal(phoneSession.body.data.actor.userId, userId)
  })

  await t.test('registration is rejected for a duplicate email, a short password or a bad address', async () => {
    await app.post('/api/v1/auth/registrations')
      .send({ name: 'Someone Else', email, password: 'another-long-passphrase' }).expect(409)
    await app.post('/api/v1/auth/registrations')
      .send({ name: 'Too Short', email: 'short@example.com', password: 'sh0rt' }).expect(400)
    await app.post('/api/v1/auth/registrations')
      .send({ name: 'Bad Email', email: 'not-an-email', password: 'a-long-enough-passphrase' }).expect(400)
    await app.post('/api/v1/auth/registrations')
      .send({ name: 'No Password', email: 'nopass@example.com' }).expect(400)
  })

  await t.test('GET /me returns the fields the settings screen needs', async () => {
    const me = await app.get('/api/v1/me').set({ Authorization: `Bearer ${accessToken}` }).expect(200)
    const profile = me.body.data
    assert.equal(profile.phoneNumber, '+6590001234', 'phoneNumber was in users but no route returned it')
    assert.equal(profile.relationshipToElderly, 'parent')
    assert.equal(profile.notificationPreferences.pushNotificationsEnabled, true)
    assert.equal(profile.notificationPreferences.alertSensitivity, 'only_important_changes')
    assert.equal(profile.notificationPreferences.preferredDailySummaryTime, '19:00')
  })

  await t.test('PATCH /me is partial and validated', async () => {
    const authorization = { Authorization: `Bearer ${accessToken}` }

    // Sending only the phone must not blank the name.
    const phoneOnly = await app.patch('/api/v1/me').set(authorization).send({ phoneNumber: '+6599998888' }).expect(200)
    assert.equal(phoneOnly.body.data.phoneNumber, '+6599998888')
    assert.equal(phoneOnly.body.data.name, 'Wei Ling', 'a partial update must not blank an unsent field')

    // One preference at a time must not reset the others.
    const onePreference = await app.patch('/api/v1/me').set(authorization)
      .send({ notificationPreferences: { alertSensitivity: 'only_urgent_alerts' } }).expect(200)
    assert.equal(onePreference.body.data.notificationPreferences.alertSensitivity, 'only_urgent_alerts')
    assert.equal(onePreference.body.data.notificationPreferences.pushNotificationsEnabled, true)
    assert.equal(onePreference.body.data.notificationPreferences.preferredDailySummaryTime, '19:00')

    await app.patch('/api/v1/me').set(authorization).send({ name: '   ' }).expect(400)
    await app.patch('/api/v1/me').set(authorization)
      .send({ notificationPreferences: { alertSensitivity: 'whatever' } }).expect(400)
    await app.patch('/api/v1/me').set(authorization)
      .send({ notificationPreferences: { pushNotificationsEnabled: 'yes' } }).expect(400)
    await app.patch('/api/v1/me').set(authorization).send({}).expect(400)
    // Email is not editable here — it is the login identity.
    const emailIgnored = await app.patch('/api/v1/me').set(authorization).send({ email: 'hijack@example.com', name: 'Wei L' }).expect(200)
    assert.equal(emailIgnored.body.data.email, email)
  })

  await t.test('the profile routes require a session', async () => {
    await app.get('/api/v1/me').expect(401)
    await app.patch('/api/v1/me').send({ name: 'Nobody' }).expect(401)
  })

  await t.test('setup progress is persisted, versioned and idempotent', async () => {
    const authorization = { Authorization: `Bearer ${accessToken}` }
    const initial = await app.get('/api/v1/setup-progress').set(authorization).expect(200)
    assert.equal(initial.body.data.total, 8)
    assert.equal(initial.body.data.categories['research-participation'], 'not-started')
    assert.equal(initial.body.data.version, 1)

    const key = 'setup-progress-test-key'
    const changed = await app.patch('/api/v1/setup-progress').set({ ...authorization, 'If-Match': '1', 'Idempotency-Key': key })
      .send({ category: 'language-accessibility', status: 'complete' }).expect(200)
    assert.equal(changed.body.data.categories['language-accessibility'], 'complete')
    assert.equal(changed.body.data.version, 2)
    const replay = await app.patch('/api/v1/setup-progress').set({ ...authorization, 'If-Match': '1', 'Idempotency-Key': key })
      .send({ category: 'language-accessibility', status: 'complete' }).expect(200)
    assert.deepEqual(replay.body.data, changed.body.data)
    await app.patch('/api/v1/setup-progress').set({ ...authorization, 'If-Match': '1', 'Idempotency-Key': 'setup-progress-stale-key' })
      .send({ category: 'routines', status: 'in-progress' }).expect(409)
    const persisted = await app.get('/api/v1/setup-progress').set(authorization).expect(200)
    assert.equal(persisted.body.data.version, 2)
    assert.equal(persisted.body.data.categories['language-accessibility'], 'complete')
  })

  await t.test('password reset code verifies and rotates the password', async () => {
    const resetRequest = await app.post('/api/v1/auth/password-reset-requests').send({ email }).expect(202)
    assert.equal(resetRequest.body.data.state, 'accepted')
    const event = await db.collection<any>(collections.outboxEvents).findOne({ eventType: 'password_reset.requested', aggregateId: userId }, { sort: { occurredAt: -1 } })
    assert.ok(event?.payload?.sealedCode)
    const code = openSecret(String(event.payload.sealedCode))
    const verified = await app.post('/api/v1/auth/password-reset-verifications').send({ email, code }).expect(200)
    assert.ok(verified.body.data.resetToken)
    await app.post('/api/v1/auth/password-resets').send({ token: verified.body.data.resetToken, newPassword: 'a-new-long-passphrase' }).expect(200)
    await app.get('/api/v1/me').set({ Authorization: `Bearer ${accessToken}` }).expect(401)
    await app.post('/api/v1/auth/sessions').send({ email, password: 'a-new-long-passphrase' }).expect(201)
  })

  await t.test('the server policy can disable verification without rewriting pending accounts', async () => {
    const pendingEmail = 'pilot-pending@example.com'
    process.env.AUTH_EMAIL_VERIFICATION_REQUIRED = 'true'
    await app.post('/api/v1/auth/registrations').send({ name: 'Pending Pilot', email: pendingEmail, password }).expect(202)
    const pending = await db.collection<any>(collections.users).findOne({ emailNormalized: pendingEmail })
    const setup = await db.collection<any>(collections.setupProgress).findOne({ userId: pending?._id })
    assert.equal(pending?.status, 'pending_verification')
    assert.ok(setup)

    process.env.AUTH_EMAIL_VERIFICATION_REQUIRED = 'false'
    const policy = await app.get('/api/v1/auth/policy').expect(200)
    assert.equal(policy.body.data.emailVerificationRequired, false)
    const directEmail = 'pilot-direct@example.com'
    const direct = await app.post('/api/v1/auth/registrations').send({ name: 'Direct Pilot', email: directEmail, password }).expect(201)
    assert.equal(direct.body.data.emailVerified, false)
    assert.ok(direct.body.data.accessToken)
    assert.equal(await db.collection<any>(collections.outboxEvents).countDocuments({ eventType: 'account_verification.requested', 'payload.email': directEmail }), 0)
    const signedIn = await app.post('/api/v1/auth/sessions').send({ email: pendingEmail, password }).expect(201)
    assert.equal(signedIn.body.data.actor.email, pendingEmail)
    const stillPending = await db.collection<any>(collections.users).findOne({ _id: pending?._id })
    assert.equal(stillPending?.status, 'pending_verification')
    assert.equal(stillPending?.emailVerifiedAt, null)
    assert.ok(await db.collection<any>(collections.setupProgress).findOne({ userId: pending?._id }))

    process.env.AUTH_EMAIL_VERIFICATION_REQUIRED = 'true'
    await app.post('/api/v1/auth/sessions').send({ email: pendingEmail, password }).expect(403)
  })
})
