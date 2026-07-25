import assert from 'node:assert/strict'
import test from 'node:test'
import { MongoMemoryReplSet } from 'mongodb-memory-server'
import request from 'supertest'

import { closeMongo, getDb } from '../../lib/mongo.js'
import { collections } from '../platform/collections.js'

// v1 could not create a user over HTTP at all before this — `users` was written only by the CLI bootstrap,
// the legacy sign-in bridge and the migration script — which is why caregiver accounts had to be created
// through the legacy API. These cover the new registration route and the profile read/write the settings
// screen needs, both preconditions for the app leaving legacy behind.

test('a caregiver can register, sign in and manage their own profile entirely over v1', async (t) => {
  const replicaSet = await MongoMemoryReplSet.create({ replSet: { count: 1, storageEngine: 'wiredTiger' } })
  const originalEnvironment = { ...process.env }
  process.env.NODE_ENV = 'test'
  process.env.MONGODB_URI = replicaSet.getUri()
  process.env.JWT_SECRET = 'registration-jwt-secret-at-least-32-characters'
  process.env.PAIRING_PEPPER = 'registration-pairing-pepper-at-least-32-characters'
  process.env.CREDENTIAL_ENCRYPTION_KEY = 'registration-encryption-key-at-least-32-characters'
  process.env.AUTH_RATE_LIMIT_PER_MINUTE = '1000'
  process.env.API_RATE_LIMIT_PER_MINUTE = '5000'

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

  await t.test('registration creates the tenant, the user and a usable session', async () => {
    const created = await app.post('/api/v1/auth/registrations')
      .send({ name: 'Wei Ling', email: '  NewCaregiver@Example.com ', password, phoneNumber: '+6590001234', relationshipToElderly: 'parent' })
      .expect(201)

    const { actor, accessToken: token, refreshToken } = created.body.data
    assert.ok(token && refreshToken, 'an account you cannot immediately use is not useful')
    assert.match(actor.userId, /^usr_/)
    assert.match(actor.tenantId, /^ten_/)
    assert.equal(actor.email, email, 'email is normalised')
    // Matches what the legacy bridge produces, so migrated and new accounts behave identically.
    assert.deepEqual(actor.roles, ['caregiver', 'tenant_admin'])
    accessToken = token
    userId = actor.userId

    const stored = await db.collection<any>(collections.users).findOne({ _id: userId })
    assert.equal(stored?.emailNormalized, email, 'without emailNormalized the account could never sign in again')
    assert.equal(stored?.status, 'active')
    assert.ok(stored?.passwordHash)
    const tenant = await db.collection<any>(collections.tenants).findOne({ _id: actor.tenantId })
    assert.equal(tenant?.status, 'active')
  })

  await t.test('the new account can sign in through the normal v1 route', async () => {
    const session = await app.post('/api/v1/auth/sessions').send({ email, password }).expect(201)
    assert.equal(session.body.data.actor.userId, userId)
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
})
