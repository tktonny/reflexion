import assert from 'node:assert/strict'
import test from 'node:test'
import { ObjectId } from 'mongodb'
import { MongoMemoryReplSet } from 'mongodb-memory-server'

import { closeMongo, getDb } from './mongo.js'
import { collections } from '../v1/platform/collections.js'

// The migration used to carry only displayName/preferredLanguage/timezone/ageBand, silently dropping the
// rest of the loved one's profile. Two homes, split by who consumes the data: `patients.profile` for what
// only the caregiver's app displays, and the care plan for anything that changes how Aria talks — because
// the care plan is what GET /devices/:deviceId/configuration already ships to the mirror.

test('migrating a legacy patient carries the whole profile into v1', async (t) => {
  const replicaSet = await MongoMemoryReplSet.create({ replSet: { count: 1, storageEngine: 'wiredTiger' } })
  const originalEnvironment = { ...process.env }
  process.env.MONGODB_URI = replicaSet.getUri()
  process.env.JWT_SECRET = 'profile-migration-jwt-secret-at-least-32-chars'
  process.env.PAIRING_PEPPER = 'profile-migration-pairing-pepper-at-least-32-chars'
  process.env.CREDENTIAL_ENCRYPTION_KEY = 'profile-migration-encryption-key-at-least-32-chars'

  const { ensureV1Patient, ensureV1TenantUser } = await import('./legacyV1Bridge.js')
  const db = await getDb()

  t.after(async () => {
    await closeMongo()
    await replicaSet.stop()
    process.env = originalEnvironment
  })

  const nurseId = new ObjectId()
  const patientId = new ObjectId()
  const { tenantId, userId } = await ensureV1TenantUser(db, {
    _id: nurseId, name: 'Chloe', email: 'chloe@example.com', passwordHash: 'x', phoneNumber: '+6590001111',
  } as never)

  await ensureV1Patient(db, tenantId, userId, {
    _id: patientId, name: 'Mei Ling Tan', preferredLanguage: 'mandarin', timezone: 'Asia/Singapore',
    age: 78, gender: 'female', photoUrl: 'https://example.com/mei.jpg', phoneNumber: '+6590002222',
    speechSpeed: 'Slow', usualWakeTime: '07:00', speechOrHearingConditions: 'Mild hearing loss, left ear',
    keyTopics: ['family', 'food', 'others'], keyTopicsOtherText: 'Gardening',
  } as never)

  await t.test('display-only fields land in patients.profile, with ageBand derived from the exact age', async () => {
    const patient = await db.collection<any>(collections.patients).findOne({ _id: patientId.toHexString() })
    assert.equal(patient?.displayName, 'Mei Ling Tan')
    assert.equal(patient?.ageBand, '75_84')
    assert.equal(patient?.profile?.age, 78)
    assert.equal(patient?.profile?.gender, 'female')
    assert.equal(patient?.profile?.photoUrl, 'https://example.com/mei.jpg')
    assert.equal(patient?.profile?.phoneNumber, '+6590002222')
    assert.equal(patient?.profile?.speechSpeed, 'slow', 'legacy stored it capitalised')
  })

  await t.test('conversational fields land in the care plan, where the mirror can actually see them', async () => {
    const plan = await db.collection<any>(collections.carePlans).findOne({ patientId: patientId.toHexString() })
    assert.ok(plan, 'a plan must be seeded or the wake time and topics reach nothing')
    assert.equal(plan.dailyRoutine.wakeTime, '07:00')
    assert.deepEqual(plan.communicationPreferences.topics, ['family', 'food', 'others'])
    assert.equal(plan.communicationPreferences.otherTopic, 'Gardening')
    assert.equal(plan.communicationPreferences.speechSpeed, 'slow')
    assert.equal(plan.communicationPreferences.speechOrHearingNotes, 'Mild hearing loss, left ear')
    assert.equal(plan.status, 'active')
    assert.equal(plan.source, 'legacy_profile_migration', 'provenance stays queryable')
  })

  await t.test('re-running does not duplicate, and never overwrites a plan someone has since edited', async () => {
    await db.collection<any>(collections.carePlans).updateOne(
      { patientId: patientId.toHexString() },
      { $set: { 'dailyRoutine.wakeTime': '06:30', version: 2 } },
    )
    await ensureV1Patient(db, tenantId, userId, {
      _id: patientId, name: 'Mei Ling Tan', preferredLanguage: 'mandarin', age: 78, usualWakeTime: '07:00',
    } as never)

    const plans = await db.collection<any>(collections.carePlans).find({ patientId: patientId.toHexString() }).toArray()
    assert.equal(plans.length, 1, 'a second run must not stack another plan')
    assert.equal(plans[0].dailyRoutine.wakeTime, '06:30', 'an edited plan belongs to whoever edited it')
  })

  await t.test('a patient with none of the conversational fields gets no empty plan', async () => {
    const bareId = new ObjectId()
    await ensureV1Patient(db, tenantId, userId, {
      _id: bareId, name: 'Ah Kow', preferredLanguage: 'english',
    } as never)
    const plan = await db.collection<any>(collections.carePlans).findOne({ patientId: bareId.toHexString() })
    assert.equal(plan, null, 'an empty care plan is noise, not data')
    // The consent backfill still applies, since without it no check-in can run.
    const consent = await db.collection<any>(collections.consents).findOne({ patientId: bareId.toHexString() })
    assert.equal(consent?.status, 'granted')
  })
})
