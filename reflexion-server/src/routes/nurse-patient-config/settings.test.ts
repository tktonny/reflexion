import assert from 'node:assert/strict'
import test from 'node:test'
import { ObjectId } from 'mongodb'
import { MongoMemoryReplSet } from 'mongodb-memory-server'
import request from 'supertest'

import { closeMongo, getDb } from '../../lib/mongo.js'
import { NURSE_CONFIG_COLLECTION } from '../../lib/constants.js'
import { collections } from '../../v1/platform/collections.js'
import { issueAccessToken } from '../../v1/platform/tokens.js'

// Covers the caregiver-settings write path and the two data-safety properties it must hold:
//   1. GET /nurse-patient-config/latest refuses to serve a record without an explicit nurseId (it used
//      to fall back to the newest config, handing an unauthenticated caller someone else's PII).
//   2. PATCH is a PARTIAL update — a client that only sends a name can never blank the fields it did not
//      send, which is what made the previously-blank edit form a data-loss path.
// Note: constants.DB_NAME is captured at import time, so the suite deliberately leaves MONGODB_DB unset
// and works in the default `ref` database that both the legacy and v1 helpers resolve to.

const NURSE_ID = new ObjectId()
const OTHER_NURSE_ID = new ObjectId()
const PATIENT_ID = new ObjectId()
const SECOND_PATIENT_ID = new ObjectId()
const TENANT_ID = `ten_${NURSE_ID.toHexString()}`
const USER_ID = `usr_${NURSE_ID.toHexString()}`
const AUTH_SESSION_ID = 'auth_settings_test'

const storedPatient = (id: ObjectId, name: string) => ({
  _id: id,
  name,
  phoneNumber: '+6590000001',
  age: 78,
  gender: 'female',
  preferredLanguage: 'mandarin',
  usualWakeTime: '07:00',
  speechOrHearingConditions: 'Mild hearing loss on the left',
  speechSpeed: 'Slow',
  mirrorName: 'Living room mirror',
  photoUrl: '',
  keyTopics: ['family', 'food'],
  keyTopicsOtherText: null,
})

test('caregiver settings writes persist without blanking untouched fields', async (t) => {
  const replicaSet = await MongoMemoryReplSet.create({ replSet: { count: 1, storageEngine: 'wiredTiger' } })
  const originalEnvironment = { ...process.env }
  process.env.NODE_ENV = 'test'
  process.env.MONGODB_URI = replicaSet.getUri()
  process.env.JWT_SECRET = 'settings-test-jwt-secret-at-least-32-characters'
  process.env.PAIRING_PEPPER = 'settings-test-pairing-pepper-at-least-32-characters'
  process.env.CREDENTIAL_ENCRYPTION_KEY = 'settings-test-encryption-key-at-least-32-characters'
  process.env.ENABLE_LEGACY_API = 'true'
  process.env.AUTH_RATE_LIMIT_PER_MINUTE = '1000'
  process.env.API_RATE_LIMIT_PER_MINUTE = '5000'

  // Imported after the env is in place so createApp() sees ENABLE_LEGACY_API.
  const { createApp } = await import('../../app.js')
  const app = request(createApp())
  const db = await getDb()

  await db.collection(NURSE_CONFIG_COLLECTION).insertMany([
    {
      _id: NURSE_ID,
      name: 'Chloe Tan',
      email: 'chloe@example.com',
      phoneNumber: '+6590001111',
      pushNotificationsEnabled: false,
      alertSensitivity: 'only_important_changes',
      preferredDailySummaryTime: '09:00',
      storeSessionSummaries: true,
      patients: [storedPatient(PATIENT_ID, 'Mei Ling Tan'), storedPatient(SECOND_PATIENT_ID, 'Ah Kow Lim')],
      createdAt: new Date('2026-01-01T00:00:00Z'),
    },
    {
      _id: OTHER_NURSE_ID,
      name: 'Someone Else',
      email: 'someone@example.com',
      phoneNumber: '+6599999999',
      patients: [],
      // Newest by createdAt — this is the record the old latest-fallback would have leaked.
      createdAt: new Date('2026-06-01T00:00:00Z'),
    },
  ])
  await db.collection<any>(collections.patients).insertOne({
    _id: PATIENT_ID.toHexString(), tenantId: TENANT_ID, displayName: 'Mei Ling Tan',
    preferredLanguage: 'mandarin', timezone: 'Asia/Singapore', ageBand: '75_84', status: 'active', version: 1,
  })
  await db.collection<any>(collections.authSessions).insertOne({
    _id: AUTH_SESSION_ID, tenantId: TENANT_ID, userId: USER_ID, status: 'active',
    refreshExpiresAt: new Date(Date.now() + 86_400_000),
  })

  t.after(async () => {
    // Let the fire-and-forget audit writes drain before the client goes away.
    await new Promise((resolve) => setTimeout(resolve, 50))
    await closeMongo()
    await replicaSet.stop()
    process.env = originalEnvironment
  })

  await t.test('latest refuses to serve an arbitrary caregiver record', async () => {
    const missing = await app.get('/nurse-patient-config/latest').expect(400)
    assert.match(missing.body.error, /nurse id is required/i)
    await app.get('/nurse-patient-config/latest?nurseId=not-an-object-id').expect(400)
  })

  await t.test('latest returns every field the settings edit form seeds from', async () => {
    const found = await app.get(`/nurse-patient-config/latest?nurseId=${NURSE_ID.toHexString()}`).expect(200)
    assert.equal(found.body.caregiverName, 'Chloe Tan')
    assert.equal(found.body.storeSessionSummaries, true)
    const patient = found.body.patients.find((item: any) => item.patientId === PATIENT_ID.toHexString())
    assert.equal(patient.gender, 'female')
    assert.equal(patient.usualWakeTime, '07:00')
    assert.equal(patient.speechOrHearingConditions, 'Mild hearing loss on the left')
    assert.deepEqual(patient.keyTopics, ['family', 'food'])
  })

  await t.test('nurse settings PATCH validates and persists', async () => {
    await app.patch('/nurse-patient-config/settings').send({ name: 'No Id' }).expect(400)
    await app.patch('/nurse-patient-config/settings')
      .send({ nurseId: NURSE_ID.toHexString(), alertSensitivity: 'whatever' }).expect(400)
    await app.patch('/nurse-patient-config/settings')
      .send({ nurseId: NURSE_ID.toHexString(), name: '   ' }).expect(400)
    await app.patch('/nurse-patient-config/settings')
      .send({ nurseId: new ObjectId().toHexString(), name: 'Ghost' }).expect(404)

    const saved = await app.patch('/nurse-patient-config/settings').send({
      nurseId: NURSE_ID.toHexString(),
      name: 'Chloe T',
      phoneNumber: '+6590002222',
      pushNotificationsEnabled: true,
      alertSensitivity: 'only_urgent_alerts',
      preferredDailySummaryTime: '19:00',
      storeSessionSummaries: false,
    }).expect(200)

    assert.equal(saved.body.caregiverName, 'Chloe T')
    assert.equal(saved.body.email, 'chloe@example.com', 'the response carries the fields the app re-stores')
    assert.equal(saved.body.storeSessionSummaries, false)
    assert.equal(saved.body.patients.length, 2)

    // storeSessionSummaries: false must survive the round trip through GET, not read back as true.
    const reread = await app.get(`/nurse-patient-config/latest?nurseId=${NURSE_ID.toHexString()}`).expect(200)
    assert.equal(reread.body.storeSessionSummaries, false)
    assert.equal(reread.body.alertSensitivity, 'only_urgent_alerts')
  })

  await t.test('a partial patient PATCH leaves the fields it did not send alone', async () => {
    const saved = await app.patch(`/nurse-patient-config/settings/patients/${PATIENT_ID.toHexString()}`)
      .send({ nurseId: NURSE_ID.toHexString(), name: 'Mei Ling' }).expect(200)

    assert.equal(saved.body.patient.name, 'Mei Ling')
    assert.equal(saved.body.patient.usualWakeTime, '07:00')
    assert.equal(saved.body.patient.gender, 'female')
    assert.deepEqual(saved.body.patient.keyTopics, ['family', 'food'])
    assert.equal(saved.body.patient.speechOrHearingConditions, 'Mild hearing loss on the left')

    const config = await db.collection(NURSE_CONFIG_COLLECTION).findOne({ _id: NURSE_ID })
    const patients = (config?.patients || []) as Record<string, any>[]
    const untouched = patients.find((item) => item._id?.equals?.(SECOND_PATIENT_ID))
    assert.equal(untouched?.name, 'Ah Kow Lim', 'editing one patient must not disturb a sibling')
  })

  await t.test('renaming a patient keeps the v1 read model in step', async () => {
    const v1Patient = await db.collection<any>(collections.patients).findOne({ _id: PATIENT_ID.toHexString() })
    assert.equal(v1Patient?.displayName, 'Mei Ling')
  })

  await t.test('patient PATCH rejects invalid profiles and unknown ids', async () => {
    const nurseId = NURSE_ID.toHexString()
    const patientId = PATIENT_ID.toHexString()
    await app.patch(`/nurse-patient-config/settings/patients/${patientId}`)
      .send({ nurseId, keyTopics: ['others'] }).expect(400)
    await app.patch(`/nurse-patient-config/settings/patients/${patientId}`)
      .send({ nurseId, keyTopics: [] }).expect(400)
    await app.patch(`/nurse-patient-config/settings/patients/${patientId}`)
      .send({ nurseId, age: 'seventy' }).expect(400)
    await app.patch(`/nurse-patient-config/settings/patients/${patientId}`)
      .send({ nurseId, gender: 'unspecified' }).expect(400)
    await app.patch(`/nurse-patient-config/settings/patients/${patientId}`)
      .send({ nurseId }).expect(400)
    await app.patch(`/nurse-patient-config/settings/patients/${new ObjectId().toHexString()}`)
      .send({ nurseId, name: 'Ghost' }).expect(404)
  })

  await t.test('keyTopics with others requires the free-text and then stores both', async () => {
    const saved = await app.patch(`/nurse-patient-config/settings/patients/${PATIENT_ID.toHexString()}`)
      .send({ nurseId: NURSE_ID.toHexString(), keyTopics: ['family', 'others'], keyTopicsOtherText: 'Gardening' })
      .expect(200)
    assert.deepEqual(saved.body.patient.keyTopics, ['family', 'others'])
    assert.equal(saved.body.patient.keyTopicsOtherText, 'Gardening')
  })

  await t.test('a job-written daily alert is readable through GET /notifications', async () => {
    // The regression this guards: finalizeDay used to insert notifications with no recipientUserId and
    // state:'queued', while the read model filters {tenantId, recipientUserId} and states unread|read — so
    // every daily alert was invisible to every client. Producer and consumer are asserted together here.
    const { dailyCheckNotificationCopy, materializeNotifications } = await import('../../v1/notifications/service.js')
    const copy = dailyCheckNotificationCopy('missed_7pm', 'Mei Ling')
    const created = await materializeNotifications(db, {
      tenantId: TENANT_ID,
      patientId: PATIENT_ID.toHexString(),
      recipientUserIds: [USER_ID],
      type: 'missed_7pm',
      title: copy.title,
      body: copy.body,
      dedupeKey: `${PATIENT_ID.toHexString()}:2026-07-25:missed_7pm`,
      source: { type: 'daily_check', id: 'daily' },
      extra: { localDate: '2026-07-25', statusAtSend: 'amber', reason: 'CHECKIN_MISSED_TODAY' },
    })
    assert.equal(created, 1)

    const token = issueAccessToken({
      sub: USER_ID, kind: 'human', tid: TENANT_ID, uid: USER_ID, sid: AUTH_SESSION_ID,
      roles: ['caregiver'], scopes: [],
    }, 3600)
    const authorization = { Authorization: `Bearer ${token}` }

    const feed = await app.get('/api/v1/notifications').set(authorization).expect(200)
    const alert = feed.body.data.find((item: any) => item.type === 'missed_7pm')
    assert.ok(alert, 'the daily alert must reach the caregiver feed')
    assert.equal(alert.title, 'No check-in yet today')
    assert.equal(alert.state, 'unread')
    assert.equal(alert.localDate, '2026-07-25', 'the app deep-links to that day from this field')

    const unreadOnly = await app.get('/api/v1/notifications?state=unread').set(authorization).expect(200)
    assert.ok(unreadOnly.body.data.some((item: any) => item.notificationId === alert.notificationId))

    const read = await app.post(`/api/v1/notifications/${alert.notificationId}/read`).set(authorization).expect(200)
    assert.equal(read.body.data.state, 'read')

    // Another caregiver in a different tenant must never see it.
    const outsiderSessionId = 'auth_settings_outsider'
    await db.collection<any>(collections.authSessions).insertOne({
      _id: outsiderSessionId, tenantId: 'ten_outsider', userId: 'usr_outsider', status: 'active',
      refreshExpiresAt: new Date(Date.now() + 86_400_000),
    })
    const outsiderToken = issueAccessToken({
      sub: 'usr_outsider', kind: 'human', tid: 'ten_outsider', uid: 'usr_outsider', sid: outsiderSessionId,
      roles: ['caregiver'], scopes: [],
    }, 3600)
    const outsiderFeed = await app.get('/api/v1/notifications')
      .set({ Authorization: `Bearer ${outsiderToken}` }).expect(200)
    assert.equal(outsiderFeed.body.data.length, 0)
  })

  await t.test('the alert feed is newest-first and pages without gaps or repeats', async () => {
    // Regression guard: ids are `notif_<random uuid>`, so the feed used to sort and paginate on `_id` and
    // came out in arbitrary order — "load more" returned an arbitrary slice rather than the next one.
    const { materializeNotifications } = await import('../../v1/notifications/service.js')
    const base = Date.UTC(2026, 6, 20, 9, 0, 0)
    for (let day = 0; day < 7; day++) {
      await db.collection<any>(collections.notifications).updateOne(
        { _id: `notif_feed_${day}` },
        { $setOnInsert: {
          _id: `notif_feed_${day}`, tenantId: TENANT_ID, recipientUserId: USER_ID,
          patientId: PATIENT_ID.toHexString(), type: 'missed_7pm', state: 'unread',
          title: `Day ${day}`, body: 'x', dedupeKey: `feed:${day}`,
          createdAt: new Date(base + day * 86_400_000), updatedAt: new Date(base),
        } },
        { upsert: true },
      )
    }
    void materializeNotifications

    const token = issueAccessToken({
      sub: USER_ID, kind: 'human', tid: TENANT_ID, uid: USER_ID, sid: AUTH_SESSION_ID,
      roles: ['caregiver'], scopes: [],
    }, 3600)
    const authorization = { Authorization: `Bearer ${token}` }

    const collected: string[] = []
    let cursor: string | null = null
    for (let page = 0; page < 6; page++) {
      const query: string = cursor ? `limit=3&cursor=${encodeURIComponent(cursor)}` : 'limit=3'
      const response = await app.get(`/api/v1/notifications?${query}`).set(authorization).expect(200)
      collected.push(...response.body.data.map((item: any) => item.notificationId as string))
      cursor = (response.body.meta.nextCursor as string | null) ?? null
      if (!cursor) break
    }

    const feedOnly = collected.filter((id) => id.startsWith('notif_feed_'))
    assert.equal(new Set(feedOnly).size, feedOnly.length, 'pagination must not repeat a notification')
    assert.equal(feedOnly.length, 7, 'pagination must not skip a notification')
    assert.deepEqual(
      feedOnly,
      ['notif_feed_6', 'notif_feed_5', 'notif_feed_4', 'notif_feed_3', 'notif_feed_2', 'notif_feed_1', 'notif_feed_0'],
      'newest first, across page boundaries',
    )

    // A malformed cursor restarts from the top rather than erroring or returning nothing.
    const garbage = await app.get('/api/v1/notifications?limit=3&cursor=not-a-cursor').set(authorization).expect(200)
    assert.equal(garbage.body.data.length, 3)
  })

  await t.test('batch statuses report a per-patient outcome instead of dropping rows', async () => {
    // Regression guard: the batch route used to swallow every per-id failure and return only the successes,
    // so the dashboard counted an unreadable patient into no status bucket and showed a confident zero.
    const token = issueAccessToken({
      sub: USER_ID, kind: 'human', tid: TENANT_ID, uid: USER_ID, sid: AUTH_SESSION_ID,
      roles: ['caregiver'], scopes: [],
    }, 3600)
    const authorization = { Authorization: `Bearer ${token}` }

    const unknownId = new ObjectId().toHexString()
    const response = await app
      .get(`/api/v1/patient-statuses?ids=${PATIENT_ID.toHexString()},${unknownId}`)
      .set(authorization).expect(200)

    const byId = new Map(response.body.data.map((row: any) => [row.patientId, row]))
    assert.equal(response.body.data.length, 2, 'every requested id must come back with an outcome')
    assert.equal((byId.get(unknownId) as any).outcome, 'unavailable')
    assert.equal((byId.get(unknownId) as any).status, null)
    // The known patient has no care relationship in this fixture, so it is also unavailable — the point is
    // that it is REPORTED rather than silently missing.
    assert.ok(['ok', 'unavailable'].includes((byId.get(PATIENT_ID.toHexString()) as any).outcome))

    await app.get('/api/v1/patient-statuses').set(authorization).expect(400)
    const tooMany = Array.from({ length: 26 }, () => new ObjectId().toHexString()).join(',')
    await app.get(`/api/v1/patient-statuses?ids=${tooMany}`).set(authorization).expect(400)
  })

  await t.test('push device registration upserts on the Expo token', async () => {
    const token = issueAccessToken({
      sub: USER_ID, kind: 'human', tid: TENANT_ID, uid: USER_ID, sid: AUTH_SESSION_ID,
      roles: ['caregiver'], scopes: [],
    }, 3600)
    const authorization = { Authorization: `Bearer ${token}` }

    await app.post('/api/v1/notification-devices').send({ expoPushToken: 'ExponentPushToken[abc]' }).expect(401)
    await app.post('/api/v1/notification-devices').set(authorization)
      .send({ expoPushToken: 'not-a-token' }).expect(400)

    const first = await app.post('/api/v1/notification-devices').set(authorization)
      .send({ expoPushToken: 'ExponentPushToken[abc123]', platform: 'android', appVersion: '1.0.0' }).expect(200)
    assert.match(first.body.data.deviceId, /^ndev_/)
    assert.equal(first.body.data.platform, 'android')
    assert.equal(first.body.data.state, 'active')

    const second = await app.post('/api/v1/notification-devices').set(authorization)
      .send({ expoPushToken: 'ExponentPushToken[abc123]', platform: 'android' }).expect(200)
    assert.equal(second.body.data.deviceId, first.body.data.deviceId, 're-registering must not create a second row')

    const rows = await db.collection<any>(collections.notificationDevices)
      .find({ tenantId: TENANT_ID }).toArray()
    assert.equal(rows.length, 1)
    assert.equal(rows[0].userId, USER_ID)
  })
})
