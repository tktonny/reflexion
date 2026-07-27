import assert from 'node:assert/strict'
import test from 'node:test'
import { MongoMemoryReplSet } from 'mongodb-memory-server'
import request from 'supertest'

import { closeMongo, getDb } from '../../lib/mongo.js'
import { collections } from '../platform/collections.js'
import { CAREGIVER_RELATIONSHIP_SCOPES } from '../platform/scopes.js'
import { issueAccessToken } from '../platform/tokens.js'

// admin.ts is the whole backend for admin-web, which is in production, and it had no tests at all — the
// package's `npm test` glob was unquoted, so `sh` expanded `src/**/*.test.ts` to a single directory level and
// the suite silently ran 1 of 17 files. Quoting the glob made the documented coverage gate honest and it went
// red on this file at 5% function coverage.
//
// The two sides of this router have deliberately different gates: /admin/* needs an operator or tenant_admin,
// while /support/* is open to any authenticated human so a caregiver and an operator can hold one conversation.
// That split is the thing most worth pinning down, because caregivers no longer carry `tenant_admin`.

const TENANT_ID = 'ten_admin_suite'
const OTHER_TENANT_ID = 'ten_admin_other'
const ADMIN_ID = 'usr_admin_operator'
const CAREGIVER_ID = 'usr_admin_caregiver'
const OTHER_CAREGIVER_ID = 'usr_admin_caregiver_two'
const ADMIN_SESSION = 'auth_admin_operator'
const CAREGIVER_SESSION = 'auth_admin_caregiver'
const OTHER_CAREGIVER_SESSION = 'auth_admin_caregiver_two'

const bearer = (userId: string, sessionId: string, roles: string[], tenantId = TENANT_ID) =>
  `Bearer ${issueAccessToken({ sub: userId, kind: 'human', tid: tenantId, uid: userId, sid: sessionId, roles, scopes: [] }, 3600)}`

test('the admin and support router enforces its two gates and onboards a patient completely', async (t) => {
  const replicaSet = await MongoMemoryReplSet.create({ replSet: { count: 1, storageEngine: 'wiredTiger' } })
  const originalEnvironment = { ...process.env }
  process.env.NODE_ENV = 'test'
  process.env.MONGODB_URI = replicaSet.getUri()
  process.env.JWT_SECRET = 'admin-suite-jwt-secret-at-least-32-characters'
  process.env.PAIRING_PEPPER = 'admin-suite-pairing-pepper-at-least-32-chars'
  process.env.CREDENTIAL_ENCRYPTION_KEY = 'admin-suite-encryption-key-at-least-32-chars'
  process.env.AUTH_RATE_LIMIT_PER_MINUTE = '1000'
  process.env.API_RATE_LIMIT_PER_MINUTE = '5000'

  const { createApp } = await import('../../app.js')
  const app = request(createApp())
  const db = await getDb()
  const now = new Date()

  for (const [sessionId, userId, tenantId] of [
    [ADMIN_SESSION, ADMIN_ID, TENANT_ID],
    [CAREGIVER_SESSION, CAREGIVER_ID, TENANT_ID],
    [OTHER_CAREGIVER_SESSION, OTHER_CAREGIVER_ID, TENANT_ID],
  ]) {
    await db.collection<any>(collections.authSessions).insertOne({
      _id: sessionId, tenantId, userId, status: 'active', refreshExpiresAt: new Date(Date.now() + 86_400_000),
    })
  }
  await db.collection<any>(collections.users).insertMany([
    { _id: ADMIN_ID, tenantId: TENANT_ID, name: 'Ops', email: 'ops@example.com', roles: ['operator'], status: 'active', createdAt: now },
    { _id: CAREGIVER_ID, tenantId: TENANT_ID, name: 'Chloe', email: 'chloe@example.com', roles: ['caregiver'], status: 'active', createdAt: now },
    { _id: OTHER_CAREGIVER_ID, tenantId: TENANT_ID, name: 'Sam', email: 'sam@example.com', roles: ['caregiver'], status: 'active', createdAt: now },
    // Another tenant's user must never appear in a listing.
    { _id: 'usr_admin_outsider', tenantId: OTHER_TENANT_ID, name: 'Outsider', email: 'out@example.com', roles: ['caregiver'], status: 'active', createdAt: now },
  ])

  const admin = { Authorization: bearer(ADMIN_ID, ADMIN_SESSION, ['operator']) }
  const caregiver = { Authorization: bearer(CAREGIVER_ID, CAREGIVER_SESSION, ['caregiver']) }
  const otherCaregiver = { Authorization: bearer(OTHER_CAREGIVER_ID, OTHER_CAREGIVER_SESSION, ['caregiver']) }

  t.after(async () => {
    await closeMongo()
    await replicaSet.stop()
    process.env = originalEnvironment
  })

  await t.test('every /admin route refuses a caregiver and an anonymous caller', async () => {
    for (const path of ['/api/v1/admin/overview', '/api/v1/admin/users', '/api/v1/admin/patients', '/api/v1/admin/support/threads']) {
      await app.get(path).set(caregiver).expect(403)
      await app.get(path).expect(401)
    }
    await app.post('/api/v1/admin/patients')
      .set({ ...caregiver, 'Idempotency-Key': 'admin_forbidden_create_0001' })
      .send({ displayName: 'Nope', preferredLanguage: 'english', timezone: 'Asia/Singapore' }).expect(403)
  })

  await t.test('the overview counts only this tenant', async () => {
    const overview = await app.get('/api/v1/admin/overview').set(admin).expect(200)
    assert.equal(overview.body.data.users, 3, 'the outsider belongs to another tenant')
    assert.equal(overview.body.data.patients, 0)
    assert.equal(overview.body.data.openThreads, 0)
    assert.equal(overview.body.data.devices, 0)
  })

  await t.test('the user list is tenant-scoped and pages', async () => {
    const all = await app.get('/api/v1/admin/users').set(admin).expect(200)
    assert.equal(all.body.data.length, 3)
    assert.ok(!all.body.data.some((user: { userId: string }) => user.userId === 'usr_admin_outsider'))
    assert.equal(all.body.meta.nextCursor, null)

    const firstPage = await app.get('/api/v1/admin/users?limit=2').set(admin).expect(200)
    assert.equal(firstPage.body.data.length, 2)
    assert.ok(firstPage.body.meta.nextCursor, 'a truncated page must carry a cursor')
    const secondPage = await app.get(`/api/v1/admin/users?limit=2&cursor=${firstPage.body.meta.nextCursor}`).set(admin).expect(200)
    const seen = [...firstPage.body.data, ...secondPage.body.data].map((user: { userId: string }) => user.userId)
    assert.equal(new Set(seen).size, seen.length, 'pages must not overlap')
  })

  let patientId = ''

  await t.test('onboarding a patient links the caregiver with the full scope set', async () => {
    const created = await app.post('/api/v1/admin/patients')
      .set({ ...admin, 'Idempotency-Key': 'admin_create_patient_00000001' })
      .send({
        displayName: 'Ah Ma', preferredLanguage: 'mandarin', timezone: 'Asia/Singapore',
        ageBand: '80-89', caregiverUserId: CAREGIVER_ID,
      }).expect(201)
    patientId = created.body.data.patientId
    assert.match(patientId, /^pat_/)
    assert.equal(created.body.data.version, 1)
    assert.equal(created.body.data.ageBand, '80-89')

    // The scope list here used to be a fourth hand-written copy missing `device:assign`, which left an
    // operator-onboarded family unable to pair their mirror.
    const relationship = await db.collection<any>(collections.careRelationships).findOne({ patientId })
    assert.deepEqual([...(relationship?.scopes || [])].sort(), [...CAREGIVER_RELATIONSHIP_SCOPES].sort())
    assert.equal(relationship?.userId, CAREGIVER_ID)

    const audit = await db.collection<any>(collections.auditEvents).findOne({ action: 'admin.patient.created' })
    assert.equal(audit?.object?.id, patientId)

    // The linked caregiver can now reach the patient through the relationship alone.
    await app.get(`/api/v1/patients/${patientId}`).set(caregiver).expect(200)
    await app.get(`/api/v1/patients/${patientId}`).set(otherCaregiver).expect(403)
  })

  await t.test('a replayed onboarding returns the first patient rather than a second one', async () => {
    const replay = await app.post('/api/v1/admin/patients')
      .set({ ...admin, 'Idempotency-Key': 'admin_create_patient_00000001' })
      .send({
        displayName: 'Ah Ma', preferredLanguage: 'mandarin', timezone: 'Asia/Singapore',
        ageBand: '80-89', caregiverUserId: CAREGIVER_ID,
      }).expect(201)
    assert.equal(replay.body.data.patientId, patientId)
    assert.equal(await db.collection<any>(collections.patients).countDocuments({ tenantId: TENANT_ID }), 1)
  })

  await t.test('onboarding rejects a bad timezone and a caregiver from another tenant', async () => {
    await app.post('/api/v1/admin/patients')
      .set({ ...admin, 'Idempotency-Key': 'admin_create_bad_timezone_001' })
      .send({ displayName: 'X', preferredLanguage: 'english', timezone: 'Mars/Olympus' }).expect(400)
    const rejected = await app.post('/api/v1/admin/patients')
      .set({ ...admin, 'Idempotency-Key': 'admin_create_bad_caregiver_01' })
      .send({ displayName: 'X', preferredLanguage: 'english', timezone: 'Asia/Singapore', caregiverUserId: 'usr_admin_outsider' })
      .expect(400)
    assert.equal(rejected.body.error.code, 'CAREGIVER_NOT_FOUND')
    // The rejection must not have left a patient behind.
    assert.equal(await db.collection<any>(collections.patients).countDocuments({ tenantId: TENANT_ID }), 1)
  })

  await t.test('the patient list searches by name and stays tenant-scoped', async () => {
    const listed = await app.get('/api/v1/admin/patients').set(admin).expect(200)
    assert.equal(listed.body.data.length, 1)
    const hit = await app.get('/api/v1/admin/patients?q=ah').set(admin).expect(200)
    assert.equal(hit.body.data.length, 1, 'the search is case-insensitive')
    const miss = await app.get('/api/v1/admin/patients?q=zzzz').set(admin).expect(200)
    assert.equal(miss.body.data.length, 0)
  })

  await t.test('a patient patch is partial, validated, and bumps the version', async () => {
    const patched = await app.patch(`/api/v1/admin/patients/${patientId}`).set(admin)
      .send({ displayName: 'Ah Ma Tan', status: 'paused' }).expect(200)
    assert.equal(patched.body.data.displayName, 'Ah Ma Tan')
    assert.equal(patched.body.data.status, 'paused')
    assert.equal(patched.body.data.version, 2, 'every write bumps the version')
    assert.equal(patched.body.data.preferredLanguage, 'mandarin', 'an omitted field is left alone')

    await app.patch(`/api/v1/admin/patients/${patientId}`).set(admin).send({}).expect(400)
    await app.patch(`/api/v1/admin/patients/${patientId}`).set(admin).send({ status: 'retired' }).expect(400)
    await app.patch(`/api/v1/admin/patients/${patientId}`).set(admin).send({ timezone: 'Mars/Olympus' }).expect(400)
    await app.patch('/api/v1/admin/patients/pat_does_not_exist').set(admin).send({ displayName: 'X' }).expect(404)

    // Clearing ageBand is expressible, and the language/band round-trip keeps working.
    const cleared = await app.patch(`/api/v1/admin/patients/${patientId}`).set(admin)
      .send({ ageBand: '', preferredLanguage: 'english' }).expect(200)
    assert.equal(cleared.body.data.ageBand, null)
    assert.equal(cleared.body.data.preferredLanguage, 'english')
  })

  let threadId = ''

  await t.test('a caregiver opens a support thread and only sees their own', async () => {
    const opened = await app.post('/api/v1/support/threads')
      .set({ ...caregiver, 'Idempotency-Key': 'support_open_thread_00000001' })
      .send({ subject: 'The mirror is not waking up', body: 'It stopped responding this morning.' })
      .expect(201)
    threadId = opened.body.data.threadId
    assert.match(threadId, /^thr_/)
    assert.equal(opened.body.data.status, 'open')
    assert.equal(opened.body.data.adminUnread, true, 'a new thread is unread for the operator')
    assert.equal(opened.body.data.caregiverUnread, false)

    const mine = await app.get('/api/v1/support/threads').set(caregiver).expect(200)
    assert.equal(mine.body.data.length, 1)
    const theirs = await app.get('/api/v1/support/threads').set(otherCaregiver).expect(200)
    assert.equal(theirs.body.data.length, 0, 'threads are private to the caregiver who opened them')

    await app.post('/api/v1/support/threads')
      .set({ ...caregiver, 'Idempotency-Key': 'support_open_missing_body_01' })
      .send({ subject: 'No body' }).expect(400)
  })

  await t.test('a caregiver can reply to their own thread and to no one else\'s', async () => {
    const replied = await app.post(`/api/v1/support/threads/${threadId}/messages`).set(caregiver)
      .send({ body: 'It is still not waking up.' }).expect(201)
    assert.equal(replied.body.data.authorType, 'caregiver')
    assert.equal(replied.body.data.authorId, CAREGIVER_ID)

    await app.post(`/api/v1/support/threads/${threadId}/messages`).set(otherCaregiver)
      .send({ body: 'Let me read that.' }).expect(403)
    await app.post('/api/v1/support/threads/thr_does_not_exist/messages').set(caregiver)
      .send({ body: 'Hello?' }).expect(404)
  })

  await t.test('feedback is authored by the token, not by a nurseId in the body', async () => {
    const sent = await app.post('/api/v1/feedback').set(caregiver)
      .send({ message: 'The wake word does not always hear my mother.', category: 'mirror' }).expect(201)
    assert.match(sent.body.data.feedbackId, /^fbk_/)

    const row = await db.collection<any>(collections.feedback).findOne({ _id: sent.body.data.feedbackId })
    assert.equal(row?.userId, CAREGIVER_ID)
    assert.equal(row?.tenantId, TENANT_ID)
    // Mirrored so the legacy {nurseId, createdAt} index and any legacy reader still see v1 rows.
    assert.equal(row?.nurseId, CAREGIVER_ID)
    assert.equal(row?.category, 'mirror')

    await app.post('/api/v1/feedback').set(caregiver).send({ message: '   ' }).expect(400)
    await app.post('/api/v1/feedback').set(caregiver).send({ message: 'x'.repeat(5001) }).expect(400)
    await app.post('/api/v1/feedback').send({ message: 'anonymous' }).expect(401)

    const plainSend = await app.post('/api/v1/feedback').set(caregiver).send({ message: 'Just a note.' }).expect(201)
    const plain = await db.collection<any>(collections.feedback).findOne({ _id: plainSend.body.data.feedbackId })
    assert.equal(plain?.category, null)
  })

  await t.test('the operator sees every thread, can filter, reply, and close it', async () => {
    const all = await app.get('/api/v1/admin/support/threads').set(admin).expect(200)
    assert.equal(all.body.data.length, 1)
    assert.equal(all.body.data[0].adminUnread, true)

    const open = await app.get('/api/v1/admin/support/threads?status=open').set(admin).expect(200)
    assert.equal(open.body.data.length, 1)
    const closed = await app.get('/api/v1/admin/support/threads?status=closed').set(admin).expect(200)
    assert.equal(closed.body.data.length, 0)
    await app.get('/api/v1/admin/support/threads?status=archived').set(admin).expect(400)

    // Reading a thread returns its messages in order and clears the operator's unread marker.
    const read = await app.get(`/api/v1/admin/support/threads/${threadId}`).set(admin).expect(200)
    assert.equal(read.body.data.messages.length, 2)
    assert.deepEqual(read.body.data.messages.map((m: { authorType: string }) => m.authorType), ['caregiver', 'caregiver'])
    const afterRead = await db.collection<any>(collections.supportThreads).findOne({ _id: threadId })
    assert.equal(afterRead?.adminUnread, false)
    await app.get('/api/v1/admin/support/threads/thr_does_not_exist').set(admin).expect(404)

    const reply = await app.post(`/api/v1/admin/support/threads/${threadId}/messages`).set(admin)
      .send({ body: 'We have pushed a fix — please restart the mirror.' }).expect(201)
    assert.equal(reply.body.data.authorType, 'admin')
    const afterReply = await db.collection<any>(collections.supportThreads).findOne({ _id: threadId })
    assert.equal(afterReply?.caregiverUnread, true, 'the caregiver side gains the unread marker')
    assert.match(String(afterReply?.lastMessagePreview), /^We have pushed a fix/)

    const resolved = await app.patch(`/api/v1/admin/support/threads/${threadId}`).set(admin)
      .send({ status: 'closed' }).expect(200)
    assert.equal(resolved.body.data.status, 'closed')
    await app.patch(`/api/v1/admin/support/threads/${threadId}`).set(admin).send({ status: 'archived' }).expect(400)
    await app.patch('/api/v1/admin/support/threads/thr_does_not_exist').set(admin).send({ status: 'open' }).expect(404)

    // A closed thread reopens when either side writes again, so a reply is never lost to a stale status.
    await app.post(`/api/v1/support/threads/${threadId}/messages`).set(caregiver)
      .send({ body: 'Restarting it worked, thank you.' }).expect(201)
    const reopened = await db.collection<any>(collections.supportThreads).findOne({ _id: threadId })
    assert.equal(reopened?.status, 'open')
    assert.equal(reopened?.adminUnread, true)
  })
})
