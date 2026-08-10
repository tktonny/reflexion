import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import test from 'node:test'
import { MongoMemoryReplSet } from 'mongodb-memory-server'
import { dirname, join } from 'node:path'
import request from 'supertest'

import { closeMongo, getDb } from '../../lib/mongo.js'
import { collections } from '../platform/collections.js'
import { issueAccessToken } from '../platform/tokens.js'

const TENANT_ID = 'ten_family_messages_test'
const USER_ID = 'usr_family_messages_caregiver'
const PATIENT_ID = 'pat_family_messages_subject'
const DEVICE_ID = 'dev_family_messages_mirror'
const CREDENTIAL_ID = 'cred_family_messages_mirror'
const AUTH_SESSION_ID = 'auth_family_messages_caregiver'

test('family text message travels from caregiver to Mirror and records interaction state', async (t) => {
  const replicaSet = await MongoMemoryReplSet.create({ replSet: { count: 1, storageEngine: 'wiredTiger' } })
  const objectStoreDir = await mkdtemp(join('/tmp', 'reflexion-family-messages-'))
  const originalEnvironment = { ...process.env }
  process.env.NODE_ENV = 'test'
  process.env.MONGODB_URI = replicaSet.getUri()
  process.env.JWT_SECRET = 'family-messages-jwt-secret-at-least-32-characters'
  process.env.PAIRING_PEPPER = 'family-messages-pairing-pepper-at-least-32-chars'
  process.env.CREDENTIAL_ENCRYPTION_KEY = 'family-messages-encryption-key-at-least-32-chars'
  process.env.AUTH_RATE_LIMIT_PER_MINUTE = '1000'
  process.env.API_RATE_LIMIT_PER_MINUTE = '5000'
  process.env.OBJECT_STORE_DRIVER = 'local'
  process.env.OBJECT_STORE_LOCAL_BASE_URL = 'http://object-store.test'
  process.env.OBJECT_STORE_LOCAL_DIR = objectStoreDir
  process.env.OBJECT_STORE_LOCAL_SECRET = 'family-messages-object-store-secret'

  const { createApp } = await import('../../app.js')
  const app = request(createApp())
  const db = await getDb()
  const now = new Date()

  await db.collection<any>(collections.patients).insertOne({
    _id: PATIENT_ID, tenantId: TENANT_ID, displayName: 'Mei Ling', preferredLanguage: 'mandarin',
    timezone: 'Asia/Singapore', ageBand: '75_84', status: 'active', version: 1,
  })
  await db.collection<any>(collections.careRelationships).insertOne({
    _id: 'rel_family_messages', tenantId: TENANT_ID, patientId: PATIENT_ID, userId: USER_ID,
    relationshipType: 'caregiver',
    scopes: ['patient:read', 'patient:write'], status: 'active', validFrom: now, validTo: null,
  })
  await db.collection<any>(collections.authSessions).insertOne({
    _id: AUTH_SESSION_ID, tenantId: TENANT_ID, userId: USER_ID, status: 'active',
    refreshExpiresAt: new Date(Date.now() + 86_400_000),
  })
  await db.collection<any>(collections.devices).insertOne({
    _id: DEVICE_ID, tenantId: TENANT_ID, status: 'active', serialHash: 'family-message-serial-hash',
  })
  await db.collection<any>(collections.credentials).insertOne({
    _id: CREDENTIAL_ID, deviceId: DEVICE_ID, status: 'active', refreshExpiresAt: new Date(Date.now() + 86_400_000),
  })
  await db.collection<any>(collections.assignments).insertOne({
    _id: 'asg_family_messages', tenantId: TENANT_ID, deviceId: DEVICE_ID, patientId: PATIENT_ID, status: 'active',
  })

  const caregiver = { Authorization: `Bearer ${issueAccessToken({
    sub: USER_ID, kind: 'human', tid: TENANT_ID, uid: USER_ID, sid: AUTH_SESSION_ID,
    roles: ['caregiver'], scopes: [],
  }, 3600)}` }
  const mirror = { Authorization: `Bearer ${issueAccessToken({
    sub: DEVICE_ID, kind: 'device', tid: TENANT_ID, did: DEVICE_ID, pid: PATIENT_ID, cid: CREDENTIAL_ID,
    roles: ['device'], scopes: ['device:heartbeat'],
  }, 3600)}` }

  t.after(async () => {
    await new Promise((resolve) => setTimeout(resolve, 50))
    await closeMongo()
    await replicaSet.stop()
    await rm(objectStoreDir, { recursive: true, force: true })
    process.env = originalEnvironment
  })

  const created = await app.post(`/api/v1/patients/${PATIENT_ID}/family-messages`)
    .set({ ...caregiver, 'Idempotency-Key': 'family_message_create_1' })
    .send({ kind: 'text', senderName: 'Mei', text: 'Thinking of you today.' })
    .expect(201)
  const messageId = created.body.data.messageId
  assert.ok(messageId)
  assert.equal(created.body.data.deliveryState, 'queued')
  assert.equal(created.body.data.interactionState, null)

  const delivered = await app.get(`/api/v1/devices/${DEVICE_ID}/family-messages`).set(mirror).expect(200)
  assert.equal(delivered.body.data.length, 1)
  assert.equal(delivered.body.data[0].messageId, messageId)
  assert.equal(delivered.body.data[0].deliveryState, 'delivered')

  await app.post(`/api/v1/devices/${DEVICE_ID}/family-messages/${messageId}/interactions`)
    .set({ ...mirror, 'Idempotency-Key': `family_message_${messageId}_viewed` })
    .send({ interaction: 'viewed' })
    .expect(200)

  const audio = Buffer.from('local voice reply fixture')
  const hash = createHash('sha256').update(audio).digest('hex')
  const clientReplyId = 'reply-client-id-1'
  const plan = await app.post(`/api/v1/devices/${DEVICE_ID}/family-messages/${messageId}/voice-replies/upload-plans`)
    .set({ ...mirror, 'Idempotency-Key': 'voice_reply_plan_1' })
    .send({ clientReplyId, contentType: 'audio/webm', hash, sizeBytes: audio.byteLength, durationMs: 2400, sessionId: 'session_family_reply_1' })
    .expect(200)
  assert.equal(plan.body.data.state, 'queued')
  assert.ok(plan.body.data.uploadUrl)
  assert.deepEqual(plan.body.data.requiredHeaders, { 'content-type': 'audio/webm', 'x-reflexion-sha256': hash })

  const replyId = plan.body.data.replyId
  const objectPath = join(objectStoreDir, TENANT_ID, PATIENT_ID, 'family-replies', messageId, replyId)
  await mkdir(dirname(objectPath), { recursive: true })
  await writeFile(objectPath, audio)

  const committed = await app.post(`/api/v1/devices/${DEVICE_ID}/family-messages/${messageId}/voice-replies/${replyId}/commit`)
    .set({ ...mirror, 'Idempotency-Key': 'voice_reply_commit_1' })
    .send({ hash, sizeBytes: audio.byteLength })
    .expect(200)
  assert.equal(committed.body.data.state, 'sent')
  assert.equal(committed.body.data.originalMessageId, messageId)
  assert.equal(committed.body.data.threadId, messageId)
  assert.equal(committed.body.data.durationMs, 2400)
  assert.match(committed.body.data.audioUrl, /^http:\/\/object-store\.test\//)

  const repeatedPlan = await app.post(`/api/v1/devices/${DEVICE_ID}/family-messages/${messageId}/voice-replies/upload-plans`)
    .set({ ...mirror, 'Idempotency-Key': 'voice_reply_plan_2' })
    .send({ clientReplyId, contentType: 'audio/webm', hash, sizeBytes: audio.byteLength, durationMs: 2400 })
  assert.equal(repeatedPlan.status, 200, JSON.stringify(repeatedPlan.body))
  assert.equal(repeatedPlan.body.data.replyId, replyId)
  assert.equal(repeatedPlan.body.data.state, 'sent')
  assert.equal(repeatedPlan.body.data.alreadySent, true)

  const repeatedCommit = await app.post(`/api/v1/devices/${DEVICE_ID}/family-messages/${messageId}/voice-replies/${replyId}/commit`)
    .set({ ...mirror, 'Idempotency-Key': 'voice_reply_commit_2' })
    .send({ hash, sizeBytes: audio.byteLength })
    .expect(200)
  assert.equal(repeatedCommit.body.data.state, 'sent')

  const audioUrl = new URL(committed.body.data.audioUrl)
  const downloaded = await app.get(`${audioUrl.pathname}${audioUrl.search}`).expect(200)
  assert.deepEqual(downloaded.body, audio)

  const caregiverReadback = await app.get(`/api/v1/patients/${PATIENT_ID}/family-messages`).set(caregiver).expect(200)
  assert.equal(caregiverReadback.body.data[0].messageId, messageId)
  assert.equal(caregiverReadback.body.data[0].deliveryState, 'delivered')
  assert.equal(caregiverReadback.body.data[0].interactionState, 'viewed')
  assert.equal(caregiverReadback.body.data[0].voiceReplies.length, 1)
  assert.equal(caregiverReadback.body.data[0].voiceReplies[0].replyId, replyId)
  assert.equal(caregiverReadback.body.data[0].voiceReplies[0].state, 'sent')
  assert.equal(new URL(caregiverReadback.body.data[0].voiceReplies[0].audioUrl).pathname, new URL(committed.body.data.audioUrl).pathname)
})
