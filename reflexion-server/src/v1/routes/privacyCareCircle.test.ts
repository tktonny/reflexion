import assert from 'node:assert/strict'
import test from 'node:test'
import { MongoMemoryReplSet } from 'mongodb-memory-server'
import request from 'supertest'

import { createApp } from '../../app.js'
import { closeMongo, getDb } from '../../lib/mongo.js'
import { processDataDeletionRequest } from '../privacy/dataDeletion.js'
import { collections } from '../platform/collections.js'
import { ensureV1Indexes } from '../platform/indexes.js'
import { issueAccessToken } from '../platform/tokens.js'

const TENANT_ID = 'ten_privacy_circle'
const USER_ID = 'usr_privacy_caregiver'
const PATIENT_ID = 'pat_privacy_loved_one'
const SESSION_ID = 'ses_privacy_session'
const AUTH_SESSION_ID = 'auth_privacy_caregiver'

test('privacy, consent and Care Circle controls are durable and scoped', async (t) => {
  const originalEnvironment = { ...process.env }
  const replicaSet = await MongoMemoryReplSet.create({ replSet: { count: 1, storageEngine: 'wiredTiger' } })
  process.env.NODE_ENV = 'test'
  process.env.MONGODB_URI = replicaSet.getUri()
  process.env.MONGODB_DB = 'reflexion_privacy_circle'
  process.env.JWT_SECRET = 'privacy-circle-jwt-secret-at-least-32-characters'
  process.env.PAIRING_PEPPER = 'privacy-circle-pairing-pepper-at-least-32-chars'
  process.env.CREDENTIAL_ENCRYPTION_KEY = 'privacy-circle-encryption-key-at-least-32-chars'
  process.env.AUTH_RATE_LIMIT_PER_MINUTE = '1000'
  process.env.API_RATE_LIMIT_PER_MINUTE = '5000'

  const db = await getDb()
  await ensureV1Indexes(db)
  const now = new Date()
  await db.collection<any>(collections.users).insertOne({ _id: USER_ID, tenantId: TENANT_ID, name: 'Chloe', email: 'chloe@example.com', roles: ['caregiver'], status: 'active', createdAt: now })
  await db.collection<any>(collections.authSessions).insertOne({ _id: AUTH_SESSION_ID, tenantId: TENANT_ID, userId: USER_ID, status: 'active', refreshExpiresAt: new Date(Date.now() + 86_400_000) })
  await db.collection<any>(collections.patients).insertOne({ _id: PATIENT_ID, tenantId: TENANT_ID, displayName: 'Mum', preferredLanguage: 'en', timezone: 'Asia/Singapore', status: 'active', version: 1, createdAt: now, updatedAt: now })
  await db.collection<any>(collections.careRelationships).insertOne({
    _id: 'rel_privacy', tenantId: TENANT_ID, patientId: PATIENT_ID, userId: USER_ID, relationshipType: 'caregiver',
    scopes: ['patient:read', 'patient:write', 'care_plan:read', 'care_plan:write', 'monitoring:read', 'session:read'], status: 'active', validFrom: now, validTo: null, createdAt: now,
  })
  await db.collection<any>(collections.consents).insertOne({ _id: 'con_privacy', tenantId: TENANT_ID, patientId: PATIENT_ID, purpose: 'home_cognitive_monitoring', documentVersion: 'v1', status: 'granted', signedAt: now, withdrawnAt: null, actorId: USER_ID, createdAt: now })
  await db.collection<any>(collections.sessions).insertOne({ _id: SESSION_ID, tenantId: TENANT_ID, patientId: PATIENT_ID, state: 'completed', createdAt: now })
  await db.collection<any>(collections.patientMemory).insertOne({ _id: PATIENT_ID, tenantId: TENANT_ID, patientId: PATIENT_ID, facts: ['A private continuity fact'], updatedAt: now })
  await db.collection<any>(collections.transcriptTurns).insertOne({ _id: 'turn_privacy', tenantId: TENANT_ID, patientId: PATIENT_ID, sessionId: SESSION_ID, turnId: 't1', text: 'A private conversation' })
  await db.collection<any>(collections.familyMessages).insertOne({ _id: 'fmsg_privacy', tenantId: TENANT_ID, patientId: PATIENT_ID, body: 'A private message', state: 'opened', createdAt: now })
  await db.collection<any>(collections.reminderOccurrences).insertOne({ _id: 'rem_privacy', tenantId: TENANT_ID, patientId: PATIENT_ID, type: 'routine', state: 'reported-complete', scheduledAt: now })
  await db.collection<any>(collections.assignments).insertOne({ _id: 'asg_privacy', tenantId: TENANT_ID, patientId: PATIENT_ID, deviceId: 'dev_privacy', status: 'active' })
  await db.collection<any>(collections.deviceTelemetry).insertOne({ _id: 'tele_privacy', meta: { tenantId: TENANT_ID, deviceId: 'dev_privacy', kind: 'heartbeat' }, recordedAt: now })

  const app = request(createApp())
  const bearer = { Authorization: `Bearer ${issueAccessToken({ sub: USER_ID, kind: 'human', tid: TENANT_ID, uid: USER_ID, sid: AUTH_SESSION_ID, roles: ['caregiver'], scopes: [] }, 3600)}` }
  t.after(async () => {
    await closeMongo()
    await replicaSet.stop()
    process.env = originalEnvironment
  })

  await t.test('privacy view separates consent and returns retention/deletion choices', async () => {
    const response = await app.get(`/api/v1/patients/${PATIENT_ID}/privacy`).set(bearer).expect(200)
    assert.equal(response.body.data.consent.status, 'accepted')
    assert.equal(response.body.data.research.status, 'separate')
    assert.deepEqual(response.body.data.deletionCategories.map((item: { category: string }) => item.category), ['sessions', 'messages', 'routine-responses', 'device-events'])
  })

  let invitationId = ''
  await t.test('invite, edit permissions and revoke a Care Circle invitation', async () => {
    const invited = await app.post(`/api/v1/patients/${PATIENT_ID}/care-circle/invitations`)
      .set({ ...bearer, 'Idempotency-Key': 'privacy_circle_invite_0001' })
      .send({ emailOrPhone: 'sam@example.com', role: 'view-only' }).expect(202)
    invitationId = invited.body.data.memberId
    assert.equal(invited.body.data.permissions[0], 'view-loved-ones')
    const edited = await app.patch(`/api/v1/patients/${PATIENT_ID}/care-circle/${invitationId}`).set(bearer)
      .send({ role: 'custom-access', permissions: ['view-loved-ones', 'receive-notifications'] }).expect(200)
    assert.deepEqual(edited.body.data.permissions, ['view-loved-ones', 'receive-notifications'])
    const list = await app.get(`/api/v1/patients/${PATIENT_ID}/care-circle`).set(bearer).expect(200)
    assert.equal(list.body.data.invitations.length, 1)
    await app.delete(`/api/v1/patients/${PATIENT_ID}/care-circle/${invitationId}`)
      .set({ ...bearer, 'Idempotency-Key': 'privacy_circle_revoke_0001' }).expect(202)
    const after = await app.get(`/api/v1/patients/${PATIENT_ID}/care-circle`).set(bearer).expect(200)
    assert.equal(after.body.data.invitations.length, 0)
    assert.equal((await db.collection<any>(collections.careCircleInvitations).findOne({ _id: invitationId }))?.state, 'revoked')
  })

  await t.test('consent withdrawal is visible and blocks future capture', async () => {
    const caregiverGrant = await app.post(`/api/v1/patients/${PATIENT_ID}/consents`)
      .set({ ...bearer, 'Idempotency-Key': 'privacy_consent_grant_denied_0001' })
      .send({ purpose: 'home_cognitive_monitoring', documentVersion: 'v1', status: 'granted' }).expect(403)
    assert.equal(caregiverGrant.body.error.code, 'OLDER_ADULT_CONSENT_REQUIRED')
    await app.post(`/api/v1/patients/${PATIENT_ID}/consents`).set({ ...bearer, 'Idempotency-Key': 'privacy_consent_withdraw_0001' })
      .send({ purpose: 'home_cognitive_monitoring', documentVersion: 'v1', status: 'withdrawn' }).expect(201)
    const privacy = await app.get(`/api/v1/patients/${PATIENT_ID}/privacy`).set(bearer).expect(200)
    assert.equal(privacy.body.data.consent.status, 'withdrawn')
  })

  await t.test('selected deletion is queued, processed and idempotent', async () => {
    const requested = await app.post(`/api/v1/patients/${PATIENT_ID}/data-deletion-requests`)
      .set({ ...bearer, 'Idempotency-Key': 'privacy_delete_selected_0001' })
      .send({ confirm: true, categories: ['sessions', 'messages', 'routine-responses', 'device-events'] }).expect(202)
    const requestId = requested.body.data.requestId
    await processDataDeletionRequest(db, requestId)
    await processDataDeletionRequest(db, requestId)
    const state = await db.collection<any>(collections.dataDeletionRequests).findOne({ _id: requestId })
    assert.equal(state?.state, 'completed')
    assert.equal(await db.collection<any>(collections.sessions).countDocuments({ _id: SESSION_ID }), 0)
    assert.equal(await db.collection<any>(collections.transcriptTurns).countDocuments({ sessionId: SESSION_ID }), 0)
    assert.equal(await db.collection<any>(collections.patientMemory).countDocuments({ _id: PATIENT_ID }), 0)
    assert.equal(await db.collection<any>(collections.familyMessages).countDocuments({ patientId: PATIENT_ID }), 0)
    assert.equal(await db.collection<any>(collections.reminderOccurrences).countDocuments({ patientId: PATIENT_ID }), 0)
    assert.equal(await db.collection<any>(collections.deviceTelemetry).countDocuments({ 'meta.deviceId': 'dev_privacy' }), 0)
  })
})
