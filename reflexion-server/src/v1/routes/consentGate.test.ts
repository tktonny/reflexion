import assert from 'node:assert/strict'
import test from 'node:test'
import { MongoMemoryReplSet } from 'mongodb-memory-server'
import request from 'supertest'

import { closeMongo, getDb } from '../../lib/mongo.js'
import { collections } from '../platform/collections.js'
import { issueAccessToken } from '../platform/tokens.js'

// Consent is a HARD gate, not bookkeeping: POST /sessions refuses a `daily_checkin` with 403
// CONSENT_REQUIRED unless a granted `home_cognitive_monitoring` consent exists, and the monitoring pipeline
// excludes any session without a consentRef. No patient-creation path had ever written one, so the core
// daily check-in could never start for any patient — while `companion` sessions, which are exempt, worked.
// These tests pin the gate, the new read route, and that the backfill actually unblocks a check-in.

const TENANT_ID = 'ten_consent_test'
const USER_ID = 'usr_consent_caregiver'
const PATIENT_ID = 'pat_consent_subject'
const DEVICE_ID = 'dev_consent_mirror'
const CREDENTIAL_ID = 'cred_consent_mirror'
const AUTH_SESSION_ID = 'auth_consent_caregiver'

test('a patient without consent cannot run a check-in, and the backfill unblocks it', async (t) => {
  const replicaSet = await MongoMemoryReplSet.create({ replSet: { count: 1, storageEngine: 'wiredTiger' } })
  const originalEnvironment = { ...process.env }
  process.env.NODE_ENV = 'test'
  process.env.MONGODB_URI = replicaSet.getUri()
  process.env.JWT_SECRET = 'consent-gate-jwt-secret-at-least-32-characters'
  process.env.PAIRING_PEPPER = 'consent-gate-pairing-pepper-at-least-32-chars'
  process.env.CREDENTIAL_ENCRYPTION_KEY = 'consent-gate-encryption-key-at-least-32-chars'
  process.env.AUTH_RATE_LIMIT_PER_MINUTE = '1000'
  process.env.API_RATE_LIMIT_PER_MINUTE = '5000'

  const { createApp } = await import('../../app.js')
  const app = request(createApp())
  const db = await getDb()
  const now = new Date()

  await db.collection<any>(collections.patients).insertOne({
    _id: PATIENT_ID, tenantId: TENANT_ID, displayName: 'Mei Ling', preferredLanguage: 'mandarin',
    timezone: 'Asia/Singapore', ageBand: '75_84', status: 'active', version: 1,
  })
  await db.collection<any>(collections.careRelationships).insertOne({
    _id: 'rel_consent', tenantId: TENANT_ID, patientId: PATIENT_ID, userId: USER_ID,
    relationshipType: 'caregiver',
    scopes: ['patient:read', 'patient:write', 'monitoring:read', 'session:write', 'care_plan:read'],
    status: 'active', validFrom: now, validTo: null,
  })
  await db.collection<any>(collections.authSessions).insertOne({
    _id: AUTH_SESSION_ID, tenantId: TENANT_ID, userId: USER_ID, status: 'active',
    refreshExpiresAt: new Date(Date.now() + 86_400_000),
  })
  await db.collection<any>(collections.credentials).insertOne({
    _id: CREDENTIAL_ID, deviceId: DEVICE_ID, status: 'active', refreshExpiresAt: new Date(Date.now() + 86_400_000),
  })
  await db.collection<any>(collections.devices).insertOne({
    _id: DEVICE_ID, tenantId: TENANT_ID, status: 'active', displayName: 'Consent Mirror', createdAt: now,
  })
  await db.collection<any>(collections.assignments).insertOne({
    _id: 'asg_consent', tenantId: TENANT_ID, deviceId: DEVICE_ID, patientId: PATIENT_ID, status: 'active',
  })

  const caregiver = { Authorization: `Bearer ${issueAccessToken({
    sub: USER_ID, kind: 'human', tid: TENANT_ID, uid: USER_ID, sid: AUTH_SESSION_ID,
    roles: ['caregiver'], scopes: [],
  }, 3600)}` }
  const mirror = { Authorization: `Bearer ${issueAccessToken({
    sub: DEVICE_ID, kind: 'device', tid: TENANT_ID, did: DEVICE_ID, pid: PATIENT_ID, cid: CREDENTIAL_ID,
    roles: ['device'], scopes: ['session:write', 'session:read', 'consent:read', 'consent:write'],
  }, 3600)}` }

  t.after(async () => {
    await new Promise((resolve) => setTimeout(resolve, 50))
    await closeMongo()
    await replicaSet.stop()
    process.env = originalEnvironment
  })

  await t.test('the read route says exactly what is missing', async () => {
    const state = await app.get(`/api/v1/patients/${PATIENT_ID}/consents`).set(caregiver).expect(200)
    assert.deepEqual(state.body.data.consents, [])
    assert.deepEqual(state.body.data.requiredPurposes, ['home_cognitive_monitoring'])
    assert.deepEqual(state.body.data.missingPurposes, ['home_cognitive_monitoring'])
  })

  await t.test('the daily check-in is refused, while a companion chat is not', async () => {
    const refused = await app.post('/api/v1/sessions').set({ ...mirror, 'Idempotency-Key': 'consent_checkin_1' })
      .send({ type: 'daily_checkin', requestedLanguage: 'mandarin' }).expect(403)
    assert.equal(refused.body.error.code, 'CONSENT_REQUIRED')

    // The exemption is why the mirror appeared to work: companion chats never needed consent.
    const companion = await app.post('/api/v1/sessions').set({ ...mirror, 'Idempotency-Key': 'consent_companion_1' })
      .send({ type: 'companion', requestedLanguage: 'mandarin' }).expect(201)
    assert.equal(companion.body.data.type, 'companion')
  })

  await t.test('the caregiver cannot grant product consent, while the Mirror can record a decline', async () => {
    const caregiverGrant = await app.post(`/api/v1/patients/${PATIENT_ID}/consents`)
      .set({ ...caregiver, 'Idempotency-Key': 'consent_caregiver_grant_denied_1' })
      .send({ purpose: 'home_cognitive_monitoring', documentVersion: 'v1', status: 'granted' })
      .expect(403)
    assert.equal(caregiverGrant.body.error.code, 'OLDER_ADULT_CONSENT_REQUIRED')

    const declined = await app.post(`/api/v1/patients/${PATIENT_ID}/consents`)
      .set({ ...mirror, 'Idempotency-Key': 'consent_mirror_declined_1' })
      .send({ purpose: 'home_cognitive_monitoring', documentVersion: 'v1', status: 'declined' })
      .expect(201)
    assert.equal(declined.body.data.status, 'declined')
    const mirrorState = await app.get(`/api/v1/patients/${PATIENT_ID}/consents`).set(mirror).expect(200)
    assert.deepEqual(mirrorState.body.data.missingPurposes, ['home_cognitive_monitoring'])
    const configuration = await app.get(`/api/v1/devices/${DEVICE_ID}/configuration`).set(mirror).expect(200)
    assert.equal(configuration.body.data.patient.consent.status, 'declined')
  })

  await t.test('the backfill grants it, marked so it can never pass for a real consent', async () => {
    const { BACKFILL_CONSENT_DOCUMENT_VERSION, ensureCheckInConsent } = await import('../../lib/legacyV1Bridge.js')
    assert.equal(await ensureCheckInConsent(db, { tenantId: TENANT_ID, patientId: PATIENT_ID, actorId: USER_ID }), 'created')
    // Idempotent — a second run must not stack another record.
    assert.equal(await ensureCheckInConsent(db, { tenantId: TENANT_ID, patientId: PATIENT_ID, actorId: USER_ID }), 'present')

    const rows = await db.collection<any>(collections.consents).find({ patientId: PATIENT_ID }).toArray()
    assert.equal(rows.length, 2)
    const backfill = rows.find((row: any) => row.documentVersion === BACKFILL_CONSENT_DOCUMENT_VERSION)
    assert.ok(backfill)
    assert.equal(backfill.source, 'legacy_onboarding_backfill', 'provenance must be queryable')
    assert.equal(backfill.actorId, USER_ID)
  })

  await t.test('the same check-in now succeeds and carries the consent reference', async () => {
    const created = await app.post('/api/v1/sessions').set({ ...mirror, 'Idempotency-Key': 'consent_checkin_2' })
      .send({ type: 'daily_checkin', requestedLanguage: 'mandarin' }).expect(201)
    assert.equal(created.body.data.type, 'daily_checkin')

    // consentRef is what stops the monitoring pipeline excluding the session later.
    const session = await db.collection<any>(collections.sessions).findOne({ _id: created.body.data.sessionId })
    assert.ok(session?.consentRef?.consentId, 'without consentRef the pipeline excludes the session')
    assert.equal(session.consentRef.purpose, 'home_cognitive_monitoring')

    await app.post(`/api/v1/devices/${DEVICE_ID}/consent`)
      .set({ ...mirror, 'Idempotency-Key': 'consent_mirror_declined_2' })
      .send({ documentVersion: 'v1', status: 'declined' }).expect(201)
    const blockedAgain = await app.post('/api/v1/sessions').set({ ...mirror, 'Idempotency-Key': 'consent_checkin_declined_again' })
      .send({ type: 'daily_checkin', requestedLanguage: 'mandarin' }).expect(403)
    assert.equal(blockedAgain.body.error.code, 'CONSENT_REQUIRED')
    await app.post(`/api/v1/devices/${DEVICE_ID}/consent`)
      .set({ ...mirror, 'Idempotency-Key': 'consent_mirror_granted_2' })
      .send({ documentVersion: 'v1', status: 'granted' }).expect(201)

    const state = await app.get(`/api/v1/patients/${PATIENT_ID}/consents`).set(caregiver).expect(200)
    assert.deepEqual(state.body.data.missingPurposes, [])
  })

  await t.test('withdrawing consent blocks check-ins again', async () => {
    await app.post(`/api/v1/patients/${PATIENT_ID}/consents`)
      .set({ ...caregiver, 'Idempotency-Key': 'consent_withdraw_1' })
      .send({ purpose: 'home_cognitive_monitoring', documentVersion: 'v1', status: 'withdrawn' })
      .expect(201)

    const refused = await app.post('/api/v1/sessions').set({ ...mirror, 'Idempotency-Key': 'consent_checkin_3' })
      .send({ type: 'daily_checkin', requestedLanguage: 'mandarin' }).expect(403)
    assert.equal(refused.body.error.code, 'CONSENT_REQUIRED')

    const state = await app.get(`/api/v1/patients/${PATIENT_ID}/consents`).set(caregiver).expect(200)
    assert.deepEqual(state.body.data.missingPurposes, ['home_cognitive_monitoring'])
  })
})
