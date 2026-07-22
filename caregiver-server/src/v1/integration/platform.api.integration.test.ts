import assert from 'node:assert/strict'
import test from 'node:test'
import { MongoMemoryReplSet } from 'mongodb-memory-server'
import request from 'supertest'

import { createApp } from '../../app.js'
import { closeMongo, getDb } from '../../lib/mongo.js'
import { collections } from '../platform/collections.js'
import { ensureV1Indexes } from '../platform/indexes.js'
import { issueAccessToken } from '../platform/tokens.js'

const TENANT_ID = 'ten_api_integration'
const USER_ID = 'usr_api_admin'
const PATIENT_ID = 'pat_api_primary'
const DEVICE_ID = 'dev_api_primary'
const AUTH_SESSION_ID = 'auth_api_admin'
const DEVICE_CREDENTIAL_ID = 'cred_api_primary'

const humanToken = () => issueAccessToken({
  sub: USER_ID,
  kind: 'human',
  tid: TENANT_ID,
  uid: USER_ID,
  sid: AUTH_SESSION_ID,
  roles: ['tenant_admin', 'provider'],
  scopes: [],
}, 3600)

const deviceToken = () => issueAccessToken({
  sub: DEVICE_ID,
  kind: 'device',
  tid: TENANT_ID,
  did: DEVICE_ID,
  pid: PATIENT_ID,
  cid: DEVICE_CREDENTIAL_ID,
  roles: ['device'],
  scopes: ['session:write', 'session:read', 'care_plan:read', 'reminder:respond', 'device:heartbeat'],
}, 3600)

const bearer = (token: string) => ({ Authorization: `Bearer ${token}` })
const idempotencyKey = (label: string) => `${label}_platform_api_000001`

test('all unified v1 business routes work through HTTP with real Mongo authorization', async (t) => {
  const originalFetch = globalThis.fetch
  const originalEnvironment = { ...process.env }
  const replicaSet = await MongoMemoryReplSet.create({ replSet: { count: 1, storageEngine: 'wiredTiger' } })
  process.env.NODE_ENV = 'test'
  process.env.MONGODB_URI = replicaSet.getUri()
  process.env.MONGODB_DB = 'reflexion_platform_api_integration'
  process.env.JWT_SECRET = 'platform-api-jwt-secret-at-least-32-characters'
  process.env.PAIRING_PEPPER = 'platform-api-pairing-pepper-at-least-32-characters'
  process.env.CREDENTIAL_ENCRYPTION_KEY = 'platform-api-encryption-key-at-least-32-characters'
  process.env.AUTH_RATE_LIMIT_PER_MINUTE = '1000'
  process.env.API_RATE_LIMIT_PER_MINUTE = '5000'
  process.env.BRAVE_SEARCH_API_KEY = 'integration-brave-key'

  globalThis.fetch = async (input, init) => {
    const url = String(input)
    if (url.startsWith('https://api.open-meteo.com/v1/forecast')) {
      return Response.json({
        timezone: 'Asia/Shanghai',
        current: {
          time: new Date().toISOString(), temperature_2m: 26, apparent_temperature: 27,
          precipitation: 0, weather_code: 1, wind_speed_10m: 8,
        },
        daily: {
          time: ['2026-07-22'], weather_code: [1], temperature_2m_max: [29], temperature_2m_min: [23],
          precipitation_probability_max: [10], sunrise: ['2026-07-22T05:10'], sunset: ['2026-07-22T19:05'],
        },
      })
    }
    if (url.startsWith('https://geocoding-api.open-meteo.com/v1/search')) {
      return Response.json({ results: [{ latitude: 31.23, longitude: 121.47, name: 'Shanghai', country: 'China', timezone: 'Asia/Shanghai' }] })
    }
    if (url.startsWith('https://api.search.brave.com/res/v1/web/search')) {
      assert.equal(new Headers(init?.headers).get('x-subscription-token'), 'integration-brave-key')
      return Response.json({ web: { results: [{ title: 'Trusted result', url: 'https://example.org/result', description: 'A safe result', age: '1 hour ago' }] } })
    }
    throw new Error(`Unexpected provider request: ${url}`)
  }

  try {
    const db: any = await getDb()
    await ensureV1Indexes(db)
    await seedPlatform(db)
    const app = request(createApp())
    const human = humanToken()
    const device = deviceToken()
    let createdPatientId = ''

    await t.test('patient CRUD, relationships, consent and enrollment routes', async () => {
      await app.get('/api/v1/patients').set(bearer(device)).expect(401)
      const initial = await app.get('/api/v1/patients?limit=1').set(bearer(human)).expect(200)
      assert.equal(initial.body.data.length, 1)

      const created = await app.post('/api/v1/patients')
        .set({ ...bearer(human), 'Idempotency-Key': idempotencyKey('create-patient') })
        .send({ displayName: 'Helen', preferredLanguage: 'en-SG', timezone: 'Asia/Singapore', ageBand: '75-84', relationshipType: 'daughter' })
        .expect(201)
      createdPatientId = created.body.data.patientId
      assert.equal(created.body.data.version, 1)

      const patient = await app.get(`/api/v1/patients/${createdPatientId}`).set(bearer(human)).expect(200)
      assert.equal(patient.body.data.displayName, 'Helen')
      const versionConflict = await app.patch(`/api/v1/patients/${createdPatientId}`)
        .set({ ...bearer(human), 'If-Match': '9' })
        .send({ displayName: 'Wrong version' })
        .expect(409)
      assert.equal(versionConflict.body.error.code, 'VERSION_CONFLICT')
      const updated = await app.patch(`/api/v1/patients/${createdPatientId}`)
        .set({ ...bearer(human), 'If-Match': '1' })
        .send({ displayName: 'Helen Tan', preferredLanguage: 'zh-CN' })
        .expect(200)
      assert.equal(updated.body.data.version, 2)

      const relationships = await app.get(`/api/v1/patients/${createdPatientId}/care-relationships`).set(bearer(human)).expect(200)
      assert.equal(relationships.body.data.length, 1)
      assert.equal(relationships.body.data[0].relationshipType, 'daughter')

      const consent = await app.post(`/api/v1/patients/${createdPatientId}/consents`)
        .set({ ...bearer(human), 'Idempotency-Key': idempotencyKey('patient-consent') })
        .send({ purpose: 'home_cognitive_monitoring', documentVersion: '2026-07', status: 'granted' })
        .expect(201)
      assert.equal(consent.body.data.status, 'granted')

      await db.collection(collections.programEnrollments).insertOne({
        _id: 'enr_api_current', tenantId: TENANT_ID, patientId: createdPatientId,
        program: 'home-monitoring', status: 'active', enrolledAt: new Date(),
      })
      const enrollment = await app.get(`/api/v1/patients/${createdPatientId}/program-enrollments/current`).set(bearer(human)).expect(200)
      assert.equal(enrollment.body.data.enrollmentId, 'enr_api_current')
    })

    let medicationPlanId = ''
    const occurrenceId = 'rem_api_taken'
    await t.test('care plan, medication, reminder and caregiver task routes', async () => {
      const planBody = {
        effectiveFrom: new Date().toISOString(),
        dailyRoutine: { wakeTime: '08:00' },
        communicationPreferences: { location: { city: 'Shanghai', latitude: 31.23, longitude: 121.47 } },
        safetyNotes: 'Speak slowly.',
      }
      const createdPlan = await app.put(`/api/v1/patients/${PATIENT_ID}/care-plan`)
        .set({ ...bearer(human), 'Idempotency-Key': idempotencyKey('create-care-plan') })
        .send(planBody)
        .expect(201)
      assert.equal(createdPlan.body.data.version, 1)
      const deviceRead = await app.get(`/api/v1/patients/${PATIENT_ID}/care-plan`).set(bearer(device)).expect(200)
      assert.equal(deviceRead.body.data.safetyNotes, 'Speak slowly.')
      const updatedPlan = await app.put(`/api/v1/patients/${PATIENT_ID}/care-plan`)
        .set({ ...bearer(human), 'Idempotency-Key': idempotencyKey('update-care-plan'), 'If-Match': '1' })
        .send({ ...planBody, safetyNotes: 'Speak slowly and clearly.' })
        .expect(200)
      assert.equal(updatedPlan.body.data.version, 2)

      const medication = await app.post(`/api/v1/patients/${PATIENT_ID}/medication-plans`)
        .set({ ...bearer(human), 'Idempotency-Key': idempotencyKey('create-medication') })
        .send({ displayName: 'Afternoon tablet', instructions: 'Take with water', source: 'caregiver', schedule: { timezone: 'Asia/Shanghai', times: ['14:00'] } })
        .expect(201)
      medicationPlanId = medication.body.data.planId
      const medications = await app.get(`/api/v1/patients/${PATIENT_ID}/medication-plans`).set(bearer(device)).expect(200)
      assert.equal(medications.body.data.length, 1)
      const patched = await app.patch(`/api/v1/medication-plans/${medicationPlanId}`)
        .set({ ...bearer(human), 'If-Match': '1' })
        .send({ displayName: 'Afternoon heart tablet', status: 'active' })
        .expect(200)
      assert.equal(patched.body.data.version, 2)
      await app.post(`/api/v1/patients/${PATIENT_ID}/medication-plans`)
        .set({ ...bearer(device), 'Idempotency-Key': idempotencyKey('device-medication-write') })
        .send({ displayName: 'Unsafe', source: 'caregiver', schedule: { timezone: 'Asia/Shanghai', times: ['10:00'] } })
        .expect(403)

      const now = Date.now()
      await db.collection(collections.reminderOccurrences).insertMany([
        {
          _id: occurrenceId, tenantId: TENANT_ID, patientId: PATIENT_ID, sourceId: medicationPlanId,
          scheduledAt: new Date(now + 60 * 60_000), type: 'medication', displayText: 'Afternoon heart tablet', status: 'scheduled',
        },
        {
          _id: 'rem_api_upcoming', tenantId: TENANT_ID, patientId: PATIENT_ID, sourceId: `${medicationPlanId}-second`,
          scheduledAt: new Date(now + 2 * 60 * 60_000), type: 'medication', displayText: 'Evening tablet', status: 'scheduled',
        },
      ])
      const reminders = await app.get(`/api/v1/patients/${PATIENT_ID}/reminder-occurrences`)
        .query({ from: new Date(now).toISOString(), to: new Date(now + 3 * 60 * 60_000).toISOString() })
        .set(bearer(device))
        .expect(200)
      assert.equal(reminders.body.data.length, 2)
      const response = await app.post(`/api/v1/reminder-occurrences/${occurrenceId}/responses`)
        .set({ ...bearer(device), 'Idempotency-Key': idempotencyKey('reminder-response') })
        .send({ status: 'taken', respondedAt: new Date().toISOString(), note: 'Confirmed by patient' })
        .expect(200)
      assert.equal(response.body.data.status, 'taken')

      const task = await app.post(`/api/v1/patients/${PATIENT_ID}/caregiver-tasks`)
        .set({ ...bearer(device), 'Idempotency-Key': idempotencyKey('caregiver-task') })
        .send({ category: 'follow_up', priority: 'routine', title: 'Please call this evening', details: 'Patient requested a call.' })
        .expect(201)
      assert.ok(task.body.data.taskId)
    })

    await t.test('allowlisted assistant tools execute and redact audit inputs', async () => {
      const sessionId = 'ses_api_tools'
      await db.collection(collections.sessions).insertOne({
        _id: sessionId, tenantId: TENANT_ID, patientId: PATIENT_ID, deviceId: DEVICE_ID,
        type: 'companion', state: 'active', stateVersion: 2, acquisition: { language: 'zh-CN' }, createdAt: new Date(), updatedAt: new Date(),
      })
      const invoke = (tool: string, args: Record<string, unknown>, suffix: string) => app
        .post(`/api/v1/sessions/${sessionId}/tool-invocations`)
        .set({ ...bearer(device), 'Idempotency-Key': idempotencyKey(`tool-${suffix}`) })
        .send({ tool, arguments: args })

      const weather = await invoke('weather.get', { latitude: 31.23, longitude: 121.47, city: 'Shanghai' }, 'weather').expect(200)
      assert.equal(weather.body.data.output.current.condition, 'partly_cloudy')
      const searched = await invoke('web.search', { query: 'safe community activity', freshness: 'pw' }, 'search').expect(200)
      assert.equal(searched.body.data.output.results.length, 1)
      const medications = await invoke('medication.list', {}, 'medications').expect(200)
      assert.equal(medications.body.data.output[0].planId, medicationPlanId)
      const upcoming = await invoke('reminders.upcoming', {}, 'reminders').expect(200)
      assert.equal(upcoming.body.data.output.length, 1)

      const searchAudit = await db.collection(collections.toolInvocations).findOne({ tool: 'web.search' })
      assert.deepEqual(searchAudit?.arguments, { queryHashOnly: true, freshness: 'pw' })
      assert.doesNotMatch(JSON.stringify(searchAudit), /safe community activity/)
      await app.post(`/api/v1/sessions/${sessionId}/tool-invocations`)
        .set({ ...bearer(human), 'Idempotency-Key': idempotencyKey('human-tool') })
        .send({ tool: 'medication.list', arguments: {} })
        .expect(401)
    })

    await t.test('deterministic status, monitoring, review and notification routes', async () => {
      const now = new Date()
      await db.collection(collections.operationalBaselines).insertOne({
        _id: 'base_api_operational', tenantId: TENANT_ID, patientId: PATIENT_ID, baselineType: 'reassurance_mvp',
        state: 'complete', sessionCount: 7, window: { days: 14, requiredSessions: 7 }, algorithmVersion: 'reassurance-ewma-v1', revision: 1,
      })
      await db.collection(collections.featureSnapshots).insertOne({
        _id: 'feat_api_1', tenantId: TENANT_ID, patientId: PATIENT_ID, sessionId: 'ses_api_history',
        sessionType: 'daily_checkin', inclusion: 'include', capturedAt: now, schemaVersion: 'home-features-v1', pipelineVersion: 'longitudinal-v1',
      })
      await db.collection(collections.monitoringWindows).insertOne({
        _id: 'win_api_1', tenantId: TENANT_ID, patientId: PATIENT_ID, sessionId: 'ses_api_history',
        windowStart: new Date(now.getTime() - 7 * 86_400_000), windowEnd: now, inclusion: 'include', status: 'on_track',
        sourceScoreId: 'score_api_1', ruleVersion: 'transparent-rules-v1', updatedAt: now,
      })
      await db.collection(collections.baselineModels).insertOne({
        _id: 'base_api_research', baselineId: 'base_api_research', tenantId: TENANT_ID, patientId: PATIENT_ID,
        baselineType: 'longitudinal_research', state: 'active', revision: 1, pipelineVersion: 'longitudinal-v1',
        eligibility: { usableSessions: 12 }, scalarAggregates: {}, createdAt: now,
      })

      const status = await app.get(`/api/v1/patients/${PATIENT_ID}/status`).set(bearer(human)).expect(200)
      assert.equal(status.body.data.baselineState, 'complete')
      const day = now.toISOString().slice(0, 10)
      const away = await app.post(`/api/v1/patients/${PATIENT_ID}/away-periods`)
        .set({ ...bearer(human), 'Idempotency-Key': idempotencyKey('away-period') })
        .send({ startsOn: day, endsOn: day, timezone: 'Asia/Shanghai', reason: 'Family visit' })
        .expect(201)
      assert.equal(away.body.data.state, 'active')
      const flag = await app.post(`/api/v1/patients/${PATIENT_ID}/manual-flags`)
        .set({ ...bearer(human), 'Idempotency-Key': idempotencyKey('manual-flag') })
        .send({ severity: 'needs_attention', reason: 'Caregiver requested a check-in.' })
        .expect(201)
      assert.equal(flag.body.data.severity, 'needs_attention')
      const flaggedStatus = await app.get(`/api/v1/patients/${PATIENT_ID}/status`).set(bearer(human)).expect(200)
      assert.equal(flaggedStatus.body.data.status, 'needs_attention')

      const summary = await app.get(`/api/v1/patients/${PATIENT_ID}/monitoring/summary`).set(bearer(human)).expect(200)
      assert.equal(summary.body.data.providerDetail.baselineId, 'base_api_research')
      const timeline = await app.get(`/api/v1/patients/${PATIENT_ID}/monitoring/timeline`).set(bearer(human)).expect(200)
      assert.equal(timeline.body.data[0].providerDetail.sourceScoreId, 'score_api_1')
      const baseline = await app.get(`/api/v1/patients/${PATIENT_ID}/monitoring/baseline`).set(bearer(human)).expect(200)
      assert.equal(baseline.body.data.longitudinal.state, 'complete')

      await db.collection(collections.reviewCases).insertOne({
        _id: 'case_api_1', tenantId: TENANT_ID, patientId: PATIENT_ID, reason: 'persistent_change',
        priority: 'elevated', state: 'open', sourceRefs: ['ses_api_history'], createdAt: now,
      })
      const cases = await app.get('/api/v1/review-cases?state=open').set(bearer(human)).expect(200)
      assert.equal(cases.body.data[0].caseId, 'case_api_1')
      const reviewCase = await app.get('/api/v1/review-cases/case_api_1').set(bearer(human)).expect(200)
      assert.equal(reviewCase.body.data.dispositions.length, 0)
      const disposition = await app.post('/api/v1/review-cases/case_api_1/dispositions')
        .set({ ...bearer(human), 'Idempotency-Key': idempotencyKey('review-disposition') })
        .send({ outcome: 'follow_up_ordered', notes: 'Arrange a call.', closeCase: false })
        .expect(201)
      assert.equal(disposition.body.data.outcome, 'follow_up_ordered')

      await db.collection(collections.notifications).insertOne({
        _id: 'notif_api_1', tenantId: TENANT_ID, recipientUserId: USER_ID, patientId: PATIENT_ID,
        type: 'worth_checking', state: 'unread', title: 'Worth checking in', body: 'A review is available.',
        source: { type: 'review_case', id: 'case_api_1' }, dedupeKey: 'case_api_1', createdAt: now, updatedAt: now,
      })
      const notifications = await app.get('/api/v1/notifications?state=unread').set(bearer(human)).expect(200)
      assert.equal(notifications.body.data[0].notificationId, 'notif_api_1')
      const read = await app.post('/api/v1/notifications/notif_api_1/read').set(bearer(human)).expect(200)
      assert.equal(read.body.data.state, 'read')
    })
  } finally {
    await new Promise((resolve) => setTimeout(resolve, 50))
    globalThis.fetch = originalFetch
    await closeMongo()
    await replicaSet.stop()
    for (const key of Object.keys(process.env)) if (!(key in originalEnvironment)) delete process.env[key]
    Object.assign(process.env, originalEnvironment)
  }
})

async function seedPlatform(db: any) {
  const now = new Date()
  await db.collection(collections.tenants).insertOne({ _id: TENANT_ID, name: 'API Integration', status: 'active', createdAt: now })
  await db.collection(collections.users).insertOne({
    _id: USER_ID, tenantId: TENANT_ID, authSubject: 'integration:admin', emailNormalized: 'admin@integration.invalid',
    name: 'API Admin', roles: ['tenant_admin', 'provider'], scopes: [], status: 'active', createdAt: now,
  })
  await db.collection(collections.authSessions).insertOne({
    _id: AUTH_SESSION_ID, tenantId: TENANT_ID, userId: USER_ID, status: 'active', refreshExpiresAt: new Date(now.getTime() + 86_400_000), createdAt: now,
  })
  await db.collection(collections.patients).insertOne({
    _id: PATIENT_ID, tenantId: TENANT_ID, displayName: 'Margaret', preferredLanguage: 'zh-CN', timezone: 'Asia/Shanghai', status: 'active', version: 1, createdAt: now,
  })
  await db.collection(collections.careRelationships).insertOne({
    _id: 'rel_api_admin', tenantId: TENANT_ID, patientId: PATIENT_ID, userId: USER_ID, status: 'active', validTo: null,
    scopes: ['patient:read', 'patient:write', 'device:assign', 'care_plan:read', 'care_plan:write', 'monitoring:read', 'review:read', 'review:write'], createdAt: now,
  })
  await db.collection(collections.devices).insertOne({
    _id: DEVICE_ID, serialHash: 'serial-api-integration', tenantId: TENANT_ID, status: 'active', displayName: 'Integration Mirror',
    technicalState: 'ok', lastSeenAt: now, createdAt: now, updatedAt: now,
  })
  await db.collection(collections.assignments).insertOne({
    _id: 'asg_api_primary', tenantId: TENANT_ID, deviceId: DEVICE_ID, patientId: PATIENT_ID,
    assignmentType: 'primary', status: 'active', assignedAt: now, version: 1,
  })
  await db.collection(collections.credentials).insertOne({
    _id: DEVICE_CREDENTIAL_ID, tenantId: TENANT_ID, deviceId: DEVICE_ID, patientId: PATIENT_ID,
    status: 'active', issuedAt: now, refreshExpiresAt: new Date(now.getTime() + 86_400_000), version: 1,
  })
}
