import assert from 'node:assert/strict'
import test from 'node:test'
import { MongoMemoryReplSet } from 'mongodb-memory-server'
import request from 'supertest'

import { closeMongo, getDb } from '../../lib/mongo.js'
import { collections } from '../platform/collections.js'
import { issueAccessToken } from '../platform/tokens.js'

// The month calendar, one day's transcripts and the duration trend existed only as LEGACY routes, which is
// why the caregiver app could not leave the legacy API. Those routes are also tokenless — identity is a
// patient id in a query string — so the authorization boundary asserted here is new, not just moved.

const TENANT_ID = 'ten_history'
const USER_ID = 'usr_history_caregiver'
const OTHER_USER_ID = 'usr_history_stranger'
const PATIENT_ID = 'pat_history_subject'
const AUTH_SESSION_ID = 'auth_history'
const OTHER_AUTH_SESSION_ID = 'auth_history_stranger'

const token = (userId: string, sessionId: string) => `Bearer ${issueAccessToken({
  sub: userId, kind: 'human', tid: TENANT_ID, uid: userId, sid: sessionId, roles: ['caregiver'], scopes: [],
}, 3600)}`

test('the caregiver history read models work over v1 and enforce the care relationship', async (t) => {
  const replicaSet = await MongoMemoryReplSet.create({ replSet: { count: 1, storageEngine: 'wiredTiger' } })
  const originalEnvironment = { ...process.env }
  process.env.NODE_ENV = 'test'
  process.env.MONGODB_URI = replicaSet.getUri()
  process.env.JWT_SECRET = 'history-jwt-secret-at-least-32-characters-long'
  process.env.PAIRING_PEPPER = 'history-pairing-pepper-at-least-32-characters'
  process.env.CREDENTIAL_ENCRYPTION_KEY = 'history-encryption-key-at-least-32-characters'
  process.env.AUTH_RATE_LIMIT_PER_MINUTE = '1000'
  process.env.API_RATE_LIMIT_PER_MINUTE = '5000'

  const { createApp } = await import('../../app.js')
  const app = request(createApp())
  const db = await getDb()
  const now = new Date()

  await db.collection<any>(collections.patients).insertOne({
    _id: PATIENT_ID, tenantId: TENANT_ID, displayName: 'Mei Ling', preferredLanguage: 'mandarin',
    timezone: 'Asia/Singapore', status: 'active', version: 1,
  })
  await db.collection<any>(collections.careRelationships).insertOne({
    _id: 'rel_history', tenantId: TENANT_ID, patientId: PATIENT_ID, userId: USER_ID,
    relationshipType: 'caregiver',
    scopes: ['patient:read', 'monitoring:read', 'session:read'],
    status: 'active', validFrom: now, validTo: null,
  })
  for (const [id, userId] of [[AUTH_SESSION_ID, USER_ID], [OTHER_AUTH_SESSION_ID, OTHER_USER_ID]]) {
    await db.collection<any>(collections.authSessions).insertOne({
      _id: id, tenantId: TENANT_ID, userId, status: 'active',
      refreshExpiresAt: new Date(Date.now() + 86_400_000),
    })
  }

  // Two sessions on the same local day: one completed, one not.
  const day = '2026-07-20'
  const at = (hour: number) => new Date(Date.UTC(2026, 6, 20, hour - 8, 0, 0)) // Asia/Singapore is UTC+8
  await db.collection<any>(collections.sessions).insertMany([
    {
      _id: 'ses_history_done', tenantId: TENANT_ID, patientId: PATIENT_ID, type: 'daily_checkin',
      state: 'completed', createdAt: at(9), localCompletedAt: at(9), updatedAt: at(9),
      acquisition: { language: 'mandarin', durationMs: 240_000, patientTurns: 2 },
    },
    {
      _id: 'ses_history_open', tenantId: TENANT_ID, patientId: PATIENT_ID, type: 'daily_checkin',
      state: 'created', createdAt: at(18), updatedAt: at(18), acquisition: { language: 'mandarin' },
    },
  ])
  await db.collection<any>(collections.transcriptTurns).insertMany([
    { _id: 'turn_1', sessionId: 'ses_history_done', sequence: 1, role: 'assistant', text: '早安，今天感觉怎么样？' },
    { _id: 'turn_2', sessionId: 'ses_history_done', sequence: 2, role: 'patient', text: '还不错，睡得很好。' },
  ])

  t.after(async () => {
    await new Promise((resolve) => setTimeout(resolve, 50))
    await closeMongo()
    await replicaSet.stop()
    process.env = originalEnvironment
  })

  const caregiver = { Authorization: token(USER_ID, AUTH_SESSION_ID) }
  const stranger = { Authorization: token(OTHER_USER_ID, OTHER_AUTH_SESSION_ID) }

  await t.test('the month calendar counts sessions per local day', async () => {
    const month = await app.get(`/api/v1/patients/${PATIENT_ID}/session-days?month=2026-07`).set(caregiver).expect(200)
    const days = month.body.data.days
    assert.equal(days.length, 31, 'a full month, so the grid never has to guess')
    const twentieth = days.find((entry: any) => entry.date === day)
    assert.equal(twentieth.count, 2)
    assert.equal(twentieth.completedCount, 1)
    assert.equal(twentieth.hasCompletedSession, true)
    const quiet = days.find((entry: any) => entry.date === '2026-07-21')
    assert.equal(quiet.count, 0)
    assert.equal(quiet.hasCompletedSession, false)

    await app.get(`/api/v1/patients/${PATIENT_ID}/session-days?month=July`).set(caregiver).expect(400)
    await app.get(`/api/v1/patients/${PATIENT_ID}/session-days`).set(caregiver).expect(400)
  })

  await t.test('one day returns its sessions with transcripts and per-line metrics', async () => {
    const detail = await app.get(`/api/v1/patients/${PATIENT_ID}/session-days/${day}`).set(caregiver).expect(200)
    assert.equal(detail.body.data.patientName, 'Mei Ling')
    assert.equal(detail.body.data.sessions.length, 2)
    const completed = detail.body.data.sessions.find((entry: any) => entry.id === 'ses_history_done')
    assert.equal(completed.duration, 240, 'durationMs is reported in seconds')
    assert.equal(completed.logs.length, 2)
    assert.ok(completed.words > 0, 'words are counted per line')
    assert.equal(completed.exchanges, 2)

    await app.get(`/api/v1/patients/${PATIENT_ID}/session-days/20-07-2026`).set(caregiver).expect(400)
  })

  await t.test('the trend covers the requested window, oldest first', async () => {
    const trend = await app.get(`/api/v1/patients/${PATIENT_ID}/session-trend?days=30`).set(caregiver).expect(200)
    assert.equal(trend.body.data.trend.length, 30)
    const dates = trend.body.data.trend.map((entry: any) => entry.date)
    assert.deepEqual([...dates].sort(), dates, 'oldest first, so the chart reads left to right')
    for (const entry of trend.body.data.trend) {
      assert.equal(typeof entry.missed, 'boolean')
      assert.equal(typeof entry.duration, 'number')
    }

    const week = await app.get(`/api/v1/patients/${PATIENT_ID}/session-trend?days=7`).set(caregiver).expect(200)
    assert.equal(week.body.data.trend.length, 7)
    await app.get(`/api/v1/patients/${PATIENT_ID}/session-trend?days=90`).set(caregiver).expect(400)
  })

  await t.test('a caregiver with no relationship to this patient is refused, unlike the legacy routes', async () => {
    for (const path of [
      `/api/v1/patients/${PATIENT_ID}/session-days?month=2026-07`,
      `/api/v1/patients/${PATIENT_ID}/session-days/${day}`,
      `/api/v1/patients/${PATIENT_ID}/session-trend?days=7`,
    ]) {
      await app.get(path).set(stranger).expect(403)
      await app.get(path).expect(401)
    }
  })

  await t.test('the mirror list includes loved ones with no mirror yet', async () => {
    const before = await app.get('/api/v1/device-assignments').set(caregiver).expect(200)
    assert.equal(before.body.data.assignments.length, 1)
    assert.equal(before.body.data.assignments[0].patientName, 'Mei Ling')
    assert.equal(before.body.data.assignments[0].deviceId, null, 'the row must exist so pairing can be offered')

    await db.collection<any>(collections.devices).insertOne({
      _id: 'dev_history', tenantId: TENANT_ID, serial: 'SN-123', softwareVersion: '1.0.0',
      status: 'active', lastHeartbeatAt: now,
    })
    await db.collection<any>(collections.assignments).insertOne({
      _id: 'asg_history', tenantId: TENANT_ID, deviceId: 'dev_history', patientId: PATIENT_ID,
      assignmentType: 'primary', mirrorName: 'Living room mirror', status: 'active', assignedAt: now, version: 1,
    })

    const after = await app.get('/api/v1/device-assignments').set(caregiver).expect(200)
    const row = after.body.data.assignments[0]
    assert.equal(row.mirrorName, 'Living room mirror')
    assert.equal(row.deviceId, 'dev_history')
    assert.equal(row.device.serial, 'SN-123')
    assert.ok(row.assignedAt)

    // A stranger sees none of it.
    const strangerView = await app.get('/api/v1/device-assignments').set(stranger).expect(200)
    assert.deepEqual(strangerView.body.data.assignments, [])
    await app.get('/api/v1/device-assignments').expect(401)
  })
})
