import assert from 'node:assert/strict'
import test from 'node:test'
import { MongoMemoryReplSet } from 'mongodb-memory-server'

import { closeMongo, getDb } from '../lib/mongo.js'
import { collections } from '../v1/platform/collections.js'

// A session only ever left `created`/`active` when the client called POST /sessions/:id/abandonments. A
// mirror that loses power or drops off the network mid-conversation never makes that call, so production
// accumulated 31 open sessions out of 72 — the oldest two days old, none of which would ever resolve.

const TENANT_ID = 'ten_stale'
const PATIENT_ID = 'pat_stale'
const HOUR_MS = 3_600_000

test('the finalise loop writes off sessions the mirror never finished', async (t) => {
  const replicaSet = await MongoMemoryReplSet.create({ replSet: { count: 1, storageEngine: 'wiredTiger' } })
  const originalEnvironment = { ...process.env }
  process.env.NODE_ENV = 'test'
  process.env.MONGODB_URI = replicaSet.getUri()
  process.env.JWT_SECRET = 'stale-session-jwt-secret-at-least-32-chars'
  process.env.PAIRING_PEPPER = 'stale-session-pairing-pepper-at-least-32-c'
  process.env.CREDENTIAL_ENCRYPTION_KEY = 'stale-session-encryption-key-at-least-32-c'

  const { abandonStaleSessions } = await import('./finalizeDay.js')
  const db = await getDb()

  t.after(async () => {
    await closeMongo()
    await replicaSet.stop()
    process.env = originalEnvironment
  })

  const seed = (id: string, state: string, ageMs: number) => db.collection<any>(collections.sessions).insertOne({
    _id: id, tenantId: TENANT_ID, patientId: PATIENT_ID, type: 'daily_checkin', state, stateVersion: 1,
    createdAt: new Date(Date.now() - ageMs), updatedAt: new Date(Date.now() - ageMs),
  })

  await Promise.all([
    seed('ses_stale_active', 'active', 2 * HOUR_MS),
    seed('ses_stale_created', 'created', 26 * HOUR_MS),
    // Inside the window: someone may still be talking.
    seed('ses_recent_active', 'active', 5 * 60_000),
    // Already resolved — a sweep must not touch a finished conversation.
    seed('ses_old_completed', 'completed', 48 * HOUR_MS),
    seed('ses_old_excluded', 'excluded', 48 * HOUR_MS),
    seed('ses_old_abandoned', 'abandoned', 48 * HOUR_MS),
  ])

  await t.test('only the open, quiet ones are written off', async () => {
    const { abandoned } = await abandonStaleSessions(db)
    assert.deepEqual([...abandoned].sort(), ['ses_stale_active', 'ses_stale_created'])

    const swept = await db.collection<any>(collections.sessions).findOne({ _id: 'ses_stale_active' })
    assert.equal(swept?.state, 'abandoned')
    assert.equal(swept?.abandonReason, 'stale_no_client_activity')
    assert.ok(swept?.abandonedAt instanceof Date)
    assert.equal(swept?.stateVersion, 2, 'the state change is versioned like any other')

    const untouched = await db.collection<any>(collections.sessions).findOne({ _id: 'ses_recent_active' })
    assert.equal(untouched?.state, 'active', 'a conversation five minutes old is still a conversation')
    for (const id of ['ses_old_completed', 'ses_old_excluded']) {
      const finished = await db.collection<any>(collections.sessions).findOne({ _id: id })
      assert.notEqual(finished?.state, 'abandoned', `${id} was already resolved`)
    }
  })

  await t.test('each one emits the same event a client abandon would', async () => {
    const events = await db.collection<any>(collections.outboxEvents)
      .find({ eventType: 'session.abandoned' }).toArray()
    assert.equal(events.length, 2)
    assert.deepEqual([...new Set(events.map((event) => event.payload?.reason))], ['stale_no_client_activity'])
    assert.deepEqual([...events.map((event) => String(event.aggregateId))].sort(),
      ['ses_stale_active', 'ses_stale_created'])
    // Consumers must not have to distinguish a swept session from a client-abandoned one.
    assert.ok(events.every((event) => event.aggregateType === 'session' && event.tenantId === TENANT_ID))
  })

  await t.test('a second pass is a no-op rather than a second round of events', async () => {
    const { abandoned } = await abandonStaleSessions(db)
    assert.deepEqual(abandoned, [], 'nothing is left open to sweep')
    assert.equal(await db.collection<any>(collections.outboxEvents)
      .countDocuments({ eventType: 'session.abandoned' }), 2)
  })
})
