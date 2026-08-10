import assert from 'node:assert/strict'
import test from 'node:test'
import { MongoMemoryReplSet } from 'mongodb-memory-server'

import { closeMongo, getDb } from '../../lib/mongo.js'
import { collections } from '../platform/collections.js'

/*
 * processCompletedSession had no test at all, which is how a bug survived in it that only appears the
 * SECOND time a session is processed.
 *
 * The pipeline normally sees a session once, so nothing exercised re-processing until a change to the
 * quality thresholds made it worth re-running the ones the old bar had discarded. Every one of them then
 * failed eight times over and landed in the dead-letter queue: upsertMonitoringWindow filters on
 * (tenant, patient, windowEnd, ruleVersion) with no revision, so the second pass matches the window the
 * first pass wrote, and `$set` carrying a newly minted `_id` makes MongoDB refuse the write.
 *
 * The window upsert is on BOTH paths out of the gate check — the excluded branch and the full pipeline — so
 * an excluded session is enough to reproduce it, and needs no artifacts, consent or device fixture.
 */

const TENANT_ID = 'ten_reprocess'
const PATIENT_ID = 'pat_reprocess'
const SESSION_ID = 'ses_reprocess'

test('a session can be processed twice, which is what a threshold change requires', async (t) => {
  const replicaSet = await MongoMemoryReplSet.create({ replSet: { count: 1, storageEngine: 'wiredTiger' } })
  const originalEnvironment = { ...process.env }
  process.env.NODE_ENV = 'test'
  process.env.MONGODB_URI = replicaSet.getUri()
  process.env.JWT_SECRET = 'reprocess-jwt-secret-at-least-32-characters'
  process.env.PAIRING_PEPPER = 'reprocess-pairing-pepper-at-least-32-chars'
  process.env.CREDENTIAL_ENCRYPTION_KEY = 'reprocess-encryption-key-at-least-32-chars'

  const { processCompletedSession } = await import('./pipeline.js')
  const db = await getDb()

  t.after(async () => {
    await closeMongo()
    await replicaSet.stop()
    process.env = originalEnvironment
  })

  const completedAt = new Date('2026-07-26T10:00:00.000Z')
  await db.collection<any>(collections.sessions).insertOne({
    _id: SESSION_ID, tenantId: TENANT_ID, patientId: PATIENT_ID, type: 'companion',
    state: 'ingesting', stateVersion: 1, artifactIds: [], latestProcessingRevision: 0,
    localCompletedAt: completedAt, acquisition: {}, createdAt: completedAt, updatedAt: completedAt,
  })

  /** What the repair script does: the pipeline refuses to touch a session already in a terminal state. */
  const requeue = () => db.collection<any>(collections.sessions).updateOne(
    { _id: SESSION_ID }, { $set: { state: 'ingesting', updatedAt: new Date() } })

  await t.test('the first pass writes a monitoring window', async () => {
    await processCompletedSession(db, SESSION_ID, 'corr-first')
    const windows = await db.collection<any>(collections.monitoringWindows).find({ patientId: PATIENT_ID }).toArray()
    assert.equal(windows.length, 1)
    assert.equal(windows[0].sessionId, SESSION_ID)
    const session = await db.collection<any>(collections.sessions).findOne({ _id: SESSION_ID })
    assert.equal(session?.state, 'excluded', 'no device assignment, so the identity gate excludes it')
    assert.equal(session?.latestProcessingRevision, 1)
  })

  await t.test('the second pass does not throw on the immutable _id', async () => {
    await requeue()
    // Before the fix this rejected with "Performing an update on the path '_id' would modify the
    // immutable field '_id'", and the worker retried it to death.
    await processCompletedSession(db, SESSION_ID, 'corr-second')

    const session = await db.collection<any>(collections.sessions).findOne({ _id: SESSION_ID })
    assert.notEqual(session?.state, 'processing_failed', 'the retry must not be what kills the session')
    assert.equal(session?.latestProcessingRevision, 2, 'each pass is its own revision')
  })

  await t.test('the window is updated in place rather than duplicated', async () => {
    const windows = await db.collection<any>(collections.monitoringWindows).find({ patientId: PATIENT_ID }).toArray()
    assert.equal(windows.length, 1, 'the unique index has no revision in it, so there is one window per day')
    // The id is the one the first pass minted: it is the row's identity, not part of the payload.
    assert.match(String(windows[0]._id), /^win_/)
  })

  await t.test('a third pass is still fine, so a repeated repair run is safe', async () => {
    await requeue()
    await processCompletedSession(db, SESSION_ID, 'corr-third')
    const session = await db.collection<any>(collections.sessions).findOne({ _id: SESSION_ID })
    assert.notEqual(session?.state, 'processing_failed')
    assert.equal(await db.collection<any>(collections.monitoringWindows).countDocuments({ patientId: PATIENT_ID }), 1)
  })
  await t.test('unpairing the mirror does not retroactively invalidate past sessions', async () => {
    // The identity gate asked for an assignment active NOW, so every session a loved one had ever recorded
    // turned into `exclude` the next time it was processed after their mirror was unpaired — and re-pairing
    // is a normal thing to do. Production had 19 revoked/replaced assignments for one device and not a
    // single active one, which is how five real check-ins became unanalysable.
    const capturedAt = new Date('2026-07-20T02:00:00.000Z')
    await db.collection<any>(collections.sessions).insertOne({
      _id: 'ses_after_unpair', tenantId: TENANT_ID, patientId: PATIENT_ID, type: 'companion',
      state: 'ingesting', stateVersion: 1, artifactIds: [], latestProcessingRevision: 0,
      deviceId: 'dev_unpaired', localCompletedAt: capturedAt, acquisition: {},
      createdAt: capturedAt, updatedAt: capturedAt,
    })
    // Covered the capture instant, revoked two days later.
    await db.collection<any>(collections.assignments).insertOne({
      _id: 'asg_revoked_later', tenantId: TENANT_ID, patientId: PATIENT_ID, deviceId: 'dev_unpaired',
      assignmentType: 'primary', status: 'revoked',
      assignedAt: new Date('2026-07-18T00:00:00.000Z'), revokedAt: new Date('2026-07-22T00:00:00.000Z'),
    })

    await processCompletedSession(db, 'ses_after_unpair', 'corr-unpair')
    const link = await db.collection<any>(collections.identityLinks).findOne({ sessionId: 'ses_after_unpair' })
    assert.equal(link?.verdict, 'linked', 'the mirror was assigned when this conversation happened')
    assert.deepEqual(link?.reasons, ['DEVICE_ASSIGNMENT_ACTIVE_AT_CAPTURE_NOT_BIOMETRIC'],
      'and the reason says it was historical rather than live')
  })

  await t.test('an assignment that only began after the session still does not count', async () => {
    const capturedAt = new Date('2026-07-10T02:00:00.000Z')
    await db.collection<any>(collections.sessions).insertOne({
      _id: 'ses_before_pairing', tenantId: TENANT_ID, patientId: PATIENT_ID, type: 'companion',
      state: 'ingesting', stateVersion: 1, artifactIds: [], latestProcessingRevision: 0,
      deviceId: 'dev_unpaired', localCompletedAt: capturedAt, acquisition: {},
      createdAt: capturedAt, updatedAt: capturedAt,
    })
    await processCompletedSession(db, 'ses_before_pairing', 'corr-before')
    const link = await db.collection<any>(collections.identityLinks).findOne({ sessionId: 'ses_before_pairing' })
    assert.equal(link?.verdict, 'exclude', 'the assignment began eight days after this was captured')
    assert.deepEqual(link?.reasons, ['DEVICE_ASSIGNMENT_INVALID'])
  })
})
