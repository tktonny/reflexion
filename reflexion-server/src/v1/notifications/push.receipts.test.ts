import assert from 'node:assert/strict'
import test from 'node:test'
import { MongoMemoryReplSet } from 'mongodb-memory-server'

import { closeMongo, getDb } from '../../lib/mongo.js'
import { collections } from '../platform/collections.js'

/*
 * Push had no delivery verification at all, and that is not a gap you notice from the database — it is a
 * gap that makes the database confidently wrong.
 *
 * dispatchPendingPushes marked a notification `sent` as soon as Expo answered /send with an ok ticket. An
 * ok ticket means only that Expo accepted the message and queued it; whether FCM took it is reported
 * later, in a receipt keyed by the ticket id. The ticket ids were thrown away. So the single most common
 * production failure — an Expo project with no FCM V1 service-account key, where Expo cannot reach your
 * Firebase project at all — produced an ok ticket, a row reading `sent`, a receipt of InvalidCredentials
 * nobody ever asked for, and a caregiver whose phone never rang. Every push in the system looked fine.
 *
 * These tests pin the behaviour that makes `sent` falsifiable: ok receipt -> delivered, error receipt ->
 * undelivered WITH the error preserved, an unfetchable receipt left alone rather than blamed on the phone,
 * and DeviceNotRegistered retiring the token so a reinstalled phone stops being pushed to forever.
 */

const TENANT_ID = 'ten_receipts'
const USER_ID = 'usr_receipts'

type FetchStub = { calls: unknown[][]; restore: () => void }

/** Replaces global fetch with one canned Expo /getReceipts body. No network in this suite. */
function stubReceipts(body: unknown, options: { throws?: boolean } = {}): FetchStub {
  const original = globalThis.fetch
  const calls: unknown[][] = []
  globalThis.fetch = (async (...args: unknown[]) => {
    calls.push(args)
    if (options.throws) throw new Error('socket hang up')
    return { ok: true, status: 200, json: async () => body } as unknown as Response
  }) as unknown as typeof fetch
  return { calls, restore: () => { globalThis.fetch = original } }
}

async function seedSent(db: Awaited<ReturnType<typeof getDb>>, id: string, ticketIds: string[], pushedMinutesAgo = 5) {
  await db.collection<any>(collections.notifications).insertOne({
    _id: id, tenantId: TENANT_ID, recipientUserId: USER_ID, type: 'completion',
    title: 't', body: 'b', state: 'unread',
    pushState: 'sent', pushTicketIds: ticketIds,
    pushedAt: new Date(Date.now() - pushedMinutesAgo * 60_000),
    createdAt: new Date(), updatedAt: new Date(),
  })
}

test('push receipts resolve `sent` into the honest delivery outcome', async (t) => {
  const replicaSet = await MongoMemoryReplSet.create({ replSet: { count: 1, storageEngine: 'wiredTiger' } })
  const originalEnvironment = { ...process.env }
  process.env.NODE_ENV = 'test'
  process.env.MONGODB_URI = replicaSet.getUri()
  process.env.JWT_SECRET = 'receipts-jwt-secret-at-least-32-characters'
  process.env.PAIRING_PEPPER = 'receipts-pairing-pepper-at-least-32-chars'
  process.env.CREDENTIAL_ENCRYPTION_KEY = 'receipts-encryption-key-at-least-32-chars'

  const { pollPushReceipts } = await import('./push.js')
  const db = await getDb()

  t.after(async () => {
    await closeMongo()
    await replicaSet.stop()
    process.env = originalEnvironment
  })

  const notifications = db.collection<any>(collections.notifications)
  const devices = db.collection<any>(collections.notificationDevices)
  const reset = async () => {
    await notifications.deleteMany({})
    await devices.deleteMany({})
  }

  await t.test('an ok receipt marks the notification delivered', async () => {
    await reset()
    await seedSent(db, 'ntf_ok', ['tik_1'])
    const stub = stubReceipts({ data: { tik_1: { status: 'ok' } } })
    try {
      const result = await pollPushReceipts(db)
      assert.equal(result.delivered, 1)
      assert.equal(result.undelivered, 0)
    } finally { stub.restore() }

    const row = await notifications.findOne({ _id: 'ntf_ok' } as any)
    assert.equal(row?.pushState, 'delivered')
    assert.equal(row?.pushError, null)
    assert.ok(row?.pushReceiptAt instanceof Date)
  })

  await t.test('InvalidCredentials — the FCM-key failure — is recorded, not read as success', async () => {
    await reset()
    await seedSent(db, 'ntf_bad_creds', ['tik_2'])
    const stub = stubReceipts({
      data: { tik_2: { status: 'error', message: 'Unable to retrieve the FCM server key', details: { error: 'InvalidCredentials' } } },
    })
    try {
      const result = await pollPushReceipts(db)
      assert.equal(result.undelivered, 1)
      assert.equal(result.delivered, 0)
    } finally { stub.restore() }

    const row = await notifications.findOne({ _id: 'ntf_bad_creds' } as any)
    // The whole point: this row must NOT read `sent`/`delivered`. That is the bug this suite exists for.
    assert.equal(row?.pushState, 'undelivered')
    assert.equal(row?.pushError, 'InvalidCredentials')
  })

  await t.test('DeviceNotRegistered retires the token so a reinstalled phone stops being pushed to', async () => {
    await reset()
    await seedSent(db, 'ntf_dead', ['tik_3'])
    await devices.insertOne({
      _id: 'nd_1', tenantId: TENANT_ID, userId: USER_ID, state: 'active',
      platform: 'android', expoPushToken: 'ExponentPushToken[dead]', updatedAt: new Date(),
    })
    const stub = stubReceipts({ data: { tik_3: { status: 'error', details: { error: 'DeviceNotRegistered' } } } })
    try { await pollPushReceipts(db) } finally { stub.restore() }

    assert.equal((await devices.findOne({ _id: 'nd_1' } as any))?.state, 'inactive')
  })

  await t.test('one delivered device is a delivered alert, even if another failed', async () => {
    await reset()
    await seedSent(db, 'ntf_mixed', ['tik_4', 'tik_5'])
    const stub = stubReceipts({
      data: { tik_4: { status: 'error', details: { error: 'MessageTooBig' } }, tik_5: { status: 'ok' } },
    })
    try { await pollPushReceipts(db) } finally { stub.restore() }

    assert.equal((await notifications.findOne({ _id: 'ntf_mixed' } as any))?.pushState, 'delivered')
  })

  await t.test('a receipt we could not fetch is left pending, never blamed on the phone', async () => {
    await reset()
    await seedSent(db, 'ntf_net', ['tik_6'])
    await devices.insertOne({
      _id: 'nd_2', tenantId: TENANT_ID, userId: USER_ID, state: 'active',
      platform: 'android', expoPushToken: 'ExponentPushToken[live]', updatedAt: new Date(),
    })
    const stub = stubReceipts(null, { throws: true })
    try {
      const result = await pollPushReceipts(db)
      assert.equal(result.pending, 1)
      assert.equal(result.undelivered, 0)
    } finally { stub.restore() }

    // Still `sent`, so the next tick retries it — and our own network failure has not deactivated a live device.
    assert.equal((await notifications.findOne({ _id: 'ntf_net' } as any))?.pushState, 'sent')
    assert.equal((await devices.findOne({ _id: 'nd_2' } as any))?.state, 'active')
  })

  await t.test('a push too recent to have a receipt yet is not touched', async () => {
    await reset()
    await seedSent(db, 'ntf_fresh', ['tik_7'], 0)
    const stub = stubReceipts({ data: {} })
    try {
      const result = await pollPushReceipts(db)
      assert.deepEqual(result, { delivered: 0, undelivered: 0, pending: 0 })
      assert.equal(stub.calls.length, 0, 'must not call Expo when nothing is eligible')
    } finally { stub.restore() }
  })

  await t.test('an unresolved ticket is eventually accepted rather than pursued forever', async () => {
    await reset()
    await seedSent(db, 'ntf_stale', ['tik_8'], 7 * 60)
    const stub = stubReceipts({ data: {} })
    try { await pollPushReceipts(db) } finally { stub.restore() }

    assert.equal((await notifications.findOne({ _id: 'ntf_stale' } as any))?.pushState, 'delivered')
  })

  await t.test('rows without ticket ids are ignored, so pre-fix history is not misread', async () => {
    await reset()
    // Everything pushed before ticket ids were kept. We cannot know their outcome, so we must not guess.
    await notifications.insertOne({
      _id: 'ntf_legacy', tenantId: TENANT_ID, recipientUserId: USER_ID, type: 'completion',
      pushState: 'sent', pushedAt: new Date(Date.now() - 60_000), createdAt: new Date(),
    } as any)
    const stub = stubReceipts({ data: {} })
    try {
      const result = await pollPushReceipts(db)
      assert.deepEqual(result, { delivered: 0, undelivered: 0, pending: 0 })
    } finally { stub.restore() }

    assert.equal((await notifications.findOne({ _id: 'ntf_legacy' } as any))?.pushState, 'sent')
  })
})
