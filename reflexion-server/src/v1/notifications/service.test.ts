import assert from 'node:assert/strict'
import test from 'node:test'
import type { Db } from 'mongodb'

import { collections } from '../platform/collections.js'
import {
  dailyCheckNotificationCopy,
  materializeNotifications,
  notificationRecipients,
} from './service.js'

// Minimal in-memory stand-in for the two Mongo operations these helpers use: a filtered `find` over
// care_relationships and an upsert into notifications. Keeps the suite fast (no mongodb-memory-server)
// while still exercising the recipient filter and the dedupe key that the unique index enforces.
type Store = Record<string, Record<string, unknown>[]>

function matches(row: Record<string, any>, filter: Record<string, any>): boolean {
  return Object.entries(filter).every(([key, expected]) => {
    if (key === '$or') return (expected as Record<string, any>[]).some((clause) => matches(row, clause))
    const actual = row[key]
    if (expected && typeof expected === 'object' && !(expected instanceof Date)) {
      const condition = expected as Record<string, any>
      if ('$exists' in condition) return (actual !== undefined) === condition.$exists
      if ('$gt' in condition) return actual != null && actual > condition.$gt
      return false
    }
    // Mongo treats `field: value` on an array field as "array contains value" (used for `scopes`).
    if (Array.isArray(actual)) return actual.includes(expected)
    return actual === expected
  })
}

function fakeDb(store: Store): Db {
  return {
    collection(name: string) {
      store[name] = store[name] || []
      const rows = store[name]
      return {
        find(filter: Record<string, any>) {
          const found = rows.filter((row) => matches(row, filter))
          return { project: () => ({ toArray: async () => found }), toArray: async () => found }
        },
        async updateOne(filter: Record<string, any>, update: Record<string, any>, options?: { upsert?: boolean }) {
          const existing = rows.find((row) => matches(row, filter))
          if (existing) return { upsertedCount: 0, matchedCount: 1 }
          if (!options?.upsert) return { upsertedCount: 0, matchedCount: 0 }
          rows.push({ ...filter, ...(update.$setOnInsert || {}) })
          return { upsertedCount: 1, matchedCount: 0 }
        },
      }
    },
  } as unknown as Db
}

const TENANT = 'ten_1'
const PATIENT = 'pat_1'

function relationship(userId: string, overrides: Record<string, unknown> = {}) {
  return {
    tenantId: TENANT, patientId: PATIENT, userId, status: 'active',
    scopes: ['patient:read', 'monitoring:read'], validTo: null, ...overrides,
  }
}

test('recipients are every active caregiver holding monitoring:read, deduplicated', async () => {
  const store: Store = {
    [collections.careRelationships]: [
      relationship('usr_a'),
      relationship('usr_a', { _id: 'second-relationship-same-user' }),
      relationship('usr_b', { validTo: new Date(Date.now() + 86_400_000) }),
      relationship('usr_c', { validTo: undefined }),
    ],
  }
  const recipients = await notificationRecipients(fakeDb(store), TENANT, PATIENT)
  assert.deepEqual(recipients.sort(), ['usr_a', 'usr_b', 'usr_c'])
})

test('recipients exclude revoked, expired, wrong-scope and other-patient relationships', async () => {
  const store: Store = {
    [collections.careRelationships]: [
      relationship('usr_revoked', { status: 'revoked' }),
      relationship('usr_expired', { validTo: new Date(Date.now() - 86_400_000) }),
      relationship('usr_no_scope', { scopes: ['patient:read'] }),
      relationship('usr_other_patient', { patientId: 'pat_2' }),
      relationship('usr_other_tenant', { tenantId: 'ten_2' }),
    ],
  }
  assert.deepEqual(await notificationRecipients(fakeDb(store), TENANT, PATIENT), [])
})

test('a notification is materialized once per recipient and re-runs are no-ops', async () => {
  const store: Store = {}
  const db = fakeDb(store)
  const input = {
    tenantId: TENANT,
    patientId: PATIENT,
    recipientUserIds: ['usr_a', 'usr_b'],
    type: 'missed_7pm',
    title: 'No check-in yet today',
    body: 'Mei has not had a check-in yet today.',
    dedupeKey: `${PATIENT}:2026-07-25:missed_7pm`,
    source: { type: 'daily_check', id: `${PATIENT}:2026-07-25:missed_7pm` },
  }
  assert.equal(await materializeNotifications(db, input), 2)
  assert.equal(await materializeNotifications(db, input), 0, 'a second pass must not duplicate')

  const rows = store[collections.notifications] as Record<string, any>[]
  assert.equal(rows.length, 2)
  // Every row must carry the fields GET /notifications filters and serializes on — a row missing
  // recipientUserId is invisible to every client, which is exactly how the daily feed used to break.
  for (const row of rows) {
    assert.ok(row.recipientUserId, 'recipientUserId is required for the read model to find the row')
    assert.equal(row.state, 'unread')
    assert.equal(row.tenantId, TENANT)
    assert.equal(row.patientId, PATIENT)
    assert.ok(row.title && row.body, 'the caregiver-facing copy is rendered at write time')
    assert.ok(row.createdAt instanceof Date)
  }
  assert.deepEqual(rows.map((row) => row.recipientUserId).sort(), ['usr_a', 'usr_b'])
})

test('extra analytics fields ride along without displacing the caregiver-facing copy', async () => {
  const store: Store = {}
  await materializeNotifications(fakeDb(store), {
    tenantId: TENANT, patientId: PATIENT, recipientUserIds: ['usr_a'], type: 'technical_issue',
    title: 'The mirror may be offline', body: 'Connection issue.',
    dedupeKey: 'k', source: { type: 'daily_check', id: 'k' },
    extra: { statusAtSend: 'amber', reason: 'MIRROR_OFFLINE_OR_UNREACHABLE', localDate: '2026-07-25' },
  })
  const row = (store[collections.notifications] as Record<string, any>[])[0]
  assert.equal(row.reason, 'MIRROR_OFFLINE_OR_UNREACHABLE')
  assert.equal(row.statusAtSend, 'amber')
  assert.equal(row.title, 'The mirror may be offline')
})

test('device trouble reads as a connection issue, never as a change in the person', () => {
  const copy = dailyCheckNotificationCopy('technical_issue', 'Mei Ling Tan')
  assert.equal(copy.title, 'The mirror may be offline')
  assert.match(copy.body, /device connection issue/)
  assert.match(copy.body, /not a change in how they are doing/)
})

test('copy uses the first name and stays non-clinical across every type', () => {
  const types = ['completion', 'late_completion', 'red_missed_streak', 'technical_issue', 'missed_7pm'] as const
  for (const type of types) {
    const copy = dailyCheckNotificationCopy(type, 'Mei Ling Tan')
    assert.ok(copy.title.length && copy.body.length, `${type} needs both a title and a body`)
    assert.doesNotMatch(
      `${copy.title} ${copy.body}`,
      /cognitive|dementia|decline|impair|diagnos|score|risk/i,
      `${type} copy must not read clinically`,
    )
    assert.doesNotMatch(copy.title, /Ling|Tan/, `${type} should address them by first name only`)
  }
})

test('copy degrades to a neutral subject when the display name is missing', () => {
  const copy = dailyCheckNotificationCopy('missed_7pm', '   ')
  assert.equal(copy.title, 'No check-in yet today')
  assert.equal(copy.body, 'Your loved one has not had a check-in yet today.')
})
