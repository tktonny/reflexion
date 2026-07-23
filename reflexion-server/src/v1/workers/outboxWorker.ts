import type { Db } from 'mongodb'
import { MongoServerError } from 'mongodb'
import { getDb } from '../../lib/mongo.js'
import { materializeMedicationReminders } from '../care/reminderScheduler.js'
import { processCompletedSession, verifyArtifact } from '../monitoring/pipeline.js'
import { collections } from '../platform/collections.js'
import { openSecret } from '../platform/crypto.js'
import { sendPasswordResetEmail } from '../notifications/email.js'
import { materializeReviewCaseNotifications } from '../notifications/service.js'

const CONSUMER_NAME = 'platform-v1-worker'
const MAX_ATTEMPTS = 8

export async function processNextOutboxEvent(db?: Db) {
  const database = db || await getDb()
  const now = new Date()
  const event = await database.collection<any>(collections.outboxEvents).findOneAndUpdate({
    state: { $in: ['pending', 'retry'] }, nextAttemptAt: { $lte: now },
    $or: [{ leaseUntil: { $lt: now } }, { leaseUntil: { $exists: false } }],
  }, { $set: { state: 'processing', leaseUntil: new Date(Date.now() + 60_000), updatedAt: now }, $inc: { attempt: 1 } }, {
    sort: { occurredAt: 1 }, returnDocument: 'after',
  })
  if (!event) return false
  try {
    const consumption = await beginConsumption(database, String(event._id))
    if (consumption === 'already_completed') {
      await markEventCompleted(database, String(event._id))
      return true
    }
    await handleEvent(database, event)
    await database.collection<any>(collections.eventConsumptions).updateOne({
      eventId: event._id, consumerName: CONSUMER_NAME,
    }, { $set: { state: 'completed', completedAt: new Date(), result: 'ok' } })
    await markEventCompleted(database, String(event._id))
  } catch (error) {
    const attempt = Number(event.attempt || 1)
    const dead = attempt >= MAX_ATTEMPTS
    await database.collection<any>(collections.outboxEvents).updateOne({ _id: event._id }, { $set: {
      state: dead ? 'dead_letter' : 'retry',
      nextAttemptAt: new Date(Date.now() + Math.min(2 ** attempt * 1000, 15 * 60_000)),
      lastError: error instanceof Error ? error.message.slice(0, 1000) : 'Unknown worker error',
      deadLetterReason: dead ? 'max_attempts_exceeded' : undefined,
      updatedAt: new Date(),
    }, $unset: { leaseUntil: '' } })
    await database.collection<any>(collections.eventConsumptions).updateOne({
      eventId: event._id, consumerName: CONSUMER_NAME,
    }, { $set: { state: 'failed', failedAt: new Date(), error: error instanceof Error ? error.message.slice(0, 1000) : 'Unknown worker error' } }, { upsert: true })
    if (event.eventType === 'session.completed') {
      const message = error instanceof Error ? error.message.slice(0, 1000) : 'Unknown worker error'
      await database.collection<any>(collections.sessions).updateOne({ _id: event.aggregateId }, { $set: {
        state: 'processing_failed',
        processingSummary: { state: 'failed', stage: 'processing', retryable: !dead, attempts: attempt, lastError: message },
        updatedAt: new Date(),
      } })
    }
  }
  return true
}

async function beginConsumption(db: Db, eventId: string) {
  const filter = { eventId, consumerName: CONSUMER_NAME }
  const existing = await db.collection<any>(collections.eventConsumptions).findOne(filter)
  if (existing?.state === 'completed') return 'already_completed' as const
  if (existing) {
    await db.collection<any>(collections.eventConsumptions).updateOne(filter, { $set: { state: 'processing', startedAt: new Date() }, $inc: { attempt: 1 } })
    return 'started' as const
  }
  try {
    await db.collection<any>(collections.eventConsumptions).insertOne({ ...filter, state: 'processing', attempt: 1, startedAt: new Date() })
  } catch (error) {
    if (!(error instanceof MongoServerError) || error.code !== 11000) throw error
    const raced = await db.collection<any>(collections.eventConsumptions).findOne(filter)
    if (raced?.state === 'completed') return 'already_completed' as const
  }
  return 'started' as const
}

async function handleEvent(db: Db, event: Record<string, any>) {
  switch (event.eventType) {
    case 'artifact.committed':
      await verifyArtifact(db, String(event.aggregateId), String(event.correlationId))
      break
    case 'session.completed':
      await processCompletedSession(db, String(event.aggregateId), String(event.correlationId))
      break
    case 'medication_plan.changed':
      await materializeMedicationReminders(db, String(event.aggregateId))
      break
    case 'password_reset.requested':
      await sendPasswordResetEmail({ email: String(event.payload.email), name: String(event.payload.name || ''), token: openSecret(String(event.payload.sealedToken)) })
      break
    case 'review_case.created':
      await materializeReviewCaseNotifications(db, String(event.aggregateId))
      break
    default:
      break
  }
}

async function markEventCompleted(db: Db, eventId: string) {
  await db.collection<any>(collections.outboxEvents).updateOne({ _id: eventId }, { $set: {
    state: 'completed', publishedAt: new Date(), updatedAt: new Date(),
  }, $unset: { leaseUntil: '' } })
}
