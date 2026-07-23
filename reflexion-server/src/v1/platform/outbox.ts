import type { ClientSession, Db } from 'mongodb'
import { collections } from './collections.js'
import { newId } from './ids.js'

export type OutboxInput = {
  eventType: string
  tenantId: string
  patientId?: string
  aggregateType: string
  aggregateId: string
  correlationId: string
  causationId?: string
  payload?: Record<string, unknown>
}

export async function appendOutbox(db: Db, input: OutboxInput, session?: ClientSession) {
  const event = {
    _id: newId('evt'),
    eventType: input.eventType,
    eventVersion: 1,
    occurredAt: new Date(),
    tenantId: input.tenantId,
    patientId: input.patientId,
    aggregateType: input.aggregateType,
    aggregateId: input.aggregateId,
    correlationId: input.correlationId,
    causationId: input.causationId,
    payload: input.payload || {},
    state: 'pending',
    attempt: 0,
    nextAttemptAt: new Date(),
    createdAt: new Date(),
  }
  await db.collection<any>(collections.outboxEvents).insertOne(event, { session })
  return event
}
