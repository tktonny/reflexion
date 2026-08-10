import type { Request } from 'express'
import { MongoServerError } from 'mongodb'
import { getDb } from '../../lib/mongo.js'
import { collections } from './collections.js'
import { conflict, badRequest } from './errors.js'
import { newId } from './ids.js'
import { getPrincipal } from './auth.js'
import { sha256 } from './crypto.js'

export type IdempotentResult<T> = { status: number; data: T }
export type IdempotencyCodec<T> = { encode: (value: T) => unknown; decode: (value: unknown) => T }

export async function executeIdempotent<T>(
  request: Request,
  routeKey: string,
  execute: () => Promise<IdempotentResult<T>>,
  explicitActorKey?: string,
  codec?: IdempotencyCodec<T>,
): Promise<IdempotentResult<T>> {
  const key = request.header('Idempotency-Key')?.trim()
  if (!key || key.length < 16 || key.length > 128) {
    throw badRequest('IDEMPOTENCY_KEY_REQUIRED', 'Idempotency-Key must contain 16 to 128 characters.')
  }
  const principal = explicitActorKey ? undefined : getPrincipal(request)
  const actorKey = explicitActorKey || `${principal!.kind}:${principal!.subjectId}`
  const requestHash = sha256(JSON.stringify(request.body ?? null))
  const db = await getDb()
  const records = db.collection<any>(collections.idempotencyRecords)
  const filter = { actorKey, routeKey, key }
  const existing = await records.findOne(filter)
  if (existing) return replay(existing, requestHash, codec)

  const record = {
    _id: newId('idem'), actorKey, routeKey, key, requestHash, state: 'processing',
    createdAt: new Date(), expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000),
  }
  try {
    await records.insertOne(record)
  } catch (error) {
    if (!(error instanceof MongoServerError) || error.code !== 11000) throw error
    const raced = await records.findOne(filter)
    if (!raced) throw error
    return replay(raced, requestHash, codec)
  }

  try {
    const result = await execute()
    await records.updateOne({ _id: record._id }, { $set: {
      state: 'completed', status: result.status, responseData: codec ? codec.encode(result.data) : result.data, completedAt: new Date(),
    } })
    return result
  } catch (error) {
    await records.deleteOne({ _id: record._id, state: 'processing' })
    throw error
  }
}

function replay<T>(record: Record<string, unknown>, requestHash: string, codec?: IdempotencyCodec<T>): IdempotentResult<T> {
  if (record.requestHash !== requestHash) {
    throw conflict('IDEMPOTENCY_KEY_REUSED', 'This idempotency key was already used with a different request.')
  }
  if (record.state !== 'completed') {
    throw conflict('REQUEST_IN_PROGRESS', 'An identical request is still being processed.', true)
  }
  return { status: Number(record.status || 200), data: codec ? codec.decode(record.responseData) : record.responseData as T }
}
