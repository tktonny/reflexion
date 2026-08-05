import { Router } from 'express'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { getDb } from '../../lib/mongo.js'
import { requireActor, getPrincipal } from '../platform/auth.js'
import { collections } from '../platform/collections.js'
import { conflict, badRequest, unauthorized } from '../platform/errors.js'
import { sendData } from '../platform/http.js'
import { executeIdempotent } from '../platform/idempotency.js'
import { objectBody, enumValue, positiveInteger } from '../platform/validation.js'

export const setupProgressRouter = Router()

export const SETUP_PROGRESS_CATEGORIES = [
  'household',
  'pair-device',
  'language-accessibility',
  'routines',
  'notifications',
  'consent-control',
  'care-circle',
  'research-participation',
] as const

export type SetupProgressCategory = typeof SETUP_PROGRESS_CATEGORIES[number]
export type SetupProgressStatus = 'not-started' | 'in-progress' | 'complete' | 'skipped'

const SETUP_PROGRESS_STATUSES = ['not-started', 'in-progress', 'complete', 'skipped'] as const

function defaultCategories(): Record<SetupProgressCategory, SetupProgressStatus> {
  return Object.fromEntries(SETUP_PROGRESS_CATEGORIES.map((category) => [category, 'not-started'])) as Record<SetupProgressCategory, SetupProgressStatus>
}

function progressId(userId: string) {
  // Deterministic identity makes concurrent cold-start GETs safe even before the deployment index has been
  // created, and means setup progress can never be duplicated for one caregiver.
  return `setp_${userId}`
}

async function ensureProgress(tenantId: string, userId: string) {
  const db = await getDb()
  const now = new Date()
  const row = await db.collection<any>(collections.setupProgress).findOneAndUpdate(
    { tenantId, userId },
    { $setOnInsert: {
      _id: progressId(userId), tenantId, userId, categories: defaultCategories(), version: 1,
      createdAt: now, updatedAt: now,
    } },
    { upsert: true, returnDocument: 'after' },
  )
  if (!row) throw new Error('Unable to initialise setup progress.')
  return row
}

function serializeProgress(row: Record<string, any>) {
  const categories = defaultCategories()
  for (const category of SETUP_PROGRESS_CATEGORIES) {
    const value = row.categories?.[category]
    if (SETUP_PROGRESS_STATUSES.includes(value)) categories[category] = value
  }
  const completeCount = SETUP_PROGRESS_CATEGORIES.filter((category) => categories[category] === 'complete').length
  const complete = SETUP_PROGRESS_CATEGORIES.every((category) => categories[category] === 'complete' || categories[category] === 'skipped')
  return {
    setupProgressId: row._id,
    userId: row.userId,
    categories,
    completeCount,
    total: SETUP_PROGRESS_CATEGORIES.length,
    state: complete ? 'complete' : 'in-progress',
    version: Number(row.version || 1),
    completedAt: row.completedAt ? new Date(row.completedAt).toISOString() : null,
  }
}

function parseIfMatch(value: string | undefined) {
  if (!value) throw badRequest('IF_MATCH_REQUIRED', 'If-Match is required for setup progress changes.')
  const normalized = value.trim().replace(/^W\//, '').replace(/^"|"$/g, '')
  const parsed = Number(normalized)
  return positiveInteger(parsed, 'If-Match', true)
}

setupProgressRouter.get('/setup-progress', requireActor('human'), asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw unauthorized()
  const row = await ensureProgress(principal.tenantId, principal.userId)
  sendData(response, serializeProgress(row))
}))

setupProgressRouter.patch('/setup-progress', requireActor('human'), asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw unauthorized()
  const body = objectBody(request.body)
  const category = enumValue(body.category, 'category', SETUP_PROGRESS_CATEGORIES)
  const status = enumValue(body.status, 'status', SETUP_PROGRESS_STATUSES)
  const expectedVersion = parseIfMatch(request.header('If-Match'))

  const result = await executeIdempotent(request, 'setup-progress.patch', async () => {
    const db = await getDb()
    const current = await ensureProgress(principal.tenantId, principal.userId)
    const currentStatus = current.categories?.[category] || 'not-started'
    if (Number(current.version || 1) !== expectedVersion) {
      throw conflict('VERSION_CONFLICT', 'Setup progress changed on another device. Refresh and try again.')
    }
    if (currentStatus === status) return { status: 200, data: serializeProgress(current) }
    const now = new Date()
    const next = await db.collection<any>(collections.setupProgress).findOneAndUpdate(
      { _id: current._id, tenantId: principal.tenantId, userId: principal.userId, version: expectedVersion },
      { $set: { [`categories.${category}`]: status, updatedAt: now }, $inc: { version: 1 } },
      { returnDocument: 'after' },
    )
    if (!next) throw conflict('VERSION_CONFLICT', 'Setup progress changed on another device. Refresh and try again.')
    const serialized = serializeProgress(next)
    if (serialized.state === 'complete' && !next.completedAt) {
      await db.collection<any>(collections.setupProgress).updateOne({ _id: next._id, completedAt: { $exists: false } }, { $set: { completedAt: now } })
      serialized.completedAt = now.toISOString()
    }
    return { status: 200, data: serialized }
  })
  sendData(response, result.data, result.status)
}))

