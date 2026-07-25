import { Router } from 'express'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { getDb } from '../../lib/mongo.js'
import { getPrincipal, requireActor } from '../platform/auth.js'
import { collections } from '../platform/collections.js'
import { badRequest, forbidden, notFound } from '../platform/errors.js'
import { sendData, sendPage } from '../platform/http.js'
import { newId } from '../platform/ids.js'
import { enumValue, objectBody, optionalString, pagination, requiredString } from '../platform/validation.js'

export const notificationsRouter = Router()
const requireHuman = requireActor('human')

/**
 * The caregiver's alert feed, newest first.
 *
 * Sorted on `createdAt`, NOT on `_id`. Ids here are `notif_<random uuid hex>` (platform/ids.ts), so they
 * carry no time order at all — sorting and paginating on `_id` served the feed in an arbitrary order and
 * made "load more" return an arbitrary slice. `_id` is kept only as a tiebreaker so two notifications written
 * in the same millisecond still have a total order, which is what makes the cursor stable.
 *
 * The matching index already exists: {tenantId, recipientUserId, state, createdAt: -1}.
 */
notificationsRouter.get('/notifications', requireHuman, asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw forbidden()
  const { limit, cursor } = pagination(request.query as Record<string, unknown>)
  const filter: Record<string, any> = { tenantId: principal.tenantId, recipientUserId: principal.userId }
  const after = parseFeedCursor(cursor)
  if (after) {
    filter.$or = [
      { createdAt: { $lt: after.createdAt } },
      { createdAt: after.createdAt, _id: { $lt: after.id } },
    ]
  }
  if (request.query.state !== undefined) filter.state = enumValue(request.query.state, 'state', ['unread', 'read'] as const)
  const rows = await (await getDb()).collection<any>(collections.notifications)
    .find(filter).sort({ createdAt: -1, _id: -1 }).limit(limit + 1).toArray()
  const hasMore = rows.length > limit
  const page = rows.slice(0, limit)
  sendPage(response, page.map(serializeNotification), hasMore ? feedCursor(page[page.length - 1]) : null)
}))

/** `<createdAt ISO>|<id>` — both halves are needed to resume a (createdAt, _id) sort without gaps. */
function feedCursor(item: Record<string, any>): string {
  return `${new Date(item.createdAt).toISOString()}|${item._id}`
}

function parseFeedCursor(cursor?: string): { createdAt: Date; id: string } | null {
  if (!cursor) return null
  const separator = cursor.indexOf('|')
  if (separator === -1) return null
  const createdAt = new Date(cursor.slice(0, separator))
  const id = cursor.slice(separator + 1)
  // A malformed or stale cursor restarts from the newest page rather than 500ing or returning nothing.
  if (Number.isNaN(createdAt.getTime()) || !id) return null
  return { createdAt, id }
}

notificationsRouter.post('/notifications/:notificationId/read', requireHuman, asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw forbidden()
  const db = await getDb()
  const notification = await db.collection<any>(collections.notifications).findOneAndUpdate({
    _id: request.params.notificationId,
    tenantId: principal.tenantId,
    recipientUserId: principal.userId,
  }, { $set: { state: 'read', readAt: new Date(), updatedAt: new Date() } }, { returnDocument: 'after' })
  if (!notification) throw notFound('Notification')
  sendData(response, serializeNotification(notification))
}))

const DEVICE_PLATFORMS = ['ios', 'android', 'web', 'unknown'] as const

/**
 * Registers the phone that should receive a caregiver's push notifications.
 *
 * Deliberately does NOT require an Idempotency-Key: the Expo token is the device identity, so the write
 * is an upsert on (tenantId, expoPushToken) and is naturally idempotent — the app re-registers on every
 * sign-in and every Alerts-tab visit. Re-registering after an account switch moves the row to the new
 * user rather than leaving a stale mapping that would push one caregiver's alerts to another's phone.
 */
notificationsRouter.post('/notification-devices', requireHuman, asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw forbidden()
  const body = objectBody(request.body)
  const expoPushToken = requiredString(body, 'expoPushToken', 200)
  if (!/^Expo(nent)?PushToken\[[^\]]+\]$/.test(expoPushToken)) {
    throw badRequest('VALIDATION_FAILED', 'expoPushToken must be an Expo push token.')
  }
  const platform = 'platform' in body ? enumValue(body.platform, 'platform', DEVICE_PLATFORMS) : 'unknown'
  const appVersion = optionalString(body, 'appVersion', 40)
  const now = new Date()
  const db = await getDb()
  const device = await db.collection<any>(collections.notificationDevices).findOneAndUpdate(
    { tenantId: principal.tenantId, expoPushToken },
    {
      $set: { userId: principal.userId, platform, appVersion: appVersion || null, state: 'active', lastSeenAt: now, updatedAt: now },
      $setOnInsert: { _id: newId('ndev'), tenantId: principal.tenantId, expoPushToken, createdAt: now },
    },
    { upsert: true, returnDocument: 'after' },
  )
  sendData(response, {
    deviceId: device?._id,
    platform: device?.platform,
    state: device?.state,
    registeredAt: new Date(device?.createdAt || now).toISOString(),
  })
}))

function serializeNotification(item: Record<string, any>) {
  return {
    notificationId: item._id,
    patientId: item.patientId,
    type: item.type,
    state: item.state,
    title: item.title,
    body: item.body,
    source: item.source,
    // Local calendar day the alert is about (daily check-in alerts only) — lets a client deep-link to
    // that day's sessions without re-deriving it from the dedupe key.
    localDate: item.localDate || null,
    createdAt: new Date(item.createdAt).toISOString(),
    readAt: item.readAt ? new Date(item.readAt).toISOString() : null,
  }
}
