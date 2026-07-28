import { Router } from 'express'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { getDb } from '../../lib/mongo.js'
import { getPrincipal, requireActor } from '../platform/auth.js'
import { collections } from '../platform/collections.js'
import { badRequest, forbidden, notFound } from '../platform/errors.js'
import { sendData, sendPage } from '../platform/http.js'
import { newId } from '../platform/ids.js'
import { enumValue, objectBody, optionalString, pagination, requiredString } from '../platform/validation.js'
import { ANDROID_CHANNEL_ID, EXPO_TOKEN_RE, fetchExpoReceipts, sendExpoPush } from '../notifications/push.js'

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

/**
 * Sends one real push to the caller's own registered phones and reports what actually happened to it.
 *
 * This exists because "is push working?" was unanswerable from the product. A caregiver could see the app
 * say "This phone is registered" and still never get an alert, and the only way to find out why was to SSH
 * into the server, read the Expo token out of Mongo and hand-poll Expo's receipts — which is how the
 * InvalidCredentials class of failure (Expo accepts the message, then cannot reach Firebase for want of an
 * FCM V1 key) stayed invisible. Registration proves the phone can be addressed; only a delivered push
 * proves the chain works, so the app needs a way to ask for one.
 *
 * No Idempotency-Key, matching /notification-devices above: sending a duplicate test alert on a double-tap
 * is harmless, and refusing the second one would be worse — a caregiver who tapped twice deserves an answer
 * both times. It writes nothing to `notifications`, so a test never appears in the real alert feed.
 */
notificationsRouter.post('/notification-devices/test', requireHuman, asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw forbidden()
  const db = await getDb()
  const devices = await db.collection<any>(collections.notificationDevices)
    .find({ tenantId: principal.tenantId, userId: principal.userId, state: 'active' }).toArray()
  const tokens = devices.map((device) => String(device.expoPushToken)).filter((token) => EXPO_TOKEN_RE.test(token))
  if (!tokens.length) {
    sendData(response, { outcome: 'no_registered_phone', devices: 0, delivered: 0, detail: null })
    return
  }

  const tickets = await sendExpoPush(tokens.map((to) => ({
    to,
    title: 'Reflexion test alert',
    body: 'Push notifications are working on this phone.',
    sound: 'default' as const,
    data: { type: 'test' },
    channelId: ANDROID_CHANNEL_ID,
    priority: 'high' as const,
  })))

  const rejected = tickets.find((ticket) => ticket.status !== 'ok')
  const ticketIds = tickets.filter((ticket) => ticket.status === 'ok' && ticket.id).map((ticket) => String(ticket.id))
  if (!ticketIds.length) {
    sendData(response, {
      outcome: 'rejected', devices: tokens.length, delivered: 0,
      detail: rejected?.details?.error || rejected?.message || 'expo_error',
    })
    return
  }

  // Expo needs a moment before a receipt exists. One short wait beats making the app poll: the caregiver is
  // standing there watching their phone, and an answer that arrives after they walk away is not an answer.
  await new Promise((resolve) => setTimeout(resolve, 3_000))
  const receipts = await fetchExpoReceipts(ticketIds)
  const resolved = ticketIds.map((id) => receipts[id]).filter(Boolean)
  const failure = resolved.find((receipt) => receipt.status === 'error')

  if (resolved.some((receipt) => receipt.status === 'ok')) {
    sendData(response, { outcome: 'delivered', devices: tokens.length, delivered: 1, detail: null })
    return
  }
  if (failure) {
    sendData(response, {
      outcome: 'undelivered', devices: tokens.length, delivered: 0,
      detail: failure.details?.error || failure.message || 'receipt_error',
    })
    return
  }
  // Accepted by Expo but not yet confirmed. The worker's receipt pass will settle it either way.
  sendData(response, { outcome: 'accepted', devices: tokens.length, delivered: 0, detail: null })
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
