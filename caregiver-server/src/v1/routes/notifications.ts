import { Router } from 'express'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { getDb } from '../../lib/mongo.js'
import { getPrincipal, requireActor } from '../platform/auth.js'
import { collections } from '../platform/collections.js'
import { forbidden, notFound } from '../platform/errors.js'
import { sendData, sendPage } from '../platform/http.js'
import { enumValue, pagination } from '../platform/validation.js'

export const notificationsRouter = Router()
const requireHuman = requireActor('human')

notificationsRouter.get('/notifications', requireHuman, asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw forbidden()
  const { limit, cursor } = pagination(request.query as Record<string, unknown>)
  const filter: Record<string, any> = { tenantId: principal.tenantId, recipientUserId: principal.userId }
  if (cursor) filter._id = { $lt: cursor }
  if (request.query.state !== undefined) filter.state = enumValue(request.query.state, 'state', ['unread', 'read'] as const)
  const rows = await (await getDb()).collection<any>(collections.notifications).find(filter).sort({ _id: -1 }).limit(limit + 1).toArray()
  const hasMore = rows.length > limit
  sendPage(response, rows.slice(0, limit).map(serializeNotification), hasMore ? String(rows[limit - 1]._id) : null)
}))

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

function serializeNotification(item: Record<string, any>) {
  return {
    notificationId: item._id,
    patientId: item.patientId,
    type: item.type,
    state: item.state,
    title: item.title,
    body: item.body,
    source: item.source,
    createdAt: new Date(item.createdAt).toISOString(),
    readAt: item.readAt ? new Date(item.readAt).toISOString() : null,
  }
}
