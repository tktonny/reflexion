import type { Db } from 'mongodb'
import { collections } from '../platform/collections.js'

// Delivers the durable in-app notifications (the `notifications` collection) to caregivers' phones via
// Expo Push. This is the "last hop" that was missing: notifications were materialized + fed in-app, but
// the registered Expo tokens in `notification_devices` were never read and no push ever left the server.
//
// Design: a Mongo-backed pending→sent dispatcher (NOT an inline send), mirroring our outbox philosophy —
// idempotent + retryable. materializeNotifications marks a row `pushState:'pending'` when a push is wanted;
// dispatchPendingPushes claims each row, looks up the recipient's active devices, POSTs to Expo, and flips
// the row to sent/failed/skipped so a re-run never double-pushes. It rides the existing reflexion-worker
// loop, so no additional process is required.

const EXPO_PUSH_URL = 'https://exp.host/--/api/v2/push/send'
const EXPO_TOKEN_RE = /^Expo(nent)?PushToken\[[^\]]+\]$/
const DISPATCH_BATCH = 50
const MAX_PUSH_ATTEMPTS = 5
const CLAIM_STALE_MS = 2 * 60_000

type ExpoMessage = { to: string; title: string; body: string; sound?: 'default'; data?: Record<string, unknown> }
type ExpoTicket = { status?: 'ok' | 'error'; id?: string; message?: string; details?: { error?: string } }

/** POST messages to Expo's push service in ≤100-message chunks; returns one ticket per message in order. */
export async function sendExpoPush(messages: ExpoMessage[]): Promise<ExpoTicket[]> {
  if (!messages.length) return []
  const tickets: ExpoTicket[] = []
  for (let offset = 0; offset < messages.length; offset += 100) {
    const chunk = messages.slice(offset, offset + 100)
    let payload: { data?: ExpoTicket[] } | null = null
    let ok = false
    try {
      const response = await fetch(EXPO_PUSH_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(chunk),
        signal: AbortSignal.timeout(15_000),
      })
      ok = response.ok
      payload = await response.json().catch(() => null) as { data?: ExpoTicket[] } | null
      if (!ok && !payload) payload = { data: chunk.map(() => ({ status: 'error', message: `expo_http_${response.status}` })) }
    } catch (error) {
      payload = { data: chunk.map(() => ({ status: 'error', message: error instanceof Error ? error.message.slice(0, 120) : 'expo_transport_error' })) }
    }
    const data = payload?.data
    if (Array.isArray(data) && data.length === chunk.length) tickets.push(...data)
    else for (const _ of chunk) tickets.push({ status: 'error', message: 'expo_unexpected_response' })
  }
  return tickets
}

/**
 * Drains notifications awaiting a phone push. Idempotent + retryable: only `pushState:'pending'` rows are
 * picked up, each is claimed to 'sending' (so two workers can't both push it), and then flipped to a
 * terminal state. Crash-safe: a stale 'sending' claim is reset to 'pending' first. Returns per-run counts.
 */
export async function dispatchPendingPushes(db: Db, limit = DISPATCH_BATCH): Promise<{ sent: number; failed: number; skipped: number }> {
  const notifications = db.collection<any>(collections.notifications)
  // Recover rows a previously-crashed worker left mid-send.
  await notifications.updateMany(
    { pushState: 'sending', pushClaimedAt: { $lt: new Date(Date.now() - CLAIM_STALE_MS) } },
    { $set: { pushState: 'pending', updatedAt: new Date() } },
  )
  const pending = await notifications.find({ pushState: 'pending' }).sort({ createdAt: 1 }).limit(limit).toArray()
  let sent = 0, failed = 0, skipped = 0
  for (const notif of pending) {
    const now = new Date()
    const claimed = await notifications.findOneAndUpdate(
      { _id: notif._id, pushState: 'pending' },
      { $set: { pushState: 'sending', pushClaimedAt: now, updatedAt: now } },
      { returnDocument: 'after' },
    )
    if (!claimed) continue // another worker grabbed it
    const devices = await db.collection<any>(collections.notificationDevices)
      .find({ tenantId: notif.tenantId, userId: notif.recipientUserId, state: 'active' }).toArray()
    const tokens = devices.map((device) => String(device.expoPushToken)).filter((token) => EXPO_TOKEN_RE.test(token))
    if (!tokens.length) {
      await notifications.updateOne({ _id: notif._id }, { $set: { pushState: 'skipped', pushError: 'no_active_device', pushedAt: new Date(), updatedAt: new Date() } })
      skipped++
      continue
    }
    try {
      const tickets = await sendExpoPush(tokens.map((to) => ({
        to,
        title: String(notif.title || 'Reflexion'),
        body: String(notif.body || ''),
        sound: 'default',
        data: { notificationId: notif._id, patientId: notif.patientId, type: notif.type, localDate: notif.localDate || null },
      })))
      // Deactivate tokens Expo reports as unregistered so a dead device stops being retried forever.
      const deadTokens = tokens.filter((_, index) => tickets[index]?.details?.error === 'DeviceNotRegistered')
      if (deadTokens.length) {
        await db.collection<any>(collections.notificationDevices).updateMany(
          { tenantId: notif.tenantId, expoPushToken: { $in: deadTokens } },
          { $set: { state: 'inactive', updatedAt: new Date() } },
        )
      }
      const okCount = tickets.filter((ticket) => ticket.status === 'ok').length
      await notifications.updateOne({ _id: notif._id }, { $set: {
        pushState: okCount > 0 ? 'sent' : 'failed',
        pushError: okCount > 0 ? null : (tickets.find((ticket) => ticket.status === 'error')?.details?.error || tickets.find((ticket) => ticket.status === 'error')?.message || 'expo_error'),
        pushTicketCount: tickets.length, pushOkCount: okCount, pushedAt: new Date(), updatedAt: new Date(),
      } })
      if (okCount > 0) sent++
      else failed++
    } catch (error) {
      const attempts = Number(notif.pushAttempts || 0) + 1
      // Transport failure: back to 'pending' for the next tick, or give up after MAX_PUSH_ATTEMPTS.
      await notifications.updateOne({ _id: notif._id }, { $set: {
        pushState: attempts >= MAX_PUSH_ATTEMPTS ? 'failed' : 'pending',
        pushError: error instanceof Error ? error.message.slice(0, 300) : 'push_dispatch_error',
        pushAttempts: attempts, updatedAt: new Date(),
      } })
      failed++
    }
  }
  return { sent, failed, skipped }
}
