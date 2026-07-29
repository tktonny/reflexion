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
const EXPO_RECEIPTS_URL = 'https://exp.host/--/api/v2/push/getReceipts'
export const EXPO_TOKEN_RE = /^Expo(nent)?PushToken\[[^\]]+\]$/
const DISPATCH_BATCH = 50
const MAX_PUSH_ATTEMPTS = 5
const CLAIM_STALE_MS = 2 * 60_000

/**
 * The Android channel the caregiver app creates at AndroidImportance.MAX (see the app's
 * configureAndroidChannel). WITHOUT this on the message, Expo delivers to its own fallback channel —
 * one the app never configured, at default importance — so the alert lands with no heads-up banner and
 * no sound, or on a China ROM not at all. Renaming it here without renaming it in the app silently
 * reintroduces that, so the two strings are a pair.
 */
export const ANDROID_CHANNEL_ID = 'reflexion-caregiver'

/** How long to let Expo sit on a ticket before we stop waiting for its receipt and call it delivered. */
const RECEIPT_GRACE_MS = 30_000
const RECEIPT_GIVE_UP_MS = 6 * 60 * 60_000
const RECEIPT_BATCH = 300

type ExpoMessage = {
  to: string
  title: string
  body: string
  sound?: 'default'
  data?: Record<string, unknown>
  channelId?: string
  priority?: 'default' | 'normal' | 'high'
}
type ExpoTicket = { status?: 'ok' | 'error'; id?: string; message?: string; details?: { error?: string } }
type ExpoReceipt = { status?: 'ok' | 'error'; message?: string; details?: { error?: string } }

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
        channelId: ANDROID_CHANNEL_ID,
        // A caregiver alert is the entire point of the product being installed, so it is worth an FCM
        // high-priority message: normal priority is queued until the phone next leaves Doze, which on a
        // phone left on a nightstand means hours, and on an aggressive China ROM means never.
        priority: 'high',
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
      // Keep the ticket ids. An 'ok' ticket only means Expo ACCEPTED and queued the message — the real
      // delivery outcome (InvalidCredentials, DeviceNotRegistered, MessageRateExceeded, ...) exists only
      // in the receipt, keyed by these ids. Discarding them, as this did, made 'sent' unfalsifiable: a
      // notification Expo silently dropped for want of FCM credentials was indistinguishable from one
      // that rang the caregiver's phone. pollPushReceipts below resolves 'sent' into the honest answer.
      const ticketIds = tickets.filter((ticket) => ticket.status === 'ok' && ticket.id).map((ticket) => String(ticket.id))
      await notifications.updateOne({ _id: notif._id }, { $set: {
        pushState: okCount > 0 ? 'sent' : 'failed',
        pushError: okCount > 0 ? null : (tickets.find((ticket) => ticket.status === 'error')?.details?.error || tickets.find((ticket) => ticket.status === 'error')?.message || 'expo_error'),
        pushTicketCount: tickets.length, pushOkCount: okCount, pushTicketIds: ticketIds,
        pushedAt: new Date(), updatedAt: new Date(),
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

/** Ask Expo for the delivery outcome of tickets we already hold. Chunked at 300 ids, as Expo requires. */
export async function fetchExpoReceipts(ticketIds: string[]): Promise<Record<string, ExpoReceipt>> {
  const receipts: Record<string, ExpoReceipt> = {}
  for (let offset = 0; offset < ticketIds.length; offset += RECEIPT_BATCH) {
    const chunk = ticketIds.slice(offset, offset + RECEIPT_BATCH)
    try {
      const response = await fetch(EXPO_RECEIPTS_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ ids: chunk }),
        signal: AbortSignal.timeout(15_000),
      })
      const payload = await response.json().catch(() => null) as { data?: Record<string, ExpoReceipt> } | null
      if (payload?.data) Object.assign(receipts, payload.data)
    } catch {
      // Transport failure: leave these ids unresolved so the next tick retries them. A receipt we could
      // not fetch must never be recorded as a delivery failure — that would blame the phone for our own
      // network problem, and DeviceNotRegistered handling below would wrongly deactivate a live device.
    }
  }
  return receipts
}

/**
 * Resolves `pushState:'sent'` rows into `delivered` or `undelivered` using Expo's receipts.
 *
 * This is the half of push delivery that was missing. Expo answers /send with a ticket the moment it
 * accepts a message; the send call succeeding tells you nothing about whether FCM took it. The most
 * common production failure — an Expo project with no FCM V1 service-account key, so Expo cannot talk to
 * your Firebase project at all — returns an ok ticket and then a receipt of InvalidCredentials. Without
 * this pass every one of those looks like a successful delivery in the database.
 *
 * Receipts are not instant, so rows are left alone for RECEIPT_GRACE_MS, and after RECEIPT_GIVE_UP_MS
 * an unresolved ticket is accepted as delivered rather than pursued forever.
 */
export async function pollPushReceipts(db: Db, limit = DISPATCH_BATCH): Promise<{
  delivered: number; undelivered: number; pending: number
}> {
  const notifications = db.collection<any>(collections.notifications)
  const now = Date.now()
  const rows = await notifications.find({
    pushState: 'sent',
    pushTicketIds: { $exists: true, $ne: [] },
    pushedAt: { $lt: new Date(now - RECEIPT_GRACE_MS) },
  }).sort({ pushedAt: 1 }).limit(limit).toArray()
  if (!rows.length) return { delivered: 0, undelivered: 0, pending: 0 }

  const receipts = await fetchExpoReceipts([...new Set(rows.flatMap((row) => row.pushTicketIds as string[]))])
  let delivered = 0, undelivered = 0, pending = 0

  for (const row of rows) {
    const ids = row.pushTicketIds as string[]
    const resolved = ids.map((id) => receipts[id]).filter(Boolean) as ExpoReceipt[]
    const expired = now - new Date(row.pushedAt).getTime() > RECEIPT_GIVE_UP_MS

    if (!resolved.length && !expired) { pending++; continue }
    // One device receiving it is a delivered alert, even if the caregiver's other phone failed.
    const anyOk = resolved.some((receipt) => receipt.status === 'ok')
    const firstError = resolved.find((receipt) => receipt.status === 'error')

    if (anyOk || (expired && !firstError)) {
      await notifications.updateOne({ _id: row._id }, { $set: {
        pushState: 'delivered', pushError: null, pushReceiptAt: new Date(), updatedAt: new Date(),
      } })
      delivered++
      continue
    }

    const error = firstError?.details?.error || firstError?.message || 'receipt_error'
    await notifications.updateOne({ _id: row._id }, { $set: {
      pushState: 'undelivered', pushError: String(error).slice(0, 300),
      pushReceiptAt: new Date(), updatedAt: new Date(),
    } })
    undelivered++

    // A receipt is the authoritative place DeviceNotRegistered shows up for Android — the send-time
    // ticket usually cannot know yet. Retiring the token here is what stops a reinstalled phone from
    // being pushed to forever.
    if (String(error) === 'DeviceNotRegistered') {
      await db.collection<any>(collections.notificationDevices).updateMany(
        { tenantId: row.tenantId, userId: row.recipientUserId, state: 'active' },
        { $set: { state: 'inactive', updatedAt: new Date() } },
      )
    }
  }
  return { delivered, undelivered, pending }
}
