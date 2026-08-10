import 'dotenv/config'
import { closeMongo, getDb } from '../lib/mongo.js'
import { processNextOutboxEvent } from '../v1/workers/outboxWorker.js'
import { dispatchPendingPushes, pollPushReceipts } from '../v1/notifications/push.js'

const once = process.argv.includes('--once')
let stopping = false
process.on('SIGINT', () => { stopping = true })
process.on('SIGTERM', () => { stopping = true })

try {
  do {
    const processed = await processNextOutboxEvent()
    // Deliver any notifications awaiting a phone push. Rides this existing process — no extra service.
    // A push failure must never kill the outbox loop.
    try { await dispatchPendingPushes(await getDb()) } catch (error) {
      console.error('[worker] push dispatch failed', error instanceof Error ? error.message : error)
    }
    // Then resolve already-sent pushes against Expo's receipts, which is the only place a push that was
    // accepted and then silently dropped (InvalidCredentials, DeviceNotRegistered) can be observed.
    try { await pollPushReceipts(await getDb()) } catch (error) {
      console.error('[worker] push receipt poll failed', error instanceof Error ? error.message : error)
    }
    if (once) break
    if (!processed) await new Promise((resolve) => setTimeout(resolve, 1000))
  } while (!stopping)
} finally {
  await closeMongo()
}
