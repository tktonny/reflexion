import 'dotenv/config'
import { closeMongo } from '../lib/mongo.js'
import { processNextOutboxEvent } from '../v1/workers/outboxWorker.js'

const once = process.argv.includes('--once')
let stopping = false
process.on('SIGINT', () => { stopping = true })
process.on('SIGTERM', () => { stopping = true })

try {
  do {
    const processed = await processNextOutboxEvent()
    if (once) break
    if (!processed) await new Promise((resolve) => setTimeout(resolve, 1000))
  } while (!stopping)
} finally {
  await closeMongo()
}
