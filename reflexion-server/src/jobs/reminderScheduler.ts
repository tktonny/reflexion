import 'dotenv/config'
import { closeMongo, getDb } from '../lib/mongo.js'
import { materializeAllReminders } from '../v1/care/reminderScheduler.js'

try {
  const count = await materializeAllReminders(await getDb())
  console.log(`Prepared ${count} medication reminder occurrences.`)
} finally {
  await closeMongo()
}
