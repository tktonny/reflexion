import 'dotenv/config'
import { closeMongo, getDb } from '../lib/mongo.js'
import { materializeAllMedicationReminders } from '../v1/care/reminderScheduler.js'

try {
  const count = await materializeAllMedicationReminders(await getDb())
  console.log(`Prepared ${count} medication reminder occurrences.`)
} finally {
  await closeMongo()
}
