import 'dotenv/config'
import { closeMongo } from '../lib/mongo.js'
import { ensureV1Indexes } from '../v1/platform/indexes.js'

try {
  await ensureV1Indexes()
  console.log('Reflexion API v1 indexes are ready.')
} finally {
  await closeMongo()
}
