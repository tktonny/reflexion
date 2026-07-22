import 'dotenv/config'
import { createApp } from './app.js'
import { closeMongo } from './lib/mongo.js'

const port = Number(process.env.PORT || 3001)
const host = process.env.HOST || '0.0.0.0'

const server = createApp().listen(port, host, () => {
  console.log(`reflexion-caregiver-app-server listening on http://${host}:${port}`)
})

async function shutdown() {
  server.close(async () => {
    await closeMongo()
    process.exit(0)
  })
}

process.on('SIGINT', shutdown)
process.on('SIGTERM', shutdown)
