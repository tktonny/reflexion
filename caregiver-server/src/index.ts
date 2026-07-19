import 'dotenv/config'
import { createApp } from './app.js'

const port = Number(process.env.PORT || 3001)
const host = process.env.HOST || '0.0.0.0'

createApp().listen(port, host, () => {
  console.log(`reflexion-caregiver-app-server listening on http://${host}:${port}`)
})
