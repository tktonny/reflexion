// Standalone Node realtime-relay server.
// One http.Server; a ws.Server is attached on the HTTP `upgrade` event at
// /api/clinic/realtime/ws (Expo Router +api.ts routes are Request->Response and
// cannot perform a WebSocket 101 upgrade — this is the target-architecture shape).
//
// Run:  node --env-file=.env server/index.mjs   (or: npm run relay)

import { createServer } from 'node:http'
import { readFileSync } from 'node:fs'
import { WebSocketServer } from 'ws'

import { qwenConfig, relayPort, realtimeWsPath } from './qwenConfig.mjs'
import { voiceProfileForSession } from './generated/orchestration.mjs'
import { runLiveQwen } from './relay.mjs'

// Minimal .env fallback loader (in case the process was started without --env-file).
function loadEnvFallback() {
  if (qwenConfig.apiKey) return
  for (const file of ['.env', '.env.local']) {
    try {
      const text = readFileSync(new URL(`../${file}`, import.meta.url), 'utf-8')
      for (const line of text.split('\n')) {
        const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/)
        if (m && process.env[m[1]] === undefined) {
          process.env[m[1]] = m[2].replace(/^["']|["']$/g, '')
        }
      }
    } catch {}
  }
}
loadEnvFallback()
// Recompute apiKey after fallback load.
qwenConfig.apiKey = qwenConfig.apiKey || process.env.QWEN_API_KEY || process.env.DASHSCOPE_API_KEY || null

const httpServer = createServer((req, res) => {
  if (req.url === '/health' || req.url === '/') {
    res.writeHead(200, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ ok: true, apiKeyPresent: Boolean(qwenConfig.apiKey), model: qwenConfig.realtimeModel, path: realtimeWsPath }))
    return
  }
  res.writeHead(404)
  res.end('not found')
})

const wss = new WebSocketServer({ noServer: true })

// Relay upgrade auth (production): verify the paired device. Enable via RELAY_ENFORCE_AUTH=true.
let relayMongo = null
async function relayVerify(deviceId, authToken) {
  if (!deviceId || !authToken) return false
  const uri = process.env.MONGODB_URI
  if (!uri) return false
  try {
    if (!relayMongo) {
      const { MongoClient } = await import('mongodb')
      relayMongo = new MongoClient(uri)
      await relayMongo.connect()
    }
    const session = await relayMongo.db('ref').collection('MirrorPairingSessions').findOne({ deviceId, authToken, status: 'paired' })
    return Boolean(session)
  } catch { return false }
}

httpServer.on('upgrade', async (req, socket, head) => {
  const url = new URL(req.url, 'http://localhost')
  if (url.pathname !== realtimeWsPath) {
    socket.destroy()
    return
  }
  const patientId = url.searchParams.get('patient_id') || 'demo-patient'
  const language = url.searchParams.get('language') || 'en'
  const persona = url.searchParams.get('persona') === 'companion' ? 'companion' : 'screening'
  if (process.env.RELAY_ENFORCE_AUTH === 'true') {
    const ok = await relayVerify(url.searchParams.get('device_id') || undefined, url.searchParams.get('auth_token') || undefined)
    if (!ok) { socket.write('HTTP/1.1 401 Unauthorized\r\n\r\n'); socket.destroy(); return }
  } else if (!process.env.__RELAY_AUTH_WARNED) {
    process.env.__RELAY_AUTH_WARNED = '1'
    console.warn('[relay] WARNING: WS upgrade auth is OFF (dev). Set RELAY_ENFORCE_AUTH=true in production.')
  }
  wss.handleUpgrade(req, socket, head, (clientWs) => {
    void handleSession(clientWs, { patientId, language, persona })
  })
})

async function handleSession(clientWs, { patientId, language, persona }) {
  const voice = voiceProfileForSession(language)
  const status = {
    session_mode: 'live_qwen',
    conversation_provider: 'qwen_omni_realtime',
    model_name: qwenConfig.realtimeModel,
    live_relay_available: Boolean(qwenConfig.apiKey),
    selected_voice: voice.voice,
    selected_language: voice.languageLabel,
    persona,
    max_session_seconds: qwenConfig.maxSessionSeconds,
  }
  console.log(`[relay] session accepted patient_id=${patientId} language=${language} persona=${persona} voice=${voice.voice}`)
  send(clientWs, { type: 'reflexion.session.ready', session: status })

  try {
    await runLiveQwen(clientWs, { patientId, language, persona })
  } catch (err) {
    console.error('[relay] live relay degraded:', err?.message)
    send(clientWs, {
      type: 'reflexion.session.degraded',
      reason: String(err?.message || err),
      session: { ...status, session_mode: 'guided_demo', live_relay_available: false },
    })
    // No Node guided-demo fallback yet; close so the client can surface the error.
    try { clientWs.close() } catch {}
  }
}

function send(ws, payload) {
  if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(payload))
}

httpServer.listen(relayPort, () => {
  console.log(`[relay] listening on http://localhost:${relayPort}  ws=${realtimeWsPath}`)
  console.log(`[relay] Qwen model=${qwenConfig.realtimeModel}  apiKey=${qwenConfig.apiKey ? 'present' : 'MISSING'}`)
  if (!qwenConfig.apiKey) console.warn('[relay] WARNING: set QWEN_API_KEY or DASHSCOPE_API_KEY in .env')
})
