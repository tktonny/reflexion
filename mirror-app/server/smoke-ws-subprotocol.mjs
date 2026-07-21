// Can a BROWSER (no custom headers) auth the Qwen realtime WS via Sec-WebSocket-Protocol?
// Browsers can't set Authorization on WebSocket, but CAN pass subprotocols. OpenAI's realtime
// accepts ["realtime","openai-insecure-api-key.<KEY>","openai-beta.realtime-v1"]. Test if Qwen does.
// If PASS -> v3 can run relay-less directly in Chrome. Run: node --env-file=.env server/smoke-ws-subprotocol.mjs
import { WebSocket } from 'ws'
import { qwenConfig } from './qwenConfig.mjs'

const KEY = qwenConfig.apiKey
const model = qwenConfig.realtimeModel
const candidates = [qwenConfig.realtimeUrlChina, qwenConfig.realtimeUrl].filter(Boolean)

async function tryOne(base) {
  const url = `${base}?model=${model}`
  return new Promise((resolve) => {
    // Mimic exactly what a browser sends: protocols array, NO Authorization header.
    const protocols = ['realtime', `openai-insecure-api-key.${KEY}`, 'openai-beta.realtime-v1']
    const ws = new WebSocket(url, protocols, { maxPayload: 0 })
    let done = false
    const finish = (ok, why) => { if (done) return; done = true; try { ws.close() } catch {} ; resolve({ ok, why }) }
    const t = setTimeout(() => finish(false, 'timeout'), 12000)
    ws.on('open', () => console.log(`  [${base}] OPEN (subprotocol accepted at handshake)`))
    ws.on('unexpected-response', (_q, res) => finish(false, `handshake HTTP ${res.statusCode}`))
    ws.on('message', (d) => {
      let m; try { m = JSON.parse(d.toString()) } catch { return }
      if (m.type === 'session.created' || m.type === 'session.updated') { clearTimeout(t); finish(true, m.type) }
      if (m.type === 'error') { clearTimeout(t); finish(false, `error: ${JSON.stringify(m.error||m).slice(0,120)}`) }
    })
    ws.on('error', (e) => finish(false, `wserr: ${e?.message}`))
  })
}

let pass = false
for (const base of candidates) {
  console.log(`\n--- ${base} (subprotocol auth) ---`)
  const r = await tryOne(base)
  console.log(`  -> ${r.ok ? 'PASS' : 'FAIL'} (${r.why})`)
  if (r.ok) { pass = true; break }
}
console.log(`\nRESULT: ${pass ? 'PASS — browser subprotocol auth works → v3 can run in Chrome relay-less' : 'FAIL — no subprotocol auth; v3 stays native/dev-build only'}`)
process.exit(pass ? 0 : 1)
