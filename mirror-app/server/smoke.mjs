// Headless verification: connect to the relay exactly like the browser would,
// and confirm the upstream Qwen session comes up (reflexion.session.ready ->
// session.created/updated). Proves auth + session.update orchestration reach Qwen
// without needing a microphone. Exits non-zero if the session never becomes ready.

import { WebSocket } from 'ws'

const port = process.env.REFLEXION_RELAY_PORT || 8787
const url = `ws://localhost:${port}/api/clinic/realtime/ws?patient_id=demo-patient&language=en`
const seen = new Set()
let qwenSessionUp = false

console.log('[smoke] connecting', url)
const ws = new WebSocket(url)

const timer = setTimeout(() => {
  console.log('[smoke] timeout reached, events seen:', [...seen].join(', ') || '(none)')
  finish()
}, 12000)

ws.on('open', () => console.log('[smoke] client connected to relay'))
ws.on('message', (data) => {
  let msg
  try { msg = JSON.parse(data.toString()) } catch { return }
  const type = String(msg.type || '')
  if (!seen.has(type)) {
    seen.add(type)
    console.log('[smoke] event:', type)
  }
  if (type === 'reflexion.session.degraded') {
    console.error('[smoke] DEGRADED:', msg.reason)
  }
  if (type === 'session.created' || type === 'session.updated') {
    qwenSessionUp = true
    // Got proof the Qwen upstream session is live; wrap up and close.
    setTimeout(finish, 1500)
  }
})
ws.on('error', (err) => { console.error('[smoke] ws error:', err.message) })
ws.on('close', () => console.log('[smoke] closed'))

function finish() {
  clearTimeout(timer)
  try { ws.close() } catch {}
  if (qwenSessionUp) {
    console.log('[smoke] RESULT: PASS — Qwen realtime upstream session is live through the relay.')
    process.exit(0)
  } else {
    console.log('[smoke] RESULT: FAIL — never saw session.created/updated from Qwen.')
    process.exit(1)
  }
}
