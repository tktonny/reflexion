// Verify the v3 path end-to-end minus device audio: mint a SHORT-LIVED token, then open a
// DIRECT WebSocket to Qwen realtime with that token in the Authorization header (exactly what
// the native RN client does), send the on-device orchestration session.update, and confirm
// Qwen streams a reply. Answers the report's open question: can an ephemeral token auth the WS?
// Run: node --env-file=.env server/smoke-direct-ws.mjs

import { WebSocket } from 'ws'
import { qwenConfig } from './qwenConfig.mjs'
import { buildLiveInstructions } from './orchestrator.mjs'

const LONG_KEY = qwenConfig.apiKey
const H = { Authorization: `Bearer ${LONG_KEY}`, 'Content-Type': 'application/json' }

// 1) mint short-lived token
const tokenRes = await fetch('https://dashscope.aliyuncs.com/api/v1/tokens?expire_in_seconds=1800', { method: 'POST', headers: H, body: '{}' })
const tokenBody = await tokenRes.json()
const token = tokenBody?.token
console.log('minted token:', token ? `${token.slice(0, 6)}…${token.slice(-4)}` : tokenBody)
if (!token) { console.error('no token'); process.exit(1) }

// 2) direct WS to Qwen with the TOKEN in the header (China host — our key is China-region)
const url = `${qwenConfig.realtimeUrlChina}?model=${qwenConfig.realtimeModel}`
console.log('connecting DIRECT ws:', url)
const ws = new WebSocket(url, { headers: { Authorization: `Bearer ${token}` }, maxPayload: 0 })

let pass = false
const seen = new Set()
const timer = setTimeout(() => finish('timeout'), 15000)

ws.on('open', () => {
  console.log('ws OPEN (token accepted for handshake ✓)')
  ws.send(JSON.stringify({
    event_id: 'evt_1',
    type: 'session.update',
    session: {
      modalities: ['text', 'audio'],
      voice: qwenConfig.defaultVoice,
      instructions: buildLiveInstructions('demo-patient', 'en', {}),
      input_audio_format: 'pcm',
      output_audio_format: 'pcm',
      turn_detection: { type: 'server_vad', threshold: qwenConfig.vadThreshold, prefix_padding_ms: qwenConfig.vadPrefixPaddingMs, silence_duration_ms: qwenConfig.vadSilenceDurationMs, create_response: true, interrupt_response: false },
      input_audio_transcription: { model: qwenConfig.transcriptionModel },
    },
  }))
})
ws.on('unexpected-response', (_req, res) => { console.error('handshake HTTP', res.statusCode, '(token rejected for WS)'); finish('handshake-failed') })
ws.on('message', (data) => {
  let m; try { m = JSON.parse(data.toString()) } catch { return }
  const t = String(m.type || '')
  if (!seen.has(t)) { seen.add(t); console.log('event:', t) }
  if (t === 'session.created' || t === 'session.updated') ws.send(JSON.stringify({ type: 'response.create' }))
  if (t === 'response.audio.delta') { pass = true; finish('got-audio') }
})
ws.on('error', (e) => console.error('ws error', e?.message))

function finish(reason) {
  clearTimeout(timer)
  try { ws.close() } catch {}
  console.log(`\nRESULT: ${pass ? 'PASS' : 'FAIL'} (${reason}) — direct WS + ephemeral token + on-device orchestration`)
  process.exit(pass ? 0 : 1)
}
