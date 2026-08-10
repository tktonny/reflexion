// API-LEVEL echo-fix preflight: does our generic China realtime host accept the qwen3.5 realtime
// models AND turn_detection.type=semantic_vad (the anti-echo lever), with our existing key + gummy
// transcription? Tries a small matrix and prints, per combo: handshake, session.updated vs error
// (with the server's error message), and whether audio streams on response.create.
//
// Run: node --env-file=.env server/smoke-semantic-vad.mjs
// Prints NO secrets.

import { WebSocket } from 'ws'
import { qwenConfig } from './qwenConfig.mjs'
import { buildLiveInstructions } from './generated/orchestration.mjs'

const KEY = qwenConfig.apiKey
if (!KEY) { console.error('no QWEN_API_KEY / DASHSCOPE_API_KEY in env'); process.exit(2) }
const HOST = qwenConfig.realtimeUrlChina
const INSTR = buildLiveInstructions('demo-patient', 'en', {})

const MATRIX = [
  { model: 'qwen3.5-omni-flash-realtime', vad: 'semantic_vad' }, // the proposed fix
  { model: 'qwen3.5-omni-plus-realtime', vad: 'semantic_vad' },  // richer fallback
  { model: 'qwen3-omni-flash-realtime', vad: 'semantic_vad' },   // does OLD model reject semantic_vad?
  { model: 'qwen3.5-omni-flash-realtime', vad: 'server_vad' },   // does 3.5 work at all on this host?
]

function turnDetection(vad) {
  return vad === 'semantic_vad'
    ? { type: 'semantic_vad', threshold: 0.5, silence_duration_ms: 800, create_response: true, interrupt_response: false }
    : { type: 'server_vad', threshold: 0.5, prefix_padding_ms: 300, silence_duration_ms: 800, create_response: true, interrupt_response: false }
}

function trial({ model, vad }) {
  return new Promise((resolve) => {
    const url = `${HOST}?model=${model}`
    const ws = new WebSocket(url, { headers: { Authorization: `Bearer ${KEY}` }, maxPayload: 0 })
    const r = { model, vad, open: false, created: false, updated: false, audio: false, error: '', http: '' }
    const seen = new Set()
    const done = (why) => { clearTimeout(timer); try { ws.close() } catch {}; r.why = why; resolve(r) }
    const timer = setTimeout(() => done('timeout'), 14000)

    ws.on('open', () => {
      r.open = true
      ws.send(JSON.stringify({
        event_id: 'evt_1', type: 'session.update',
        session: {
          modalities: ['text', 'audio'], voice: qwenConfig.defaultVoice, instructions: INSTR,
          input_audio_format: 'pcm', output_audio_format: 'pcm',
          turn_detection: turnDetection(vad),
          input_audio_transcription: { model: qwenConfig.transcriptionModel },
        },
      }))
    })
    ws.on('unexpected-response', (_req, res) => { r.http = String(res.statusCode); done('handshake-http') })
    ws.on('message', (data) => {
      let m; try { m = JSON.parse(data.toString()) } catch { return }
      const t = String(m.type || '')
      if (!seen.has(t)) seen.add(t)
      if (t === 'session.created') r.created = true
      if (t === 'session.updated') { r.updated = true; ws.send(JSON.stringify({ type: 'response.create' })) }
      if (t === 'error') { r.error = String(m?.error?.message || m?.error?.code || 'error'); done('server-error') }
      if (t === 'response.audio.delta') { r.audio = true; done('got-audio') }
    })
    ws.on('error', (e) => { if (!r.error) r.error = e?.message || 'ws-error' })
  })
}

console.log(`host: ${HOST}`)
console.log(`key: ${KEY.slice(0, 5)}…${KEY.slice(-3)} · transcription: ${qwenConfig.transcriptionModel}\n`)
const results = []
for (const combo of MATRIX) {
  process.stdout.write(`testing ${combo.model} + ${combo.vad} … `)
  const r = await trial(combo)
  results.push(r)
  const verdict = r.audio ? 'PASS (audio)' : r.updated ? 'PARTIAL (session.updated, no audio)' : 'FAIL'
  console.log(`${verdict}  [open=${r.open} created=${r.created} updated=${r.updated} audio=${r.audio}${r.http ? ` http=${r.http}` : ''}${r.error ? ` err="${r.error}"` : ''}]`)
}

console.log('\n=== SUMMARY ===')
for (const r of results) {
  const ok = r.audio ? '✅' : r.updated ? '🟡' : '❌'
  console.log(`${ok} ${r.model} + ${r.vad} → ${r.audio ? 'streams audio' : r.updated ? 'accepted config, no audio in window' : (r.error || r.http || r.why)}`)
}
const win = results.find((r) => r.model.startsWith('qwen3.5') && r.vad === 'semantic_vad' && (r.audio || r.updated))
console.log(win ? `\n➡️  API-level fix VIABLE: ${win.model} + semantic_vad accepted on the generic China host.` : `\n➡️  API-level fix NOT confirmed on this host — may need the {WorkspaceId}.maas host or model enablement.`)
process.exit(0)
