// Which voices does qwen3.5-omni-flash-realtime accept? Cherry (our default) is rejected on 3.5.
// Probe a candidate list on the generic China host + semantic_vad; a voice that streams audio is valid.
// Run: node --env-file=.env server/smoke-voice-probe.mjs

import { WebSocket } from 'ws'
import { qwenConfig } from './qwenConfig.mjs'
import { buildLiveInstructions } from './generated/orchestration.mjs'

const KEY = qwenConfig.apiKey
const HOST = qwenConfig.realtimeUrlChina
const MODEL = 'qwen3.5-omni-flash-realtime'
const INSTR = buildLiveInstructions('demo-patient', 'en', {})
const VOICES = ['Kiki', 'Rocky', 'Joseph Chen', 'JosephChen', 'Cindy', 'Jennifer', 'Aiden', 'Marcus', 'Li', 'Eric', 'Sohee']

function trial(voice) {
  return new Promise((resolve) => {
    const ws = new WebSocket(`${HOST}?model=${MODEL}`, { headers: { Authorization: `Bearer ${KEY}` }, maxPayload: 0 })
    const r = { voice, updated: false, audio: false, error: '' }
    const done = (why) => { clearTimeout(timer); try { ws.close() } catch {}; resolve(r) }
    const timer = setTimeout(() => done('timeout'), 12000)
    ws.on('open', () => ws.send(JSON.stringify({
      type: 'session.update',
      session: {
        modalities: ['text', 'audio'], voice, instructions: INSTR,
        input_audio_format: 'pcm', output_audio_format: 'pcm',
        turn_detection: { type: 'semantic_vad', threshold: 0.5, silence_duration_ms: 800, create_response: true, interrupt_response: false },
        input_audio_transcription: { model: qwenConfig.transcriptionModel },
      },
    })))
    ws.on('message', (data) => {
      let m; try { m = JSON.parse(data.toString()) } catch { return }
      const t = String(m.type || '')
      if (t === 'session.updated') { r.updated = true; ws.send(JSON.stringify({ type: 'response.create' })) }
      if (t === 'error') { r.error = String(m?.error?.message || 'error'); done('err') }
      if (t === 'response.audio.delta') { r.audio = true; done('audio') }
    })
    ws.on('error', () => {})
  })
}

console.log(`probing voices on ${MODEL} + semantic_vad\n`)
const good = []
for (const v of VOICES) {
  const r = await trial(v)
  const ok = r.audio ? '✅ streams audio' : r.error ? `❌ ${r.error}` : '🟡 no audio/timeout'
  console.log(`${v.padEnd(9)} ${ok}`)
  if (r.audio) good.push(v)
}
console.log(`\nVALID voices (streamed audio): ${good.join(', ') || '(none in list)'}`)
process.exit(0)
