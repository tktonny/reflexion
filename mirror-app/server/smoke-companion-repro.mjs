// Reproduce the companion `session.update` 400 (InternalError.Algo.InvalidParameter) that knocks
// companion off omni. We have no SG key locally, but if the bad parameter is universal, CN omni
// rejects it identically — and prints the FULL message the device HUD truncates. Sweeps variants to
// isolate whether it's the model, semantic_vad, or tools (flat vs nested).
// Run: node --env-file=.env.server.local server/smoke-companion-repro.mjs
import { WebSocket } from 'ws'
import { qwenConfig } from './qwenConfig.mjs'

const KEY = process.env.REPRO_KEY || qwenConfig.apiKey
if (!KEY) { console.error('no key (set REPRO_KEY or QWEN_API_KEY)'); process.exit(1) }
const MODEL = process.env.REPRO_MODEL || 'qwen3.5-omni-flash-realtime'
// Point at SG to reproduce the real rejection:
//   REPRO_KEY=<sg sk-ws- key> REPRO_HOST=wss://ws-s37sbnnxivio0l58.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime \
//   REPRO_DIRECT=1 node server/smoke-companion-repro.mjs
// REPRO_DIRECT=1 authenticates the WS with the key itself (skips the CN token-mint step, which is
// CN-host-only). Default (no REPRO_HOST) still runs the CN control test.
const HOST = process.env.REPRO_HOST || qwenConfig.realtimeUrlChina
const DIRECT = process.env.REPRO_DIRECT === '1'
const H = { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' }

const TOOLS_FLAT = [{ type: 'function', name: 'get_weather', description: 'weather', parameters: { type: 'object', properties: { city: { type: 'string' } } } }]
const TOOLS_NESTED = [{ type: 'function', function: { name: 'get_weather', description: 'weather', parameters: { type: 'object', properties: { city: { type: 'string' } } } } }]
const SEMANTIC = { type: 'semantic_vad', threshold: 0.5, silence_duration_ms: 900, create_response: true, interrupt_response: false }
const SERVER = { type: 'server_vad', threshold: 0.1, silence_duration_ms: 900, create_response: true, interrupt_response: false }

const VARIANTS = {
  'baseline server_vad, no tools': { turn_detection: SERVER },
  'semantic_vad, no tools': { turn_detection: SEMANTIC },
  'server_vad + tools(flat)': { turn_detection: SERVER, tools: TOOLS_FLAT },
  'semantic_vad + tools(flat)': { turn_detection: SEMANTIC, tools: TOOLS_FLAT },
  'semantic_vad + tools(nested)': { turn_detection: SEMANTIC, tools: TOOLS_NESTED },
  'semantic_vad, no interrupt_response field': { turn_detection: { type: 'semantic_vad', threshold: 0.5, silence_duration_ms: 900, create_response: true } },
}

async function mintToken() {
  const r = await fetch('https://dashscope.aliyuncs.com/api/v1/tokens?expire_in_seconds=1800', { method: 'POST', headers: H, body: '{}' })
  const b = await r.json()
  return b?.token || null
}

function testVariant(extra) {
  return new Promise(async (resolve) => {
    const auth = DIRECT ? KEY : await mintToken()
    if (!auth) return resolve('no-token')
    const ws = new WebSocket(`${HOST}?model=${MODEL}`, { headers: { Authorization: `Bearer ${auth}` }, maxPayload: 0 })
    let done = false
    const finish = (r) => { if (done) return; done = true; clearTimeout(timer); try { ws.close() } catch {} ; resolve(r) }
    const timer = setTimeout(() => finish('OK (no error in 6s)'), 6000)
    ws.on('open', () => ws.send(JSON.stringify({
      event_id: 'evt_1', type: 'session.update',
      session: { modalities: ['text', 'audio'], voice: qwenConfig.defaultVoice, instructions: 'You are a warm companion. Reply in English.', input_audio_format: 'pcm', output_audio_format: 'pcm', input_audio_transcription: { model: qwenConfig.transcriptionModel }, ...extra },
    })))
    ws.on('message', (d) => { let m; try { m = JSON.parse(d.toString()) } catch { return }
      if (m.type === 'error') finish(`ERROR ${m.error?.code || '?'}: ${m.error?.message || JSON.stringify(m.error)}`)
      if (m.type === 'session.updated') finish('session.updated ✓ (accepted)')
    })
    ws.on('error', (e) => finish(`ws-error ${e?.message}`))
    ws.on('unexpected-response', (_r, res) => finish(`handshake HTTP ${res.statusCode}`))
  })
}

console.log('host:', HOST, '\nmodel:', MODEL, DIRECT ? '(direct-key auth)' : '(minted token)', '\n')
for (const [name, extra] of Object.entries(VARIANTS)) {
  const r = await testVariant(extra)
  console.log(name.padEnd(42), '→', r)
}
process.exit(0)
