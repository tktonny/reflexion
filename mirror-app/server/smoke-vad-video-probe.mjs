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

// The device's REAL companion payload (src/orchestration/realtime.ts): voice Serena, max_tokens 256,
// 4 flat tools, semantic_vad(0.5, 1200). Screening is identical minus tools + turn_detection:null.
const DEV = { voice: 'Serena', max_tokens: 256, temperature: 0.25 }
const TOOLS4 = [
  { type: 'function', name: 'get_weather', description: "Current weather and a short forecast for a city. Omit `city` for the patient's home area.", parameters: { type: 'object', properties: { city: { type: 'string', description: 'City name, e.g. Tokyo.' } } } },
  { type: 'function', name: 'web_search', description: 'Search the web for current facts, news, or a general question you are unsure about.', parameters: { type: 'object', properties: { query: { type: 'string', description: 'The search query.' } }, required: ['query'] } },
  { type: 'function', name: 'list_medications', description: "List the patient's current medications and their schedules.", parameters: { type: 'object', properties: {} } },
  { type: 'function', name: 'upcoming_reminders', description: "List the patient's reminders or medications due in the next 24 hours.", parameters: { type: 'object', properties: {} } },
]
const SEM = { type: 'semantic_vad', threshold: 0.5, silence_duration_ms: 1200, create_response: true, interrupt_response: false }
const SERVER = { type: 'server_vad', threshold: 0.5, silence_duration_ms: 1200, create_response: true, interrupt_response: false }

const VARIANTS_OLD = {
  'device SCREENING (Serena, td:null, no tools)': { ...DEV, turn_detection: null },
  'device COMPANION FULL (Serena, semantic_vad, 4 tools)': { ...DEV, turn_detection: SEM, tools: TOOLS4 },
  '  minus tools (Serena, semantic_vad)': { ...DEV, turn_detection: SEM },
  '  minus semantic_vad (Serena, server_vad, 4 tools)': { ...DEV, turn_detection: SERVER, tools: TOOLS4 },
  '  Cherry instead of Serena (FULL)': { voice: 'Cherry', max_tokens: 256, temperature: 0.25, turn_detection: SEM, tools: TOOLS4 },
  '  each tool alone: web_search only': { ...DEV, turn_detection: SEM, tools: [TOOLS4[1]] },
  '  each tool alone: get_weather only': { ...DEV, turn_detection: SEM, tools: [TOOLS4[0]] },
  '  empty-params tools only (list_meds+reminders)': { ...DEV, turn_detection: SEM, tools: [TOOLS4[2], TOOLS4[3]] },
  'FIX: COMPANION FULL + seeded user item': { ...DEV, turn_detection: SEM, tools: TOOLS4, __seed: true },
}

async function mintToken() {
  const r = await fetch('https://dashscope.aliyuncs.com/api/v1/tokens?expire_in_seconds=1800', { method: 'POST', headers: H, body: '{}' })
  const b = await r.json()
  return b?.token || null
}

// 探测:semantic_vad 的 create_response/interrupt_response 组合,以及 video 模态 + 图像事件
const SEMV = (o) => ({ type: 'semantic_vad', threshold: 0.5, silence_duration_ms: 1200, ...o })
const VARIANTS = {
  'A. semantic_vad create_response:true  (现状默认)': { ...DEV, turn_detection: SEMV({ create_response: true, interrupt_response: false }) },
  'B. semantic_vad create_response:FALSE (我们想要的)': { ...DEV, turn_detection: SEMV({ create_response: false, interrupt_response: false }) },
  'C. semantic_vad interrupt_response:TRUE (provider 侧打断)': { ...DEV, turn_detection: SEMV({ create_response: false, interrupt_response: true }) },
  'D. turn_detection:null (生产现状)': { ...DEV, turn_detection: null },
  'E. modalities 含 video': { ...DEV, turn_detection: SEMV({ create_response: false }), __modalities: ['text','audio','video'] },
}

function testVariant(extra) {
  const seed = extra.__seed
  const sessionExtra = { ...extra }; delete sessionExtra.__seed
  const modalities = sessionExtra.__modalities; delete sessionExtra.__modalities
  return new Promise(async (resolve) => {
    const auth = DIRECT ? KEY : await mintToken()
    if (!auth) return resolve('no-token')
    const ws = new WebSocket(`${HOST}?model=${MODEL}`, { headers: { Authorization: `Bearer ${auth}` }, maxPayload: 0 })
    let done = false
    const finish = (r) => { if (done) return; done = true; clearTimeout(timer); try { ws.close() } catch {} ; resolve(r) }
    const timer = setTimeout(() => finish('OK (no error in 6s)'), 6000)
    ws.on('open', () => ws.send(JSON.stringify({
      event_id: 'evt_1', type: 'session.update',
      session: { modalities: modalities || ['text', 'audio'], voice: qwenConfig.defaultVoice, instructions: 'You are a warm companion. Reply in English.', input_audio_format: 'pcm', output_audio_format: 'pcm', input_audio_transcription: { model: qwenConfig.transcriptionModel }, ...sessionExtra },
    })))
    ws.on('message', (d) => { let m; try { m = JSON.parse(d.toString()) } catch { return }
      if (m.type === 'error') finish(`ERROR ${m.error?.code || '?'}: ${m.error?.message || JSON.stringify(m.error)}`)
      // Don't stop at session.updated — the device then sends response.create, and the rejection may
      // land there. SG requires a user-role message to exist first; `seed` injects a hidden text item
      // (input_text) before response.create — this is the candidate fix.
      if (m.type === 'session.updated') {
        if (seed) ws.send(JSON.stringify({ type: 'conversation.item.create', item: { type: 'message', role: 'user', content: [{ type: 'input_text', text: '(Please greet me warmly to begin.)' }] } }))
        ws.send(JSON.stringify({ type: 'response.create' }))
      }
      if (m.type === 'response.created' || m.type === 'response.audio.delta' || m.type === 'response.output_item.added') finish('accepted ✓ (session.update + response.create)')
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
