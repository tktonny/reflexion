// Verify the Flavor B (turn-based HTTP) LLM path against DashScope with the real key.
// Chain: Qwen chat  ->  TTS (synthesize)  ->  ASR (transcribe the TTS audio back).
// China-region key (relay showed intl 401 -> china OK), so use dashscope.aliyuncs.com.
// Run:  node --env-file=.env server/smoke-turnbased.mjs

const KEY = process.env.QWEN_API_KEY || process.env.DASHSCOPE_API_KEY
const BASE = 'https://dashscope.aliyuncs.com'
const CHAT_URL = `${BASE}/compatible-mode/v1/chat/completions`
const MM_URL = `${BASE}/api/v1/services/aigc/multimodal-generation/generation`

const CHAT_MODEL = process.env.SMOKE_CHAT_MODEL || 'qwen-plus'
const TTS_MODEL = process.env.SMOKE_TTS_MODEL || 'qwen-tts'
const ASR_MODEL = process.env.SMOKE_ASR_MODEL || 'qwen3-asr-flash'

if (!KEY) { console.error('no key'); process.exit(1) }
const H = { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' }
const short = (s, n = 400) => (typeof s === 'string' ? s : JSON.stringify(s)).slice(0, n)

async function step(name, fn) {
  console.log(`\n=== ${name} ===`)
  try { return await fn() } catch (e) { console.error(`[${name}] ERROR`, e?.message); return null }
}

// 1) CHAT (OpenAI-compatible)
const chatReply = await step('CHAT', async () => {
  const res = await fetch(CHAT_URL, {
    method: 'POST', headers: H,
    body: JSON.stringify({
      model: CHAT_MODEL,
      messages: [
        { role: 'system', content: 'You are Aria, a warm check-in companion. Reply in one short sentence.' },
        { role: 'user', content: '你好' },
      ],
    }),
  })
  const body = await res.json()
  console.log('status', res.status)
  const text = body?.choices?.[0]?.message?.content
  console.log('reply:', short(text ?? body))
  return text
})

// 2) TTS (multimodal-generation)
const ttsAudio = await step('TTS', async () => {
  const phrase = '早上好，很高兴见到你。'
  const res = await fetch(MM_URL, {
    method: 'POST', headers: H,
    body: JSON.stringify({ model: TTS_MODEL, input: { text: phrase, voice: 'Cherry' } }),
  })
  const body = await res.json()
  console.log('status', res.status)
  const audio = body?.output?.audio
  console.log('output.audio keys:', audio ? Object.keys(audio) : short(body))
  const url = audio?.url
  const b64 = audio?.data
  if (url) {
    console.log('audio url:', url.slice(0, 120))
    const wav = await fetch(url)
    const buf = Buffer.from(await wav.arrayBuffer())
    console.log('downloaded audio bytes:', buf.length, 'content-type:', wav.headers.get('content-type'))
    return { bytes: buf, format: (wav.headers.get('content-type') || '').includes('mp3') ? 'mp3' : 'wav', phrase }
  }
  if (b64) { const buf = Buffer.from(b64, 'base64'); console.log('inline b64 bytes:', buf.length); return { bytes: buf, format: 'wav', phrase } }
  return null
})

// 3) ASR (transcribe the TTS audio back) via compatible-mode audio input
await step('ASR', async () => {
  if (!ttsAudio) { console.log('skipped (no TTS audio)'); return }
  const dataUrl = `data:;base64,${ttsAudio.bytes.toString('base64')}`
  const res = await fetch(CHAT_URL, {
    method: 'POST', headers: H,
    body: JSON.stringify({
      model: ASR_MODEL,
      messages: [
        { role: 'system', content: [{ type: 'text', text: '' }] },
        { role: 'user', content: [{ type: 'input_audio', input_audio: { data: dataUrl } }] },
      ],
    }),
  })
  const body = await res.json()
  console.log('status', res.status)
  const text = body?.choices?.[0]?.message?.content
  console.log('transcript:', short(text ?? body))
  console.log('(expected ~= "' + ttsAudio.phrase + '")')
})

console.log('\n=== smoke done ===')
