// Verify the Flavor B turn-based CONVERSATION BRAIN end-to-end: build the system
// prompt with our on-device orchestration (buildLiveInstructions) and run a few
// simulated patient turns through Qwen chat, confirming Aria greets + progresses
// the hidden 4-stage plan + keeps replies short (no mic needed).
// Run: node --env-file=.env server/smoke-turnloop.mjs

import { buildLiveInstructions, openingMessageForLanguage } from './generated/orchestration.mjs'

const KEY = process.env.QWEN_API_KEY || process.env.DASHSCOPE_API_KEY
const CHAT_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
const MODEL = process.env.SMOKE_CHAT_MODEL || 'qwen-plus'
const H = { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' }

const system = buildLiveInstructions('demo-patient', 'en', {})
console.log('--- system prompt opening line ---')
console.log('scripted opening:', openingMessageForLanguage('en'))

const messages = [{ role: 'system', content: system }]

// The assistant speaks first (opening). We seed it, then simulate patient turns.
messages.push({ role: 'assistant', content: openingMessageForLanguage('en') })
console.log('\nAria (opening):', openingMessageForLanguage('en'))

const patientTurns = [
  "I'm Margaret, I'm at home in the kitchen.",
  "I slept alright, had breakfast with my daughter.",
  "My daughter reminds me to take my pills every morning.",
]

for (const turn of patientTurns) {
  messages.push({ role: 'user', content: turn })
  console.log('\nPatient:', turn)
  const res = await fetch(CHAT_URL, {
    method: 'POST', headers: H,
    body: JSON.stringify({ model: MODEL, messages, max_tokens: 80, temperature: 0.4 }),
  })
  const body = await res.json()
  if (!res.ok) { console.error('chat error', res.status, JSON.stringify(body).slice(0, 300)); break }
  const reply = body?.choices?.[0]?.message?.content ?? ''
  messages.push({ role: 'assistant', content: reply })
  console.log('Aria:', reply)
}

console.log('\n=== turn-loop smoke done ===')
