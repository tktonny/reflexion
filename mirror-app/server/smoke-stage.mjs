// Assert the redesigned orchestration drives all stages + ALWAYS does recall, against real Qwen.
// Simulates the v2 turn loop exactly (Layer 1 prompt + Layer 2 recall-floor via recallBudgetStep).
// Run: node --env-file=.env server/smoke-stage.mjs  (exit 0 = PASS)

import {
  buildLiveInstructions, openingMessageForLanguage,
  recallBudgetStep, RECALL_DIRECTIVE, WRAPUP_DIRECTIVE, looksLikeRecallProbe,
} from './generated/orchestration.mjs'

const KEY = process.env.QWEN_API_KEY || process.env.DASHSCOPE_API_KEY
const CHAT_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
const MODEL = process.env.SMOKE_CHAT_MODEL || 'qwen-plus'
const H = { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' }

async function chat(messages) {
  const res = await fetch(CHAT_URL, { method: 'POST', headers: H, body: JSON.stringify({ model: MODEL, messages, max_tokens: 120, temperature: 0.4 }) })
  const body = await res.json()
  if (!res.ok) throw new Error(`chat ${res.status}: ${JSON.stringify(body).slice(0, 200)}`)
  return String(body?.choices?.[0]?.message?.content ?? '').trim()
}

const system = buildLiveInstructions('demo-patient', 'en', {})
const llm = [{ role: 'system', content: system }]
const opening = openingMessageForLanguage('en')
llm.push({ role: 'assistant', content: opening })
console.log('Aria (opening):', opening)

// Cooperative patient; turns 4/5 are generic so the recall floor + answer are exercised.
const patientTurns = [
  "I'm Tony, I'm at home in Shanghai.",
  "It's been good — I woke up early and had congee with my wife.",
  "My wife reminds me about my pills, and I keep appointments in a little notebook.",
  "Sure, happy to.",
  "Yes, I had congee with my wife this morning.",
]

let turnCount = 0
let recallProbeIssued = false
let recallAnswered = false
const ariaReplies = []

for (const turn of patientTurns) {
  console.log('\nPatient:', turn)
  llm.push({ role: 'user', content: turn })
  turnCount += 1
  const step = recallBudgetStep({ turnCount, recallProbeIssued, recallAnswered })
  let directive = null
  if (step.action === 'force_recall') { recallProbeIssued = true; directive = RECALL_DIRECTIVE; console.log('   [floor] force_recall') }
  else if (step.action === 'wrap_up') { recallAnswered = true; directive = WRAPUP_DIRECTIVE; console.log('   [floor] wrap_up') }
  const msgs = directive ? [...llm, { role: 'system', content: directive }] : llm
  const reply = await chat(msgs)
  llm.push({ role: 'assistant', content: reply })
  ariaReplies.push(reply)
  if (!recallProbeIssued && looksLikeRecallProbe(reply)) { recallProbeIssued = true; console.log('   [soft] model self-initiated recall') }
  console.log('Aria:', reply)
  if (step.action === 'wrap_up') break
}

// --- assertions ---
const all = ariaReplies.join('\n').toLowerCase()
const dailyFn = /meal|medicine|medication|pill|appointment|keep track|remind/i.test(all)
const recall = ariaReplies.some((r) => looksLikeRecallProbe(r))
const drift = ariaReplies.some((r) => /\b(weather|coke|hobby|favou?rite (drink|food|spot))\b\?/i.test(r))

console.log('\n=== assertions ===')
console.log('topic3 daily_function asked:', dailyFn ? 'PASS' : 'FAIL')
console.log('recall probe happened:', recall ? 'PASS' : 'FAIL')
console.log('off-agenda small-talk question:', drift ? 'WARN (drift seen)' : 'none')
const pass = dailyFn && recall
console.log(`\nRESULT: ${pass ? 'PASS' : 'FAIL'}`)
process.exit(pass ? 0 : 1)
