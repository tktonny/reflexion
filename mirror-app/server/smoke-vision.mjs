// Verify the MULTIMODAL screening call shape against real Qwen: send a transcript + a base64 image
// (data URL) to a vision model via DashScope compatible-mode and confirm it returns strict JSON.
// De-risks the video-input feature before wiring the UI. Run: node --env-file=.env server/smoke-vision.mjs
import { readFileSync } from 'node:fs'

const KEY = process.env.QWEN_API_KEY || process.env.DASHSCOPE_API_KEY
const BASE = process.env.QWEN_BASE || 'https://dashscope.aliyuncs.com'
const CANDIDATES = (process.env.QWEN_VISION_MODEL ? [process.env.QWEN_VISION_MODEL] : ['qwen-vl-max', 'qwen-vl-plus', 'qwen2.5-vl-72b-instruct'])
const H = { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' }

const imgPath = process.argv[2] || 'node_modules/expo-router/assets/unmatched.png'
const b64 = readFileSync(imgPath).toString('base64')
const dataUrl = `data:image/png;base64,${b64}`
console.log(`image: ${imgPath} (${b64.length} b64 chars)`)

// Mirrors src/api/assess.ts SCREENING_SYSTEM (full 7-field shape) to verify the model fills it
// correctly when frames are attached.
const SYSTEM = `You are a clinical screening assistant reviewing a short daily voice check-in between "Aria" (assistant) and a "Patient". This is a RESEARCH SCREENING AID, NOT a diagnosis.
Assess cognitive signals PRIMARILY from the transcript: orientation, short-term recall, narrative coherence, word-finding, and daily-function independence.
If sampled video frames of the patient are provided, use them ONLY for supporting engagement/affect/alertness observations — do NOT infer dementia from facial appearance. The classification must rest on the conversation.
Return STRICT JSON only, no markdown: {"risk_score": <0..1>, "risk_tier": "low|medium|high", "screening_classification": "healthy|needs_observation|dementia", "summary": "<2-3 sentences>", "findings": ["..."], "evidence_for_risk": ["..."], "evidence_against_risk": ["..."], "visual_observations": ["..."]}`

async function tryModel(model) {
  const res = await fetch(`${BASE}/compatible-mode/v1/chat/completions`, {
    method: 'POST', headers: H,
    body: JSON.stringify({
      model, temperature: 0.2,
      messages: [
        { role: 'system', content: SYSTEM },
        { role: 'user', content: [
          { type: 'text', text: 'Transcript:\nAria: Good morning! Where are you right now?\nPatient: I am at home in the kitchen.\n\nFrames follow.' },
          { type: 'image_url', image_url: { url: dataUrl } },
        ] },
      ],
    }),
  })
  const body = await res.json()
  if (!res.ok) return { ok: false, status: res.status, reason: JSON.stringify(body).slice(0, 200) }
  const text = String(body?.choices?.[0]?.message?.content ?? '')
  return { ok: true, text }
}

let passed = false
for (const model of CANDIDATES) {
  process.stdout.write(`\n--- model: ${model} --- `)
  try {
    const r = await tryModel(model)
    if (!r.ok) { console.log(`FAIL (${r.status}) ${r.reason}`); continue }
    console.log('OK\n' + r.text)
    const match = r.text.match(/\{[\s\S]*\}/)
    const obj = JSON.parse(match ? match[0] : r.text)
    const okShape = 'screening_classification' in obj && Array.isArray(obj.visual_observations)
    if (!okShape) { console.log('  (shape missing screening_classification / visual_observations[])'); continue }
    console.log(`\nRESULT: PASS — ${model}: image_url base64 accepted, full screening shape returned, visual_observations=${obj.visual_observations.length}`)
    passed = true
    break
  } catch (e) { console.log('error parsing/among request:', e?.message) }
}
process.exit(passed ? 0 : 1)
