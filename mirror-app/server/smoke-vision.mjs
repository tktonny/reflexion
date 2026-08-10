// Verify the TWO-STAGE multimodal screening against real Qwen (mirrors app/api/assess+api.ts):
//   Stage 1 (vision, qwen-vl-max): frames -> {visual_observations} ONLY (no classification).
//   Stage 2 (text,  qwen-plus):    transcript + observations -> full screening JSON.
// The scoring model never sees the pixels. Run: node --env-file=.env server/smoke-vision.mjs
import { readFileSync } from 'node:fs'

const KEY = process.env.QWEN_API_KEY || process.env.DASHSCOPE_API_KEY
const BASE = process.env.QWEN_BASE || 'https://dashscope.aliyuncs.com'
const URL = `${BASE}/compatible-mode/v1/chat/completions`
const VISION_MODEL = process.env.QWEN_VISION_MODEL || 'qwen-vl-max'
const TEXT_MODEL = process.env.QWEN_ASSESS_MODEL || 'qwen-plus'
const H = { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' }

const VISION_OBS_SYSTEM = `You are given sampled video frames of a person during a short check-in. Describe ONLY their engagement, affect, alertness, and attentiveness as brief factual observations. Do NOT infer age, health, cognition, dementia, or any diagnosis from appearance. Return STRICT JSON only, no markdown: {"visual_observations": ["..."]}`
const SCREENING_SYSTEM = `You are a clinical screening assistant reviewing a short daily voice check-in. RESEARCH SCREENING AID, NOT a diagnosis. Assess from the transcript: orientation, recall, coherence, word-finding, daily-function. "Visual observations" are SUPPORTING context only and MUST NOT drive the classification. Return STRICT JSON only: {"risk_score": <0..1>, "risk_tier": "low|medium|high", "screening_classification": "healthy|needs_observation|dementia", "summary": "...", "findings": ["..."], "evidence_for_risk": ["..."], "evidence_against_risk": ["..."]}`

const imgPath = process.argv[2] || 'node_modules/expo-router/assets/unmatched.png'
const dataUrl = `data:image/png;base64,${readFileSync(imgPath).toString('base64')}`
const transcript = 'Aria: Good morning! Where are you right now?\nPatient: I am at home in the kitchen, just had breakfast with my daughter.'

async function chat(model, messages) {
  const res = await fetch(URL, { method: 'POST', headers: H, body: JSON.stringify({ model, temperature: 0.2, messages }) })
  const body = await res.json()
  if (!res.ok) throw new Error(`${model} ${res.status}: ${JSON.stringify(body).slice(0, 200)}`)
  const text = String(body?.choices?.[0]?.message?.content ?? '')
  const m = text.match(/\{[\s\S]*\}/)
  return JSON.parse(m ? m[0] : text)
}

let pass = true
try {
  console.log(`Stage 1 (vision ${VISION_MODEL}) ...`)
  const stage1 = await chat(VISION_MODEL, [
    { role: 'system', content: VISION_OBS_SYSTEM },
    { role: 'user', content: [{ type: 'text', text: 'Frames follow.' }, { type: 'image_url', image_url: { url: dataUrl } }] },
  ])
  const obs = Array.isArray(stage1?.visual_observations) ? stage1.visual_observations : null
  const stage1NoClass = !('screening_classification' in stage1) && !('risk_score' in stage1)
  console.log('  visual_observations:', obs)
  console.log('  stage-1 emits NO classification/score:', stage1NoClass ? 'PASS' : 'FAIL (leaked scoring into vision stage)')
  if (!obs || !stage1NoClass) pass = false

  console.log(`\nStage 2 (text ${TEXT_MODEL}) ...`)
  const userText = obs && obs.length
    ? `Transcript:\n${transcript}\n\nVisual observations (support only):\n${obs.map((o) => `- ${o}`).join('\n')}`
    : `Transcript:\n${transcript}`
  const stage2 = await chat(TEXT_MODEL, [
    { role: 'system', content: SCREENING_SYSTEM },
    { role: 'user', content: userText },
  ])
  console.log('  classification:', stage2?.screening_classification, '· risk', stage2?.risk_score)
  const okShape = 'screening_classification' in stage2 && 'risk_score' in stage2 && Array.isArray(stage2?.findings)
  if (!okShape) pass = false
  console.log('  stage-2 full screening shape:', okShape ? 'PASS' : 'FAIL')
} catch (e) {
  console.log('ERROR:', e?.message)
  pass = false
}
console.log(`\nRESULT: ${pass ? 'PASS' : 'FAIL'} — two-stage multimodal screening (vision obs -> text classification)`)
process.exit(pass ? 0 : 1)
