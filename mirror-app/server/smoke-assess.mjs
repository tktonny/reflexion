// Verify the post-conversation JUDGMENT: feed a check-in transcript to Qwen and get a
// structured cognitive-screening assessment (risk score/tier/classification + evidence).
// This is what the test screen will show after a session ("是否给出准确的判断结果").
// Run: node --env-file=.env server/smoke-assess.mjs

const KEY = process.env.QWEN_API_KEY || process.env.DASHSCOPE_API_KEY
const CHAT_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
const MODEL = process.env.SMOKE_ASSESS_MODEL || 'qwen-plus'
const H = { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' }

const SYSTEM = `You are a clinical screening assistant reviewing a short daily voice check-in transcript between "Aria" (assistant) and a "Patient". This is a RESEARCH SCREENING AID, NOT a diagnosis.
Assess cognitive signals ONLY from the transcript: orientation (person/place/time), short-term recall, narrative coherence & sequencing, word-finding/hesitation, and daily-function independence.
Be conservative: prefer "needs_observation" over "dementia" when evidence is weak; a normal chat should read "healthy".
Return STRICT JSON only, no markdown, with this exact shape:
{"risk_score": <0..1>, "risk_tier": "low|medium|high", "screening_classification": "healthy|needs_observation|dementia", "summary": "<2-3 sentences>", "findings": ["..."], "evidence_for_risk": ["..."], "evidence_against_risk": ["..."]}`

// Two canned transcripts: one coherent/healthy, one with memory/orientation issues.
const transcripts = {
  healthy: [
    'Aria: Hi, nice to meet you. What should I call you? And where are you right now?',
    "Patient: I'm Margaret. I'm at home, in my kitchen in Singapore.",
    "Aria: How's your day been so far?",
    'Patient: Good. I woke at seven, had toast and kaya with my daughter, then watered the orchids on the balcony.',
    'Aria: On a usual day, how do you keep track of meals and medicines?',
    'Patient: I take my blood pressure pill after breakfast. I keep it in a weekly box so I know if I missed one.',
  ].join('\n'),
  impaired: [
    'Aria: Hi, nice to meet you. What should I call you? And where are you right now?',
    "Patient: I'm... I'm not sure. Somewhere. Is this the hospital?",
    "Aria: How's your day been so far?",
    "Patient: I don't... I had breakfast, I think. Or was that yesterday? I can't remember.",
    'Aria: On a usual day, how do you keep track of meals and medicines?',
    "Patient: My... the girl, she does it. I don't know. What did you ask me?",
  ].join('\n'),
}

for (const [name, transcript] of Object.entries(transcripts)) {
  console.log(`\n=== ASSESS: ${name} ===`)
  const res = await fetch(CHAT_URL, {
    method: 'POST', headers: H,
    body: JSON.stringify({ model: MODEL, temperature: 0.2, messages: [
      { role: 'system', content: SYSTEM },
      { role: 'user', content: `Transcript:\n${transcript}` },
    ] }),
  })
  const body = await res.json()
  console.log('status', res.status)
  const text = body?.choices?.[0]?.message?.content ?? JSON.stringify(body).slice(0, 200)
  // lenient JSON extract
  const m = String(text).match(/\{[\s\S]*\}/)
  try { console.log(JSON.stringify(JSON.parse(m ? m[0] : text), null, 2)) }
  catch { console.log('raw:', String(text).slice(0, 500)) }
}
console.log('\n=== assess smoke done ===')
