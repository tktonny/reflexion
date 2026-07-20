import type { RequestHandler } from 'expo-router/server'

// Post-conversation cognitive-screening JUDGMENT. Runs server-side (key stays on the server), so it
// works for all 3 conversation versions (relay/http/ws). Research screening aid — NOT a diagnosis.
//
// Two-stage so the scoring model never sees the pixels (the classification is structurally forced
// to rest on the conversation, not on facial appearance):
//   Stage 1 (vision, qwen-vl-max): frames -> engagement/affect notes only.
//   Stage 2 (text,   qwen-plus):   transcript (+ notes as support) -> risk/classification.
// Verified prompt/behaviour via server/smoke-assess.mjs and server/smoke-vision.mjs.

declare const process: { env: Record<string, string | undefined> }

// Stage 2 — classification (text only).
const SYSTEM = `You are a clinical screening assistant reviewing a short daily voice check-in between "Aria" (assistant) and a "Patient". This is a RESEARCH SCREENING AID, NOT a diagnosis.
Assess cognitive signals from the transcript: orientation (person/place/time), short-term recall, narrative coherence & sequencing, word-finding/hesitation, and daily-function independence.
You may also receive "Visual observations" (engagement/affect/alertness notes). Treat them ONLY as supporting context — they MUST NOT drive the classification, which must rest on the conversation. Never infer dementia from appearance or age.
Be conservative: prefer "needs_observation" over "dementia" when evidence is weak; a normal chat should read "healthy".
Return STRICT JSON only, no markdown, with this exact shape:
{"risk_score": <0..1>, "risk_tier": "low|medium|high", "screening_classification": "healthy|needs_observation|dementia", "summary": "<2-3 sentences>", "findings": ["..."], "evidence_for_risk": ["..."], "evidence_against_risk": ["..."]}`

// Stage 1 — vision. Engagement/affect ONLY; no diagnosis, age, or classification.
const VISION_OBS_SYSTEM = `You are given sampled video frames of a person during a short check-in. Describe ONLY their engagement, affect, alertness, and attentiveness as brief factual observations. Do NOT infer age, health, cognition, dementia, or any diagnosis from appearance. Return STRICT JSON only, no markdown: {"visual_observations": ["..."]}`

const MAX_FRAMES = 6
const TIMEOUT_MS = 45000

export const OPTIONS: RequestHandler = async () => Response.json({ ok: true })

export const POST: RequestHandler = async (request) => {
  const key = process.env.QWEN_API_KEY || process.env.DASHSCOPE_API_KEY
  if (!key) return Response.json({ success: false, reason: 'missing_qwen_api_key' }, { status: 500 })

  const body = (await request.json().catch(() => null)) as { transcript?: string; language?: string; frames?: string[] } | null
  const transcript = (body?.transcript || '').trim()
  if (!transcript) return Response.json({ success: false, reason: 'empty_transcript' }, { status: 400 })
  const frames = Array.isArray(body?.frames) ? body!.frames.filter((f) => typeof f === 'string' && f).slice(-MAX_FRAMES) : []

  const base = process.env.QWEN_BASE || 'https://dashscope.aliyuncs.com'
  const url = `${base}/compatible-mode/v1/chat/completions`

  // Calls DashScope with a timeout so the request can't hang the caller. Returns assistant text.
  async function chat(model: string, messages: unknown[]): Promise<string> {
    const res = await fetch(url, {
      method: 'POST',
      headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, temperature: 0.2, messages }),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    })
    const payload = await res.json()
    if (!res.ok) throw new Error(`assess_upstream_${res.status}`)
    return String(payload?.choices?.[0]?.message?.content ?? '')
  }
  const parseJson = (text: string) => {
    const match = text.match(/\{[\s\S]*\}/)
    return JSON.parse(match ? match[0] : text)
  }

  try {
    // Stage 1 (vision): frames -> visual_observations only. Failure degrades to text-only.
    let visualObservations: string[] = []
    if (frames.length) {
      try {
        const vModel = process.env.QWEN_VISION_MODEL || 'qwen-vl-max'
        const vText = await chat(vModel, [
          { role: 'system', content: VISION_OBS_SYSTEM },
          {
            role: 'user',
            content: [
              { type: 'text', text: 'Frames of the patient during the check-in follow.' },
              ...frames.map((url) => ({ type: 'image_url', image_url: { url } })),
            ],
          },
        ])
        const parsed = parseJson(vText)
        if (Array.isArray(parsed?.visual_observations)) visualObservations = parsed.visual_observations.map(String)
      } catch {
        /* vision unavailable/oversized -> proceed transcript-only (transcript is PRIMARY) */
      }
    }

    // Stage 2 (text): classification from transcript (+ observations as support). No pixels.
    const model = process.env.QWEN_ASSESS_MODEL || 'qwen-plus'
    const userText = visualObservations.length
      ? `Transcript:\n${transcript}\n\nVisual observations (supporting context only; do NOT let these drive the classification):\n${visualObservations.map((o) => `- ${o}`).join('\n')}`
      : `Transcript:\n${transcript}`
    const text = await chat(model, [
      { role: 'system', content: SYSTEM },
      { role: 'user', content: userText },
    ])
    const assessment = parseJson(text)
    assessment.visual_observations = visualObservations
    return Response.json({ success: true, assessment })
  } catch (error) {
    return Response.json(
      { success: false, reason: error instanceof Error ? error.message : 'unknown_assess_error' },
      { status: 502 },
    )
  }
}
