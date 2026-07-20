import type { RequestHandler } from 'expo-router/server'

// Post-conversation cognitive-screening JUDGMENT from the check-in transcript.
// Runs server-side (key stays on the server), so it works for all 3 conversation
// versions (relay/http/ws). Research screening aid — NOT a diagnosis.
// Verified prompt/behaviour via server/smoke-assess.mjs.

declare const process: { env: Record<string, string | undefined> }

const SYSTEM = `You are a clinical screening assistant reviewing a short daily voice check-in between "Aria" (assistant) and a "Patient". This is a RESEARCH SCREENING AID, NOT a diagnosis.
Assess cognitive signals PRIMARILY from the transcript: orientation (person/place/time), short-term recall, narrative coherence & sequencing, word-finding/hesitation, and daily-function independence.
If sampled video frames of the patient are provided, use them ONLY for supporting engagement/affect/alertness/attentiveness observations — do NOT infer dementia or any diagnosis from facial appearance, age, or looks. The classification must rest on the conversation.
Be conservative: prefer "needs_observation" over "dementia" when evidence is weak; a normal chat should read "healthy".
Return STRICT JSON only, no markdown, with this exact shape:
{"risk_score": <0..1>, "risk_tier": "low|medium|high", "screening_classification": "healthy|needs_observation|dementia", "summary": "<2-3 sentences>", "findings": ["..."], "evidence_for_risk": ["..."], "evidence_against_risk": ["..."], "visual_observations": ["..."]}
Set "visual_observations" to [] when no frames are provided.`

const MAX_FRAMES = 6

export const OPTIONS: RequestHandler = async () => Response.json({ ok: true })

export const POST: RequestHandler = async (request) => {
  const key = process.env.QWEN_API_KEY || process.env.DASHSCOPE_API_KEY
  if (!key) return Response.json({ success: false, reason: 'missing_qwen_api_key' }, { status: 500 })

  const body = (await request.json().catch(() => null)) as { transcript?: string; language?: string; frames?: string[] } | null
  const transcript = (body?.transcript || '').trim()
  if (!transcript) return Response.json({ success: false, reason: 'empty_transcript' }, { status: 400 })
  const frames = Array.isArray(body?.frames) ? body!.frames.filter((f) => typeof f === 'string' && f).slice(-MAX_FRAMES) : []

  const base = process.env.QWEN_BASE || 'https://dashscope.aliyuncs.com'
  // Text-only -> qwen-plus; with frames -> a vision model (multimodal screening).
  const model = frames.length
    ? process.env.QWEN_VISION_MODEL || 'qwen-vl-max'
    : process.env.QWEN_ASSESS_MODEL || 'qwen-plus'
  const userContent = frames.length
    ? [
        { type: 'text', text: `Transcript:\n${transcript}\n\nSampled video frames of the patient follow.` },
        ...frames.map((url) => ({ type: 'image_url', image_url: { url } })),
      ]
    : `Transcript:\n${transcript}`
  try {
    const res = await fetch(`${base}/compatible-mode/v1/chat/completions`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        temperature: 0.2,
        messages: [
          { role: 'system', content: SYSTEM },
          { role: 'user', content: userContent },
        ],
      }),
    })
    const payload = await res.json()
    if (!res.ok) return Response.json({ success: false, reason: 'assess_failed' }, { status: 502 })
    const text = String(payload?.choices?.[0]?.message?.content ?? '')
    const match = text.match(/\{[\s\S]*\}/)
    const assessment = JSON.parse(match ? match[0] : text)
    return Response.json({ success: true, assessment })
  } catch (error) {
    return Response.json(
      { success: false, reason: error instanceof Error ? error.message : 'unknown_assess_error' },
      { status: 500 },
    )
  }
}
