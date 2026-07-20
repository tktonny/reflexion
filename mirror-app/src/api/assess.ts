import { getApiUrl } from '../../app/apiUrl'
import { qwenChat, qwenVisionChat, fetchWithTimeout, type QwenContentPart } from './qwenClient'
import type { ChatMessage } from '../hooks/conversationTypes'

// Two-stage screening (shared shape with app/api/assess+api.ts). The scoring model NEVER sees the
// pixels — Stage 1 (vision) turns frames into engagement/affect notes, Stage 2 (text) does the
// classification from the transcript + those notes. This structurally enforces "classification
// rests on the conversation", not just a prompt request.
//
// Stage 2 — classification (text only). Transcript is PRIMARY; visual observations are support only.
export const SCREENING_SYSTEM = `You are a clinical screening assistant reviewing a short daily voice check-in between "Aria" (assistant) and a "Patient". This is a RESEARCH SCREENING AID, NOT a diagnosis.
Assess cognitive signals from the transcript: orientation (person/place/time), short-term recall, narrative coherence & sequencing, word-finding/hesitation, and daily-function independence.
You may also receive "Visual observations" (engagement/affect/alertness notes). Treat them ONLY as supporting context — they MUST NOT drive the classification, which must rest on the conversation. Never infer dementia from appearance or age.
Be conservative: prefer "needs_observation" over "dementia" when evidence is weak; a normal chat should read "healthy".
Return STRICT JSON only, no markdown, with this exact shape:
{"risk_score": <0..1>, "risk_tier": "low|medium|high", "screening_classification": "healthy|needs_observation|dementia", "summary": "<2-3 sentences>", "findings": ["..."], "evidence_for_risk": ["..."], "evidence_against_risk": ["..."]}`

// Stage 1 — vision. Engagement/affect ONLY; no diagnosis, age, or classification.
export const VISION_OBS_SYSTEM = `You are given sampled video frames of a person during a short check-in. Describe ONLY their engagement, affect, alertness, and attentiveness as brief factual observations. Do NOT infer age, health, cognition, dementia, or any diagnosis from appearance. Return STRICT JSON only, no markdown: {"visual_observations": ["..."]}`

// Bound the payload/cost of the multimodal call (matches the client-side sampler cap).
export const MAX_ASSESS_FRAMES = 6

export type ScreeningAssessment = {
  risk_score: number | null
  risk_tier: 'low' | 'medium' | 'high' | null
  screening_classification: 'healthy' | 'needs_observation' | 'dementia' | null
  summary: string
  findings: string[]
  evidence_for_risk: string[]
  evidence_against_risk: string[]
  visual_observations?: string[]
}

export type AssessResponse =
  | { success: true; assessment: ScreeningAssessment }
  | { success: false; reason: string }

/** Build a "Aria:/Patient:" transcript from the chat messages (drops system lines). */
export function transcriptFromMessages(messages: ChatMessage[]): string {
  return messages
    .filter((m) => m.role === 'assistant' || m.role === 'user')
    .map((m) => `${m.role === 'assistant' ? 'Aria' : 'Patient'}: ${m.text.trim()}`)
    .join('\n')
}

/**
 * Return the screening judgment. Prefers the server endpoint /api/assess (key + patient frames
 * stay server-side). On a RESOLVED-but-failed server response we retry the SAME route text-only
 * (dropping frames) so a frames-induced failure never wipes out the primary transcript screening —
 * and we never fall back to the client-direct path there, since that would send the patient's face
 * frames straight to DashScope and defeat the relay's PII/key isolation. Only a genuine fetch
 * failure (no reachable backend — e.g. a standalone APK) uses the client-direct path.
 */
export async function assessConversation(
  transcript: string,
  language = 'en',
  frames: string[] = [],
): Promise<AssessResponse> {
  const capped = frames.slice(-MAX_ASSESS_FRAMES)
  const attempts = capped.length ? [capped, [] as string[]] : [[] as string[]]
  try {
    for (const payload of attempts) {
      const res = await fetchWithTimeout(getApiUrl('/api/assess'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript, language, frames: payload }),
      })
      if (res.ok) {
        const body = (await res.json()) as AssessResponse
        if (body.success) return body
      }
      // reachable but non-OK / unsuccessful (e.g. 413 from frames, 502 from the model):
      // the loop retries text-only; if that also fails we surface a server error below.
    }
    return { success: false, reason: 'assess_server_failed' }
  } catch {
    // Genuine fetch failure = no reachable backend (offline / standalone APK). Client-direct is the
    // intended path there; frames going direct is acceptable because there is no relay to protect.
    return assessDirect(transcript, capped)
  }
}

async function assessDirect(transcript: string, frames: string[]): Promise<AssessResponse> {
  // Stage 1 (vision): frames -> engagement/affect notes only. Failure degrades to text-only.
  const visualObservations = frames.length > 0 ? await visualObservationsDirect(frames) : []
  // Stage 2 (text): classification from transcript (+ notes as support). Transcript is PRIMARY.
  try {
    const userText = visualObservations.length
      ? `Transcript:\n${transcript}\n\nVisual observations (supporting context only; do NOT let these drive the classification):\n${visualObservations.map((o) => `- ${o}`).join('\n')}`
      : `Transcript:\n${transcript}`
    const text = await qwenChat(
      [
        { role: 'system', content: SCREENING_SYSTEM },
        { role: 'user', content: userText },
      ],
      { maxTokens: 700, temperature: 0.2 },
    )
    const match = text.match(/\{[\s\S]*\}/)
    const assessment = JSON.parse(match ? match[0] : text) as ScreeningAssessment
    assessment.visual_observations = visualObservations
    return { success: true, assessment }
  } catch (error) {
    return { success: false, reason: error instanceof Error ? error.message : 'assess_failed' }
  }
}

async function visualObservationsDirect(frames: string[]): Promise<string[]> {
  try {
    const parts: QwenContentPart[] = [{ type: 'text', text: 'Frames of the patient during the check-in follow.' }]
    for (const url of frames) parts.push({ type: 'image_url', image_url: { url } })
    const text = await qwenVisionChat(VISION_OBS_SYSTEM, parts, { maxTokens: 300, temperature: 0.2 })
    const match = text.match(/\{[\s\S]*\}/)
    const parsed = JSON.parse(match ? match[0] : text)
    return Array.isArray(parsed?.visual_observations) ? parsed.visual_observations.map(String) : []
  } catch {
    return [] // vision unavailable -> transcript-only classification still proceeds
  }
}
