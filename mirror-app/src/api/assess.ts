import { getApiUrl } from '../../app/apiUrl'
import { qwenChat, qwenVisionChat, type QwenContentPart } from './qwenClient'
import type { ChatMessage } from '../hooks/conversationTypes'

// Shared with app/api/assess+api.ts. Transcript is PRIMARY; frames add engagement/affect/alertness
// context only — never infer dementia from facial appearance.
export const SCREENING_SYSTEM = `You are a clinical screening assistant reviewing a short daily voice check-in between "Aria" (assistant) and a "Patient". This is a RESEARCH SCREENING AID, NOT a diagnosis.
Assess cognitive signals PRIMARILY from the transcript: orientation (person/place/time), short-term recall, narrative coherence & sequencing, word-finding/hesitation, and daily-function independence.
If sampled video frames of the patient are provided, use them ONLY for supporting engagement/affect/alertness/attentiveness observations — do NOT infer dementia or any diagnosis from facial appearance, age, or looks. The classification must rest on the conversation.
Be conservative: prefer "needs_observation" over "dementia" when evidence is weak; a normal chat should read "healthy".
Return STRICT JSON only, no markdown, with this exact shape:
{"risk_score": <0..1>, "risk_tier": "low|medium|high", "screening_classification": "healthy|needs_observation|dementia", "summary": "<2-3 sentences>", "findings": ["..."], "evidence_for_risk": ["..."], "evidence_against_risk": ["..."], "visual_observations": ["..."]}
Set "visual_observations" to [] when no frames are provided.`

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
 * Return the screening judgment for a transcript.
 * Prefers the server endpoint /api/assess (key stays server-side). If that isn't reachable
 * (e.g. a standalone APK with no hosted backend), falls back to a direct client call
 * (kiosk/own-device only — uses the client key).
 */
export async function assessConversation(
  transcript: string,
  language = 'en',
  frames: string[] = [],
): Promise<AssessResponse> {
  const capped = frames.slice(-MAX_ASSESS_FRAMES)
  try {
    const res = await fetch(getApiUrl('/api/assess'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transcript, language, frames: capped }),
    })
    if (res.ok) {
      const body = (await res.json()) as AssessResponse
      if (body.success) return body
    }
  } catch {
    /* server route unreachable — fall through to direct client call */
  }
  return assessDirect(transcript, capped)
}

async function assessDirect(transcript: string, frames: string[]): Promise<AssessResponse> {
  try {
    let text: string
    if (frames.length > 0) {
      // Multimodal: transcript text + sampled frames -> vision model.
      const parts: QwenContentPart[] = [{ type: 'text', text: `Transcript:\n${transcript}\n\nSampled video frames of the patient follow.` }]
      for (const url of frames) parts.push({ type: 'image_url', image_url: { url } })
      text = await qwenVisionChat(SCREENING_SYSTEM, parts, { maxTokens: 800, temperature: 0.2 })
    } else {
      text = await qwenChat(
        [
          { role: 'system', content: SCREENING_SYSTEM },
          { role: 'user', content: `Transcript:\n${transcript}` },
        ],
        { maxTokens: 700, temperature: 0.2 },
      )
    }
    const match = text.match(/\{[\s\S]*\}/)
    return { success: true, assessment: JSON.parse(match ? match[0] : text) as ScreeningAssessment }
  } catch (error) {
    return { success: false, reason: error instanceof Error ? error.message : 'assess_failed' }
  }
}
