import { getApiUrl } from '../../app/apiUrl'
import { qwenChat } from './qwenClient'
import type { ChatMessage } from '../hooks/conversationTypes'

const SCREENING_SYSTEM = `You are a clinical screening assistant reviewing a short daily voice check-in transcript between "Aria" (assistant) and a "Patient". This is a RESEARCH SCREENING AID, NOT a diagnosis.
Assess cognitive signals ONLY from the transcript: orientation (person/place/time), short-term recall, narrative coherence & sequencing, word-finding/hesitation, and daily-function independence.
Be conservative: prefer "needs_observation" over "dementia" when evidence is weak; a normal chat should read "healthy".
Return STRICT JSON only, no markdown, with this exact shape:
{"risk_score": <0..1>, "risk_tier": "low|medium|high", "screening_classification": "healthy|needs_observation|dementia", "summary": "<2-3 sentences>", "findings": ["..."], "evidence_for_risk": ["..."], "evidence_against_risk": ["..."]}`

export type ScreeningAssessment = {
  risk_score: number | null
  risk_tier: 'low' | 'medium' | 'high' | null
  screening_classification: 'healthy' | 'needs_observation' | 'dementia' | null
  summary: string
  findings: string[]
  evidence_for_risk: string[]
  evidence_against_risk: string[]
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
export async function assessConversation(transcript: string, language = 'en'): Promise<AssessResponse> {
  try {
    const res = await fetch(getApiUrl('/api/assess'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transcript, language }),
    })
    if (res.ok) {
      const body = (await res.json()) as AssessResponse
      if (body.success) return body
    }
  } catch {
    /* server route unreachable — fall through to direct client call */
  }
  return assessDirect(transcript)
}

async function assessDirect(transcript: string): Promise<AssessResponse> {
  try {
    const text = await qwenChat(
      [
        { role: 'system', content: SCREENING_SYSTEM },
        { role: 'user', content: `Transcript:\n${transcript}` },
      ],
      { maxTokens: 700, temperature: 0.2 },
    )
    const match = text.match(/\{[\s\S]*\}/)
    return { success: true, assessment: JSON.parse(match ? match[0] : text) as ScreeningAssessment }
  } catch (error) {
    return { success: false, reason: error instanceof Error ? error.message : 'assess_failed' }
  }
}
