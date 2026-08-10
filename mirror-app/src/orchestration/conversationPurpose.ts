import type { ChatMessage } from '../hooks/conversationTypes'

export type ConversationPersona = 'screening' | 'companion'

/** Daily-assistant transcripts are retained for continuity, but only a real screening is scored. */
export function isCognitiveAssessmentEligible(
  persona: ConversationPersona,
  messages: ChatMessage[],
): boolean {
  return persona === 'screening' && messages.some(
    (message) => message.role === 'user' && message.text.trim().length > 0,
  )
}
