export type ConversationConsentStatus = 'accepted' | 'declined' | 'withdrawn' | 'pending' | null | undefined

/** Daily check-ins are server-gated and may only start after the older adult has accepted consent. */
export function canStartDailyConversation(status: ConversationConsentStatus): boolean {
  return status === 'accepted'
}

/**
 * The backend remains the consent authority. If consent changes after the screen loaded, session
 * creation returns one of these stable codes; route back to the consent choice instead of presenting it
 * as an Aria/service outage.
 */
export function isConversationConsentError(message: string): boolean {
  return /^(?:CONSENT_REQUIRED|OLDER_ADULT_CONSENT_REQUIRED)$/i.test(message.trim())
}
