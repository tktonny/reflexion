/** Canonical consent purposes and public state mapping shared by caregiver and Mirror routes. */
export const DAILY_CHECKIN_CONSENT_PURPOSE = 'home_cognitive_monitoring'
export const RESEARCH_CONSENT_PURPOSE = 'optional_research_participation'

/**
 * Consent for the mirror to use its CAMERA during a conversation, streaming live frames to the model so
 * Aria can see the person she is talking with.
 *
 * Separate from the check-in consent and never implied by it: agreeing to a spoken daily check-in is not
 * agreeing to live video of your living room. It is also independent of the OS camera permission — that
 * is the device asking, this is the person deciding. Absent or withdrawn, every paired mirror runs
 * audio-only from its next configuration read, so withdrawal is the off switch.
 */
export const VIDEO_COMPANION_CONSENT_PURPOSE = 'home_video_companion'

export type ConsentRecordStatus = 'granted' | 'declined' | 'withdrawn'
export type PublicConsentStatus = 'accepted' | 'declined' | 'withdrawn' | 'pending'

export function publicConsentStatus(row: { status?: unknown; withdrawnAt?: unknown } | null | undefined): PublicConsentStatus {
  if (row?.status === 'granted' && !row.withdrawnAt) return 'accepted'
  if (row?.status === 'withdrawn' || row?.withdrawnAt) return 'withdrawn'
  if (row?.status === 'declined') return 'declined'
  return 'pending'
}
