/** Canonical consent purposes and public state mapping shared by caregiver and Mirror routes. */
export const DAILY_CHECKIN_CONSENT_PURPOSE = 'home_cognitive_monitoring'
export const RESEARCH_CONSENT_PURPOSE = 'optional_research_participation'

export type ConsentRecordStatus = 'granted' | 'declined' | 'withdrawn'
export type PublicConsentStatus = 'accepted' | 'declined' | 'withdrawn' | 'pending'

export function publicConsentStatus(row: { status?: unknown; withdrawnAt?: unknown } | null | undefined): PublicConsentStatus {
  if (row?.status === 'granted' && !row.withdrawnAt) return 'accepted'
  if (row?.status === 'withdrawn' || row?.withdrawnAt) return 'withdrawn'
  if (row?.status === 'declined') return 'declined'
  return 'pending'
}
