import type { QueryClient } from '@tanstack/react-query';

// Query keys for the caregiver config, kept here because two screens read the SAME endpoint into two
// DIFFERENT shapes and previously shared one key.
//
// `GET /nurse-patient-config/latest` is consumed raw by the dashboard, the Alerts tab and onboarding, but
// Settings normalises it into a SettingsConfig that drops fields the dashboard renders
// (`lastSpokenLabel`, `duration`, `mirrorName`). With one key and staleTime: Infinity, whichever screen
// loaded first won: visiting Settings and then Home left Home rendering a payload it never produced, with
// those fields undefined. Separate keys, invalidated together.

export const caregiverConfigKey = (nurseId?: string | null) => ['latestConfig', nurseId || 'latest'];
export const settingsConfigKey = (nurseId?: string | null) => ['settingsConfig', nurseId || 'latest'];

/** Alerts feed. Keyed per account so one phone shared by two caregivers cannot serve one's feed to the other. */
export const notificationsQueryKey = (nurseId?: string | null) => ['notificationsV1', nurseId || 'anonymous'];

/**
 * Refreshes every view of the caregiver config. Use this instead of invalidating one key by hand — the two
 * shapes come from the same endpoint, so refreshing only one leaves the other stale.
 */
export function invalidateCaregiverConfig(queryClient: QueryClient) {
  return Promise.all([
    queryClient.invalidateQueries({ queryKey: ['latestConfig'] }),
    queryClient.invalidateQueries({ queryKey: ['settingsConfig'] }),
  ]);
}

/**
 * Invalidate AND force a refetch of both shapes, including queries whose screen is not currently mounted.
 *
 * Onboarding needs this: it navigates straight to the dashboard or Settings after adding a loved one, and
 * with `staleTime: Infinity` an invalidation alone would not refill a screen that is about to mount from
 * cache — the caregiver would arrive at a list missing the person they just added.
 */
export async function refreshCaregiverConfig(queryClient: QueryClient) {
  await invalidateCaregiverConfig(queryClient);
  await Promise.all([
    queryClient.refetchQueries({ queryKey: ['latestConfig'], type: 'all' }),
    queryClient.refetchQueries({ queryKey: ['settingsConfig'], type: 'all' }),
  ]);
}

/**
 * Drops every cached response. Call on sign-out AND before a sign-in completes.
 *
 * React Query is configured with `gcTime: Infinity` (queryClient.ts), so nothing expires on its own. Signing
 * out cleared the tokens but left the cache intact, which meant the next caregiver to use the same phone
 * could read the previous one's data — loved ones' names, their check-in status, their alerts — until every
 * screen happened to refetch. `['notificationsV1']` had no account in its key at all, so it was literally
 * the same cache entry for both people.
 */
export function clearCaregiverCache(queryClient: QueryClient) {
  queryClient.clear();
}
