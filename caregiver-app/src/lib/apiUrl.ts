// Both URL builders for the backend, deliberately side by side.
//
// They are NOT interchangeable, and that is the app's sharpest footgun (CLAUDE.md calls it out): the two
// API surfaces are mounted differently on the server, so the legacy builder STRIPS a leading `/api` while
// the v1 builder APPENDS `/api/v1`. Passing a path to the wrong one produces a 404 — which this app used to
// render to caregivers as the headline "Not found". Keeping them in one file makes the asymmetry visible
// instead of leaving it to be rediscovered, and keeps them free of React Native imports so they stay
// unit-testable (see apiUrl.test.ts).

function requireBaseUrl(): string {
  const baseUrl = process.env.EXPO_PUBLIC_CAREGIVER_APP_BACKEND_URL?.trim();

  if (!baseUrl) {
    // Fail loudly. An APK built without this must not quietly succeed at reaching nothing.
    throw new Error('EXPO_PUBLIC_CAREGIVER_APP_BACKEND_URL is not set');
  }

  return baseUrl.replace(/\/+$/, '');
}

/**
 * Legacy API (sunset 2026-12-31). Those routes are mounted bare at the server root, so a leading `/api`
 * on the caller's path is removed: `/api/auth/sign-in` -> `https://host/auth/sign-in`.
 */
export function getApiUrl(path: string) {
  const normalizedPath = path.replace(/^\/api(?=\/|$)/, '').replace(/^\/?/, '/');
  return `${requireBaseUrl()}${normalizedPath}`;
}

/**
 * Canonical v1 API. Mounted under `/api/v1`, so the segment is appended:
 * `/notifications` -> `https://host/api/v1/notifications`. A base URL that already ends in `/api` is not
 * doubled. All v1 responses are enveloped as `{ data, meta }`; errors as `{ error: { code, message } }`.
 */
export function getV1Url(path: string): string {
  const base = requireBaseUrl();
  const apiBase = /\/api$/.test(base) ? base : `${base}/api`;
  const normalizedPath = path.replace(/^\/?/, '/');
  return `${apiBase}/v1${normalizedPath}`;
}
