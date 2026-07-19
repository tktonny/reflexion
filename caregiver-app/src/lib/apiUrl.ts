export function getApiUrl(path: string) {
  const baseUrl = process.env.EXPO_PUBLIC_CAREGIVER_APP_BACKEND_URL?.trim();

  if (!baseUrl) {
    throw new Error('EXPO_PUBLIC_CAREGIVER_APP_BACKEND_URL is not set');
  }

  const normalizedBaseUrl = baseUrl.replace(/\/+$/, '');
  const normalizedPath = path.replace(/^\/api(?=\/|$)/, '').replace(/^\/?/, '/');
  return `${normalizedBaseUrl}${normalizedPath}`;
}
