import { getApiUrl } from '../../app/apiUrl'

export const API_BASE_URL = 'Expo Router app/api'

export async function apiPost<TResponse>(path: string, body: unknown): Promise<TResponse> {
  const response = await fetch(getApiUrl(path), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  })

  const payload = (await response.json()) as TResponse
  if (!response.ok) {
    return payload
  }

  return payload
}
