import AsyncStorage from '@react-native-async-storage/async-storage'

import { getApiUrl } from '../../app/apiUrl'
import { QWEN } from '../config/conversationMode'
import { DEVICE_AUTH_TOKEN_STORAGE_KEY, DEVICE_ID_STORAGE_KEY } from '../constants/nursePatientConfig'

// Client-side credential provider for the local modes (v2 http / v3 ws).
// SECURE DEFAULT: fetch a short-lived token from /api/qwen-token (the long-term key stays
// server-side). Only if the endpoint is unreachable AND EXPO_PUBLIC_ALLOW_INSECURE_CLIENT_KEY
// is 'true' do we fall back to the inlined client key (kiosk/dev — extractable from the bundle).

let cached: { token: string; expiresAtMs: number } | null = null

export function clearTokenCache(): void {
  cached = null
}

async function fetchServerToken(): Promise<string | null> {
  const now = Date.now()
  if (cached && cached.expiresAtMs - 60_000 > now) return cached.token
  try {
    const [deviceId, authToken] = await Promise.all([
      AsyncStorage.getItem(DEVICE_ID_STORAGE_KEY),
      AsyncStorage.getItem(DEVICE_AUTH_TOKEN_STORAGE_KEY),
    ])
    const res = await fetch(getApiUrl('/api/qwen-token'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ deviceId, authToken }),
    })
    const body = (await res.json()) as { success?: boolean; token?: string; expiresAt?: string }
    if (!body?.success || !body.token) return null
    const parsed = body.expiresAt ? Date.parse(body.expiresAt) : NaN
    cached = { token: body.token, expiresAtMs: Number.isFinite(parsed) ? parsed : now + 1_800_000 }
    return body.token
  } catch {
    return null
  }
}

/** Bearer credential for a Qwen HTTP/WS call. Prefers the server-minted short-lived token. */
export async function getBearer(): Promise<string> {
  const token = await fetchServerToken()
  if (token) return token
  if (process.env.EXPO_PUBLIC_ALLOW_INSECURE_CLIENT_KEY === 'true' && QWEN.apiKey) {
    return QWEN.apiKey
  }
  throw new Error(
    'No Qwen credential: /api/qwen-token unreachable and EXPO_PUBLIC_ALLOW_INSECURE_CLIENT_KEY is not enabled.',
  )
}
