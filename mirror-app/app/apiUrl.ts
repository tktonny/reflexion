import Constants from 'expo-constants'
import { Platform } from 'react-native'

export function getApiUrl(path: string) {
  // Web: relative path hits the same origin (the Expo API routes).
  if (Platform.OS === 'web' && typeof window !== 'undefined') {
    return path
  }

  // Native production: an explicit backend base (the deployed Expo API host).
  const base = process.env.EXPO_PUBLIC_API_BASE
  if (base) return `${base.replace(/\/$/, '')}${path}`

  // Native dev: the Metro/dev-server host serves the API routes.
  const hostUri =
    Constants.expoConfig?.hostUri ||
    Constants.expoGoConfig?.debuggerHost ||
    Constants.manifest2?.extra?.expoGo?.debuggerHost
  if (hostUri) {
    const host = hostUri.split(':').slice(0, 2).join(':')
    return `http://${host}${path}`
  }

  // Standalone APK with no backend configured: return an ABSOLUTE but unreachable URL so fetch fails
  // cleanly (callers offline-queue / fall back to client-direct) instead of throwing on a relative
  // path. Port 9 (discard) is a VALID port that just refuses — avoids the "Invalid URL port 0" error.
  return `http://127.0.0.1:9${path}`
}
