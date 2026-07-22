import Constants from 'expo-constants'
import { Platform } from 'react-native'

export function getApiUrl(path: string) {
  const configuredBase = process.env.EXPO_PUBLIC_API_BASE || process.env.EXPO_PUBLIC_CAREGIVER_APP_BACKEND_URL
  if (configuredBase) return `${configuredBase.replace(/\/$/, '')}${path}`

  if (Platform.OS === 'web' && typeof window !== 'undefined') return path

  const hostUri =
    Constants.expoConfig?.hostUri ||
    Constants.expoGoConfig?.debuggerHost ||
    Constants.manifest2?.extra?.expoGo?.debuggerHost
  if (hostUri) {
    const host = hostUri.split(':').slice(0, 2).join(':')
    return `http://${host}${path}`
  }

  // A release without a configured server fails closed. Port 9 is deliberately unreachable.
  return `http://127.0.0.1:9${path}`
}
