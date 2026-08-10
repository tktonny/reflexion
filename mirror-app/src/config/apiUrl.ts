import Constants from 'expo-constants'
import { Platform } from 'react-native'

export function hasConfiguredApiBase() {
  return Boolean(getConfiguredApiBase())
}

export function getConfiguredApiBase() {
  return (process.env.EXPO_PUBLIC_API_BASE || process.env.EXPO_PUBLIC_CAREGIVER_APP_BACKEND_URL)?.trim() || ''
}

/** Safe for display and diagnostics: hostname only, never a URL containing credentials or query data. */
export function getApiHostname() {
  const base = getConfiguredApiBase()
  if (!base) return 'unconfigured'
  try { return new URL(base).hostname || 'unconfigured' } catch { return 'invalid' }
}

/** The Linux Electron shell proxies backend calls from the same loopback origin. */
function usesElectronApiProxy() {
  if (Platform.OS !== 'web' || typeof window === 'undefined') return false
  return (window as unknown as { reflexionMirror?: { apiProxy?: boolean } }).reflexionMirror?.apiProxy === true
}

export function getApiUrl(path: string) {
  if (usesElectronApiProxy()) return path
  const configuredBase = getConfiguredApiBase()
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
