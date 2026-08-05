import Constants from 'expo-constants'
import { Platform } from 'react-native'

/**
 * True when running inside the Linux Electron shell, which forwards /api and /health to the backend from
 * its own loopback origin (electron/main.js `proxyToBackend`).
 *
 * In that case the app MUST use relative, same-origin URLs even when EXPO_PUBLIC_API_BASE is baked in:
 * an absolute origin makes every call cross-origin from `http://127.0.0.1:8899`, which needs the backend's
 * CORS allowlist to name that origin. It did not, so Chromium blocked every request and the mirror
 * reported "unable to reach the Reflexion service" against a healthy API. Same-origin has no preflight.
 */
function usesElectronApiProxy() {
  if (Platform.OS !== 'web' || typeof window === 'undefined') return false
  return (window as unknown as { reflexionMirror?: { apiProxy?: boolean } }).reflexionMirror?.apiProxy === true
}

export function getApiUrl(path: string) {
  if (usesElectronApiProxy()) return path

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
