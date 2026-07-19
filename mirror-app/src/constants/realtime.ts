import { Platform } from 'react-native'

// Path served by the standalone Node relay (server/index.mjs).
export const REALTIME_WS_PATH = '/api/clinic/realtime/ws'
export const DEFAULT_RELAY_PORT = 8787

// Client-facing capture/playback audio contract (matches the relay session.update).
export const CAPTURE_SAMPLE_RATE = 16000
export const PLAYBACK_SAMPLE_RATE = 24000

/**
 * Resolve the relay WebSocket URL. On web the relay runs as a separate Node
 * process (default localhost:8787); override with EXPO_PUBLIC_RELAY_WS_URL.
 */
export function getRealtimeWsUrl(patientId: string, language: string): string {
  const base = resolveRelayBase()
  const query = `patient_id=${encodeURIComponent(patientId)}&language=${encodeURIComponent(language)}`
  return `${base}${REALTIME_WS_PATH}?${query}`
}

function resolveRelayBase(): string {
  const explicit = process.env.EXPO_PUBLIC_RELAY_WS_URL
  if (explicit) return explicit.replace(/\/$/, '')

  if (Platform.OS === 'web' && typeof window !== 'undefined') {
    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws'
    return `${scheme}://${window.location.hostname}:${DEFAULT_RELAY_PORT}`
  }
  // Native fallback (dev): expects EXPO_PUBLIC_RELAY_WS_URL to be set to the host LAN IP.
  return `ws://localhost:${DEFAULT_RELAY_PORT}`
}
