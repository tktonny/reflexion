import Constants from 'expo-constants'
import { AppState, type AppStateStatus } from 'react-native'

import type { HardwareReport } from '../lib/hardwareCheck'
import { deviceFetch, getDeviceCredential, randomIdempotencyKey } from '../storage/deviceCredentials'
import { dataOrThrow } from './devicePairing'

export type DeviceHeartbeatState = 'idle' | 'online' | 'offline'
type Listener = (state: DeviceHeartbeatState) => void

// 60s cadence is comfortably under the backend's 15-minute unreachable threshold. An ambient mirror
// is normally foregrounded (kiosk), so we intentionally do NOT gate ticks on AppState 'active' — a
// momentarily inactive but powered-on device must still be seen as reachable (baseline §7 Phase 7).
const HEARTBEAT_INTERVAL_MS = 60_000
let currentState: DeviceHeartbeatState = 'idle'
// Real reachability signal: whether the PREVIOUS heartbeat POST reached the backend, rather than
// navigator.onLine (which is web-only and defaults truthy on React Native, masking real outages).
let lastBackendReachable = true
const listeners = new Set<Listener>()

function publish(next: DeviceHeartbeatState) {
  currentState = next
  for (const listener of listeners) listener(next)
}

export function subscribeDeviceHeartbeat(listener: Listener) {
  listeners.add(listener)
  listener(currentState)
  return () => { listeners.delete(listener) }
}

export async function sendDeviceHeartbeat(report?: HardwareReport) {
  const credential = await getDeviceCredential()
  if (!credential) return null
  const online = typeof navigator === 'undefined' || navigator.onLine !== false
  const mic = report?.checks.find((check) => check.key === 'mic')
  const speaker = report?.checks.find((check) => check.key === 'speaker')
  const heartbeatId = `hb_${randomIdempotencyKey()}`
  try {
    const response = await deviceFetch(`/api/v1/devices/${encodeURIComponent(credential.deviceId)}/heartbeats`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Idempotency-Key': heartbeatId },
      body: JSON.stringify({
        heartbeatId,
        recordedAt: new Date().toISOString(),
        appVersion: Constants.expoConfig?.version || 'unknown',
        networkStatus: online ? 'online' : 'offline',
        micStatus: mic?.status === 'ok' ? 'ok' : mic?.status === 'fail' ? 'permission_denied' : 'unavailable',
        speakerStatus: speaker?.status === 'fail' ? 'error' : 'ok',
        backendReachable: lastBackendReachable,
        diagnostics: report ? {
          platform: report.platform,
          configuredMode: report.configuredMode,
          recommendedMode: report.recommendedMode,
          checks: Object.fromEntries(report.checks.map((check) => [check.key, check.status])),
        } : {},
      }),
    })
    const result = await dataOrThrow<{ operationId: string; state: 'accepted' }>(response)
    lastBackendReachable = true
    publish('online')
    return result
  } catch (error) {
    lastBackendReachable = false
    publish('offline')
    throw error
  }
}

export function startDeviceHeartbeat(report?: HardwareReport) {
  let appState: AppStateStatus = AppState.currentState
  let timer: ReturnType<typeof setInterval> | null = null
  // Beat regardless of AppState so an idle-but-powered mirror is not misread as offline; resuming to
  // active triggers an extra immediate beat for faster recovery after a network blip.
  const tick = () => { void sendDeviceHeartbeat(report).catch(() => undefined) }
  tick()
  timer = setInterval(tick, HEARTBEAT_INTERVAL_MS)
  const subscription = AppState.addEventListener('change', (nextState) => {
    const becameActive = appState !== 'active' && nextState === 'active'
    appState = nextState
    if (becameActive) tick()
  })
  return () => {
    if (timer) clearInterval(timer)
    subscription.remove()
  }
}
