import Constants from 'expo-constants'
import { AppState, type AppStateStatus } from 'react-native'

import type { HardwareReport } from '../lib/hardwareCheck'
import { deviceFetch, getDeviceCredential, randomIdempotencyKey } from '../storage/deviceCredentials'
import { dataOrThrow } from './devicePairing'

export type DeviceHeartbeatState = 'idle' | 'online' | 'offline'
type Listener = (state: DeviceHeartbeatState) => void

const HEARTBEAT_INTERVAL_MS = 60_000
let currentState: DeviceHeartbeatState = 'idle'
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
        backendReachable: online,
        diagnostics: report ? {
          platform: report.platform,
          configuredMode: report.configuredMode,
          recommendedMode: report.recommendedMode,
          checks: Object.fromEntries(report.checks.map((check) => [check.key, check.status])),
        } : {},
      }),
    })
    const result = await dataOrThrow<{ operationId: string; state: 'accepted' }>(response)
    publish('online')
    return result
  } catch (error) {
    publish('offline')
    throw error
  }
}

export function startDeviceHeartbeat(report?: HardwareReport) {
  let appState: AppStateStatus = AppState.currentState
  let timer: ReturnType<typeof setInterval> | null = null
  const tick = () => {
    if (appState === 'active') void sendDeviceHeartbeat(report).catch(() => undefined)
  }
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
