import { Platform } from 'react-native'
import AsyncStorage from '@react-native-async-storage/async-storage'

import { CONVERSATION_MODE, type ConversationMode } from '../config/conversationMode'
import { DEFAULT_RELAY_PORT } from '../constants/realtime'
import { isNativePcmAvailable } from '../native/pcmAudio'
import { probeSpeakerLoopback } from './speakerCheck'
import { recommendMode } from './recommendMode'

// Startup hardware / capability self-check. Runs on every launch (see app/_layout + the
// /hardware-check screen). Web-verifiable now; native hardware results appear once built to a
// device. Also acts as the v3 readiness gate and the capability layer for adaptive transport.
//
// Whether the v3 native PCM streaming module is present in this build. Probed at runtime
// (modules/expo-pcm-audio via requireOptionalNativeModule) — true only in a custom dev build on a
// device, false on web / Expo Go. No manual flag to flip: the self-check reports reality.
export const HAS_NATIVE_PCM_STREAM = isNativePcmAvailable()

export type CheckStatus = 'ok' | 'warn' | 'fail' | 'unknown'
export type HardwareCheck = { key: string; label: string; status: CheckStatus; detail: string }
export type HardwareReport = {
  platform: string
  checks: HardwareCheck[]
  recommendedMode: ConversationMode | 'none'
  recommendedReason: string
  configuredMode: ConversationMode
}

// Cached acoustic-loopback result. The probe plays an audible tone, and runHardwareChecks() is called
// up to three times per launch (layout + settings + the report screen), so it is NEVER run implicitly —
// it is opt-in via runSpeakerProbe() and everything else reads this cache. A result older than the TTL
// degrades to `unknown` rather than being trusted forever.
export type SpeakerProbeRecord = { status: 'ok' | 'fail'; detail: string; at: number }
const SPEAKER_PROBE_STORAGE_KEY = 'reflexion:speakerProbe'
const SPEAKER_PROBE_TTL_MS = 24 * 60 * 60 * 1000

async function readSpeakerProbe(): Promise<SpeakerProbeRecord | null> {
  try {
    const raw = await AsyncStorage.getItem(SPEAKER_PROBE_STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as SpeakerProbeRecord
    return parsed && (parsed.status === 'ok' || parsed.status === 'fail') && typeof parsed.at === 'number' ? parsed : null
  } catch {
    return null
  }
}

/**
 * Run the real speaker loopback (audible tone) and cache the verdict. Call only when no conversation or
 * wake-word listener owns the audio device. `unknown` results are not cached — they mean "could not
 * test", so the next opportunity should try again rather than inherit a non-answer.
 */
export async function runSpeakerProbe(): Promise<HardwareCheck> {
  const result = await probeSpeakerLoopback()
  if (result.status !== 'unknown') {
    const record: SpeakerProbeRecord = { status: result.status, detail: result.detail, at: Date.now() }
    try { await AsyncStorage.setItem(SPEAKER_PROBE_STORAGE_KEY, JSON.stringify(record)) } catch { /* best effort */ }
  }
  return { key: 'speaker', label: '扬声器 / 音频输出', status: result.status, detail: result.detail }
}

/** True when there is no usable (fresh) loopback verdict, i.e. the mirror should probe when it can. */
export async function speakerProbeIsStale(): Promise<boolean> {
  if (Platform.OS === 'web' || !HAS_NATIVE_PCM_STREAM) return false
  const probe = await readSpeakerProbe()
  return !probe || Date.now() - probe.at > SPEAKER_PROBE_TTL_MS
}

function relayHttpBase(): string {
  const explicit = process.env.EXPO_PUBLIC_RELAY_WS_URL
  if (explicit) return explicit.replace(/^ws/, 'http').replace(/\/$/, '')
  if (Platform.OS === 'web' && typeof window !== 'undefined') {
    const scheme = window.location.protocol === 'https:' ? 'https' : 'http'
    return `${scheme}://${window.location.hostname}:${DEFAULT_RELAY_PORT}`
  }
  return `http://localhost:${DEFAULT_RELAY_PORT}`
}

async function checkMicrophone(): Promise<HardwareCheck> {
  const label = '麦克风'
  if (Platform.OS === 'web') {
    const md = (globalThis as any)?.navigator?.mediaDevices
    if (!md?.getUserMedia) return { key: 'mic', label, status: 'fail', detail: 'getUserMedia 不可用' }
    try {
      const stream = await md.getUserMedia({ audio: true })
      stream.getTracks().forEach((t: any) => t.stop())
      return { key: 'mic', label, status: 'ok', detail: '已授权' }
    } catch {
      return { key: 'mic', label, status: 'fail', detail: '未授权 / 不可用' }
    }
  }
  try {
    const audio: any = await import('expo-audio')
    const current = await audio.getRecordingPermissionsAsync?.()
    if (current?.granted) return { key: 'mic', label, status: 'ok', detail: '已授权' }
    const req = await audio.requestRecordingPermissionsAsync?.()
    // Denied is `fail`, not `warn`: without the mic Aria cannot hear anything, the check-in captures
    // silence, and the heartbeat only reports `permission_denied` for `fail` — as a warn it was invisible
    // to the caregiver AND to the readiness gate.
    return req?.granted
      ? { key: 'mic', label, status: 'ok', detail: '已授权' }
      : { key: 'mic', label, status: 'fail', detail: '未授权(需在系统设置允许)' }
  } catch {
    return { key: 'mic', label, status: 'unknown', detail: '无法探测(需真机)' }
  }
}

async function checkCamera(): Promise<HardwareCheck> {
  const label = '摄像头'
  // Passive probe only — we deliberately do NOT auto-prompt (the camera is for the future face
  // pre-check, not the voice check-in), so an "undetermined" state is expected, not an error.
  if (Platform.OS === 'web') {
    try {
      const md = (globalThis as any)?.navigator?.mediaDevices
      if (md?.enumerateDevices) {
        const devices = await md.enumerateDevices()
        if (!devices.some((d: any) => d.kind === 'videoinput')) {
          return { key: 'camera', label, status: 'warn', detail: '未检测到摄像头' }
        }
      }
      let state = 'prompt'
      try {
        const perm = await (navigator as any).permissions?.query?.({ name: 'camera' })
        state = perm?.state ?? 'prompt'
      } catch { /* Safari/Firefox may not support querying camera permission */ }
      if (state === 'granted') return { key: 'camera', label, status: 'ok', detail: '已授权' }
      if (state === 'denied') return { key: 'camera', label, status: 'fail', detail: '已拒绝(浏览器里允许)' }
      return { key: 'camera', label, status: 'warn', detail: '检测到摄像头,尚未授权(人脸预检时再申请)' }
    } catch {
      return { key: 'camera', label, status: 'unknown', detail: '无法探测' }
    }
  }
  try {
    const cam: any = await import('expo-camera')
    const get = cam?.getCameraPermissionsAsync ?? cam?.Camera?.getCameraPermissionsAsync
    const perm = get ? await get() : null
    if (perm?.granted) return { key: 'camera', label, status: 'ok', detail: '已授权' }
    if (perm && perm.canAskAgain === false) return { key: 'camera', label, status: 'fail', detail: '已拒绝(系统设置里开启)' }
    return { key: 'camera', label, status: 'warn', detail: '尚未授权(人脸预检时再申请)' }
  } catch {
    return { key: 'camera', label, status: 'unknown', detail: '无法探测(需真机)' }
  }
}

function checkSpeaker(probe?: SpeakerProbeRecord | null): HardwareCheck {
  const label = '扬声器 / 音频输出'
  if (Platform.OS === 'web') {
    const has = typeof window !== 'undefined' && Boolean((window as any).AudioContext || (window as any).webkitAudioContext)
    return { key: 'speaker', label, status: has ? 'ok' : 'warn', detail: has ? 'Web Audio 可用' : '无 AudioContext' }
  }
  // This used to return a hardcoded `ok, 假定可用` — a claim nothing had verified, which made the
  // caregiver-visible heartbeat report a healthy speaker on a mirror whose speaker was dead or muted.
  // Now the only `ok` comes from a real acoustic loopback (src/lib/speakerCheck.ts); with no recent
  // probe we say `unknown`, because "not tested" and "working" are different things.
  if (!probe) return { key: 'speaker', label, status: 'unknown', detail: '尚未做回环自检' }
  const age = Math.max(0, Date.now() - probe.at)
  if (age > SPEAKER_PROBE_TTL_MS) {
    return { key: 'speaker', label, status: 'unknown', detail: `上次回环自检已过期(${Math.round(age / 3_600_000)} 小时前)` }
  }
  return { key: 'speaker', label, status: probe.status, detail: probe.detail }
}

function checkNetwork(): HardwareCheck {
  // React Native has no navigator.onLine, so on the actual mirror this was structurally always `ok`
  // (`undefined !== false`). Report `unknown` there instead of inventing an online state — real
  // reachability is what the heartbeat measures (deviceHeartbeat publishes online/offline from a live
  // request), and the boot gate uses an actual /health ping.
  if (Platform.OS !== 'web') {
    return { key: 'network', label: '网络', status: 'unknown', detail: '由心跳/后端探测判定' }
  }
  const online = typeof navigator !== 'undefined' ? (navigator as any).onLine !== false : true
  return { key: 'network', label: '网络', status: online ? 'ok' : 'fail', detail: online ? '在线' : '离线' }
}

async function checkBackend(): Promise<HardwareCheck> {
  const label = '后端 / 中继(relay)'
  const base = relayHttpBase()
  try {
    const res = await fetch(`${base}/health`, { method: 'GET' })
    if (res.ok) return { key: 'relay', label, status: 'ok', detail: base }
    return { key: 'relay', label, status: 'warn', detail: `HTTP ${res.status} @ ${base}` }
  } catch {
    return { key: 'relay', label, status: 'warn', detail: `不可达 @ ${base}(v2/直连不需要)` }
  }
}

async function checkTurnAudio(): Promise<HardwareCheck> {
  const label = '回合制音频(v2)'
  if (Platform.OS === 'web') {
    const has = typeof window !== 'undefined' && Boolean((window as any).AudioContext || (window as any).webkitAudioContext)
    return { key: 'turnaudio', label, status: has ? 'ok' : 'warn', detail: 'Web Audio 采集/播放' }
  }
  try {
    await import('expo-audio')
    return { key: 'turnaudio', label, status: 'ok', detail: 'expo-audio 录/放' }
  } catch {
    return { key: 'turnaudio', label, status: 'fail', detail: 'expo-audio 缺失' }
  }
}

function checkRealtimeAudio(): HardwareCheck {
  const label = '实时音频(v3 原生流式)'
  if (Platform.OS === 'web') return { key: 'rtaudio', label, status: 'warn', detail: 'web 走中继,不适用' }
  return HAS_NATIVE_PCM_STREAM
    ? { key: 'rtaudio', label, status: 'ok', detail: '原生 PCM 流模块已加载' }
    : { key: 'rtaudio', label, status: 'fail', detail: '原生 PCM 模块不在此构建中(需 dev build:expo run:android)' }
}

function recommend(platform: string, byKey: Record<string, HardwareCheck>): { mode: ConversationMode | 'none'; reason: string } {
  return recommendMode(platform, {
    micOk: byKey.mic?.status === 'ok',
    relayOk: byKey.relay?.status === 'ok',
    turnOk: byKey.turnaudio?.status === 'ok',
    rtOk: byKey.rtaudio?.status === 'ok',
  })
}

export async function runHardwareChecks(): Promise<HardwareReport> {
  const platform = Platform.OS
  const [mic, camera, backend, turn, speakerProbe] = await Promise.all([
    checkMicrophone(),
    checkCamera(),
    checkBackend(),
    checkTurnAudio(),
    readSpeakerProbe(),
  ])
  const checks: HardwareCheck[] = [
    checkNetwork(),
    backend,
    mic,
    checkSpeaker(speakerProbe),
    camera,
    turn,
    checkRealtimeAudio(),
  ]
  const byKey = Object.fromEntries(checks.map((c) => [c.key, c])) as Record<string, HardwareCheck>
  const rec = recommend(platform, byKey)
  return {
    platform,
    checks,
    recommendedMode: rec.mode,
    recommendedReason: rec.reason,
    configuredMode: CONVERSATION_MODE,
  }
}
