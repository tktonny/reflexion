import { Platform } from 'react-native'

import { CONVERSATION_MODE, type ConversationMode } from '../config/conversationMode'
import { DEFAULT_RELAY_PORT } from '../constants/realtime'
import { isNativePcmAvailable } from '../native/pcmAudio'
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
    return req?.granted
      ? { key: 'mic', label, status: 'ok', detail: '已授权' }
      : { key: 'mic', label, status: 'warn', detail: '未授权(需在系统设置允许)' }
  } catch {
    return { key: 'mic', label, status: 'unknown', detail: '无法探测(需真机)' }
  }
}

async function checkCamera(): Promise<HardwareCheck> {
  const label = '摄像头'
  try {
    const cam: any = await import('expo-camera')
    const get = cam?.Camera?.getCameraPermissionsAsync ?? cam?.getCameraPermissionsAsync
    const perm = get ? await get() : null
    if (perm?.granted) return { key: 'camera', label, status: 'ok', detail: '已授权' }
    return { key: 'camera', label, status: 'warn', detail: '未授权 / 未探测' }
  } catch {
    return { key: 'camera', label, status: 'unknown', detail: '无法探测' }
  }
}

function checkSpeaker(): HardwareCheck {
  const label = '扬声器 / 音频输出'
  if (Platform.OS === 'web') {
    const has = typeof window !== 'undefined' && Boolean((window as any).AudioContext || (window as any).webkitAudioContext)
    return { key: 'speaker', label, status: has ? 'ok' : 'warn', detail: has ? 'Web Audio 可用' : '无 AudioContext' }
  }
  return { key: 'speaker', label, status: 'ok', detail: '假定可用(设备扬声器)' }
}

function checkNetwork(): HardwareCheck {
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
  const [mic, camera, backend, turn] = await Promise.all([
    checkMicrophone(),
    checkCamera(),
    checkBackend(),
    checkTurnAudio(),
  ])
  const checks: HardwareCheck[] = [
    checkNetwork(),
    backend,
    mic,
    checkSpeaker(),
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
