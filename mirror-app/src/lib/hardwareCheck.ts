import { Platform } from 'react-native'
import AsyncStorage from '@react-native-async-storage/async-storage'

import { CONVERSATION_MODE, type ConversationMode } from '../config/conversationMode'
import { getApiUrl } from '../config/apiUrl'
import { DEFAULT_RELAY_PORT } from '../constants/realtime'
import { isNativePcmAvailable } from '../native/pcmAudio'
import { isWakeWordRuntimeAvailable, resolveWakeWordAssets } from '../native/wakeWord'
import { validateBootstrapCredential } from '../orchestration/deviceBootstrap'
import { getBootstrapCredential, getDeviceCredential } from '../storage/deviceCredentials'
import { probeSpeakerLoopback } from './speakerCheck'
import { recommendMode } from './recommendMode'

// Past this, the wake-bounded day boundary (and therefore "is this the first conversation today?") can
// land on the wrong day. Generous enough that ordinary NTP jitter never trips it.
const MAX_CLOCK_SKEW_MS = 5 * 60 * 1000

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
    const request = cam?.requestCameraPermissionsAsync ?? cam?.Camera?.requestCameraPermissionsAsync
    const perm = get ? await get() : null
    if (perm?.granted) return { key: 'camera', label, status: 'ok', detail: '已授权' }
    if (perm && perm.canAskAgain === false) return { key: 'camera', label, status: 'fail', detail: '已拒绝(系统设置里开启)' }
    // Actively request once at setup so the installer/caregiver grants it while they are standing there,
    // instead of ambushing the elder later when the face pre-check ships. A refusal is NOT a failure of
    // the mirror: the voice check-in needs no camera, so we degrade to audio-only (visionAvailable=false)
    // and keep going — nothing about the daily conversation depends on this.
    const requested = request ? await request() : null
    if (requested?.granted) return { key: 'camera', label, status: 'ok', detail: '已授权' }
    return { key: 'camera', label, status: 'warn', detail: '未授权 — 降级为纯音频(语音 check-in 不需要摄像头)' }
  } catch {
    return { key: 'camera', label, status: 'unknown', detail: '无法探测(需真机)' }
  }
}

/**
 * Device identity: is this mirror provisioned with a credential that still works, and is that credential
 * actually THIS device's? Two failure modes that were previously invisible:
 *
 *  1. **Expiry.** The bootstrap token is a short-lived JWT (30-day TTL). Nothing warned about it, so a
 *     test fleet would simply stop being able to pair one day with no explanation.
 *  2. **Identity collision.** A bootstrap token is bound to ONE device (`did` in its payload). Building
 *     one APK with an embedded token and installing it on several devices makes them all claim the same
 *     device record — the later one silently takes over the earlier one's identity, and both devices'
 *     conversations land on the same patient. This is why the token must be per device; provision each
 *     unit through the enrollment screen instead of baking one token into a shared build.
 *
 * Credential *liveness* is deliberately not probed here: the boot flow already calls the authenticated
 * device-configuration endpoint and falls through to pairing when the credential has been revoked
 * (app/index.tsx), which is the correct recovery. This check only reports what it can see locally.
 */
async function checkDeviceIdentity(): Promise<HardwareCheck> {
  const label = '设备身份 / 凭证'
  try {
    const [bootstrap, credential] = await Promise.all([getBootstrapCredential(), getDeviceCredential()])
    if (credential) {
      // Paired. Flag a mismatch against the embedded token, if any — that is the collision above.
      const embedded = process.env.EXPO_PUBLIC_DEVICE_BOOTSTRAP_TOKEN
      if (embedded) {
        try {
          const claims = validateBootstrapCredential(embedded)
          if (claims.deviceId && claims.deviceId !== credential.deviceId) {
            return {
              key: 'identity',
              label,
              status: 'fail',
              detail: `构建内嵌的 token 属于另一台设备(…${claims.deviceId.slice(-6)}),与本机凭证(…${credential.deviceId.slice(-6)})不一致 — 多台设备共用同一 token 会互相顶号`,
            }
          }
        } catch { /* an expired/invalid embedded token cannot cause a collision; ignore it here */ }
      }
      const refreshExpiry = Date.parse(credential.refreshCredentialExpiresAt || '')
      if (!Number.isNaN(refreshExpiry)) {
        const days = Math.floor((refreshExpiry - Date.now()) / 86_400_000)
        if (days < 0) return { key: 'identity', label, status: 'fail', detail: '凭证已过期,需要重新配对' }
        if (days <= 7) return { key: 'identity', label, status: 'warn', detail: `已配对,但凭证 ${days} 天后过期` }
      }
      return { key: 'identity', label, status: 'ok', detail: `已配对(设备 …${credential.deviceId.slice(-6)})` }
    }
    if (!bootstrap) {
      return { key: 'identity', label, status: 'fail', detail: '未预配也未配对 — 需在设备录入 bootstrap 凭证' }
    }
    // Provisioned but not yet paired: surface how long the window still is.
    try {
      const claims = validateBootstrapCredential(bootstrap.token)
      const days = Math.floor((claims.expiresAt * 1000 - Date.now()) / 86_400_000)
      return days <= 7
        ? { key: 'identity', label, status: 'warn', detail: `待配对,bootstrap 凭证 ${days} 天后过期` }
        : { key: 'identity', label, status: 'ok', detail: `待配对(凭证还有 ${days} 天)` }
    } catch {
      return { key: 'identity', label, status: 'fail', detail: 'bootstrap 凭证已过期或无效 — 需重新录入' }
    }
  } catch {
    return { key: 'identity', label, status: 'unknown', detail: '无法读取本机凭证' }
  }
}

/** Wake word: onnxruntime bridge + the three bundled ONNX assets must both be present. A silent failure
 *  here degrades the mirror to tap-to-start, so an elder who relies on "Hello Aria" thinks it is dead. */
async function checkWakeWord(): Promise<HardwareCheck> {
  const label = '唤醒词(Hello Aria)'
  if (Platform.OS === 'web') return { key: 'wakeword', label, status: 'warn', detail: 'web 不支持,改用点击开始' }
  if (!isWakeWordRuntimeAvailable()) {
    return { key: 'wakeword', label, status: 'fail', detail: 'onnxruntime 未链接到此构建 — 只能点击开始' }
  }
  try {
    await resolveWakeWordAssets()
    return { key: 'wakeword', label, status: 'ok', detail: 'onnxruntime + 模型资源就绪' }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return { key: 'wakeword', label, status: 'fail', detail: `模型资源不可用:${message.slice(0, 60)}` }
  }
}

/** The REAL backend, unlike the `relay` probe which only ever hit the dev relay on localhost:8787.
 *  Also measures clock skew from the server's own timestamp, and reports server-side readiness so the
 *  mirror learns about a missing Qwen key / object store it cannot possibly check for itself. */
async function checkApiBackend(): Promise<HardwareCheck[]> {
  const label = '后端 API'
  const clockLabel = '设备时钟'
  try {
    const started = Date.now()
    const res = await fetch(getApiUrl('/health'), { method: 'GET' })
    if (!res.ok) {
      return [
        { key: 'api', label, status: 'fail', detail: `HTTP ${res.status}` },
        { key: 'clock', label: clockLabel, status: 'unknown', detail: '后端不可达,无法校时' },
      ]
    }
    const body = await res.json().catch(() => null) as
      | { serverTime?: string; readiness?: { objectStore?: boolean; database?: boolean; qwen?: Record<string, boolean | string> } }
      | null

    // Server-side readiness the device cannot see. Reported as a warn, never a block: the backend may be
    // fine for conversations while a non-critical piece is unset, and only an operator can fix these.
    const missing: string[] = []
    if (body?.readiness) {
      if (body.readiness.objectStore === false) missing.push('对象存储未配置(录音上传会失败)')
      if (body.readiness.database === false) missing.push('数据库未配置')
      const qwen = body.readiness.qwen
      const region = typeof qwen?.defaultRegion === 'string' ? qwen.defaultRegion : 'cn'
      if (qwen && qwen[region] === false) missing.push(`Qwen ${region} 区密钥缺失`)
    }
    const api: HardwareCheck = missing.length
      ? { key: 'api', label, status: 'warn', detail: `可达,但${missing.join(';')}` }
      : { key: 'api', label, status: 'ok', detail: `可达(${Date.now() - started}ms)` }

    // Clock skew: the wake-bounded "first conversation of the day" rule and the check-in schedule both
    // read the device clock, so drift silently breaks them. Compare against the server's timestamp.
    let clock: HardwareCheck = { key: 'clock', label: clockLabel, status: 'unknown', detail: '后端未返回时间' }
    const serverTime = body?.serverTime ? Date.parse(body.serverTime) : NaN
    if (!Number.isNaN(serverTime)) {
      const skewMs = Math.abs(Date.now() - serverTime)
      const skewLabel = skewMs < 1000 ? `${skewMs}ms` : `${Math.round(skewMs / 1000)}s`
      clock = skewMs > MAX_CLOCK_SKEW_MS
        ? { key: 'clock', label: clockLabel, status: 'fail', detail: `与服务器相差 ${skewLabel} — 每日判定会出错` }
        : { key: 'clock', label: clockLabel, status: 'ok', detail: `与服务器相差 ${skewLabel}` }
    }
    return [api, clock]
  } catch {
    return [
      { key: 'api', label, status: 'fail', detail: '不可达(检查网络)' },
      { key: 'clock', label: clockLabel, status: 'unknown', detail: '后端不可达,无法校时' },
    ]
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

/**
 * Dev-relay probe (localhost:8787). This is NOT the backend — a production Android mirror talks straight
 * to Qwen and never uses the relay, so this check was permanently `warn` while occupying the slot a real
 * backend check should have had (that is now `api`, see checkApiBackend). Kept for web/Electron, where
 * the relay genuinely carries the conversation; skipped entirely on native so it stops adding noise.
 */
async function checkRelay(): Promise<HardwareCheck | null> {
  if (Platform.OS !== 'web' && CONVERSATION_MODE !== 'relay') return null
  const label = '中继 relay(web/Electron 用)'
  const base = relayHttpBase()
  try {
    const res = await fetch(`${base}/health`, { method: 'GET' })
    if (res.ok) return { key: 'relay', label, status: 'ok', detail: base }
    return { key: 'relay', label, status: 'warn', detail: `HTTP ${res.status} @ ${base}` }
  } catch {
    return { key: 'relay', label, status: 'warn', detail: `不可达 @ ${base}` }
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
  const [mic, camera, apiAndClock, relay, turn, wakeword, identity, speakerProbe] = await Promise.all([
    checkMicrophone(),
    checkCamera(),
    checkApiBackend(),
    checkRelay(),
    checkTurnAudio(),
    checkWakeWord(),
    checkDeviceIdentity(),
    readSpeakerProbe(),
  ])
  const checks: HardwareCheck[] = [
    ...apiAndClock, // api + clock (the real backend, and clock skew measured against it)
    checkNetwork(),
    ...(relay ? [relay] : []),
    identity,
    mic,
    checkSpeaker(speakerProbe),
    wakeword,
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
