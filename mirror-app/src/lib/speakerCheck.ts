import { Platform } from 'react-native'

import { decodeBase64Pcm16, pcm16Rms } from '../orchestration/energyVad'
import { createPcmAudioBridge, isNativePcmAvailable } from '../native/pcmAudio'

// REAL speaker verification by acoustic loopback: play a short tone through the speaker and listen for
// it on the mic. Before this, checkSpeaker() simply returned `ok, 假定可用` on native — it tested nothing,
// so the caregiver-visible heartbeat reported a healthy speaker even on a mirror whose speaker was dead
// or muted. For an elder's mirror that is the worst possible blind spot: a dead speaker means Aria talks
// and nobody hears her, and the check-in fails silently.
//
// WHY THIS CAN WORK AT ALL (it looks like it shouldn't): the capture path always uses
// VOICE_COMMUNICATION + AcousticEchoCanceler, which exists precisely to erase our own speaker output.
// But the AEC only has a *reference* to cancel when playback also runs on the communication path
// (MODE_IN_COMMUNICATION). The native module exposes that as a per-bridge flag, and its own comment
// notes that with it off "the assistant hears herself". So the test runs a dedicated bridge with
// communicationMode:false — the one configuration in which the mic is *supposed* to hear the speaker.
//
// It is a JOINT test: a failure means "the speaker→mic path is broken", which could be either end.
// Callers must therefore only attribute it to the speaker when the mic itself has already passed.

export type SpeakerProbeResult = {
  status: 'ok' | 'fail' | 'unknown'
  detail: string
  /** Ambient level measured before the tone (0..1 RMS). */
  noiseFloor?: number
  /** Peak level measured while the tone played (0..1 RMS). */
  tonePeak?: number
}

const PLAYBACK_SAMPLE_RATE = 24_000 // must match the native AudioTrack rate
const TONE_HZ = 880 // A5 — comfortably inside phone-speaker and voice-mic response
const TONE_MS = 450
const AMBIENT_MS = 350
const SETTLE_MS = 250 // let playback actually reach the speaker before we judge
// The tone must clear BOTH a relative bar (vs this room's ambient) and an absolute floor, so a silent
// room can't pass on noise alone and a noisy room can't fail a working speaker.
const MIN_RATIO_OVER_AMBIENT = 2.5
const MIN_ABSOLUTE_RMS = 0.01

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/** Base64 PCM16 mono @24kHz sine with short fades, so the tone is a soft chime rather than a click. */
function buildTone(): string {
  const total = Math.floor((PLAYBACK_SAMPLE_RATE * TONE_MS) / 1000)
  const fade = Math.floor(total * 0.15)
  const bytes = new Uint8Array(total * 2)
  for (let i = 0; i < total; i += 1) {
    let gain = 0.35
    if (i < fade) gain *= i / fade
    else if (i > total - fade) gain *= (total - i) / fade
    const sample = Math.round(Math.sin((2 * Math.PI * TONE_HZ * i) / PLAYBACK_SAMPLE_RATE) * gain * 32767)
    bytes[i * 2] = sample & 0xff
    bytes[i * 2 + 1] = (sample >> 8) & 0xff
  }
  let binary = ''
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i])
  const encoder = (globalThis as unknown as { btoa?: (value: string) => string }).btoa
  if (!encoder) throw new Error('no base64 encoder')
  return encoder(binary)
}

/**
 * Play a tone and confirm the mic hears it. Never throws — an un-runnable probe reports `unknown`
 * rather than a false `ok`/`fail`. MUST NOT run while a conversation or the wake-word listener owns the
 * audio device (the native AudioRecord/AudioTrack are singletons); callers gate on that.
 */
export async function probeSpeakerLoopback(): Promise<SpeakerProbeResult> {
  if (Platform.OS === 'web') return { status: 'unknown', detail: 'web 不做回环自检' }
  if (!isNativePcmAvailable()) {
    return { status: 'unknown', detail: '原生音频模块不在此构建中,无法验证扬声器' }
  }

  // communicationMode:false ⇒ playback bypasses the AEC reference path, so the mic can hear the tone.
  const bridge = createPcmAudioBridge({ communicationMode: false })
  let peak = 0
  let listening = false

  try {
    await bridge.start((base64) => {
      if (!listening) return
      try {
        const level = pcm16Rms(decodeBase64Pcm16(base64))
        if (level > peak) peak = level
      } catch {
        // a single undecodable frame must not fail the probe
      }
    })

    // 1) ambient floor
    listening = true
    await sleep(AMBIENT_MS)
    const noiseFloor = peak

    // 2) tone
    peak = 0
    bridge.play(buildTone())
    await sleep(TONE_MS + SETTLE_MS)
    const tonePeak = peak
    listening = false

    const clearsAmbient = tonePeak >= noiseFloor * MIN_RATIO_OVER_AMBIENT
    const clearsFloor = tonePeak >= MIN_ABSOLUTE_RMS
    const round = (v: number) => v.toFixed(4)
    if (clearsAmbient && clearsFloor) {
      return { status: 'ok', detail: `已听到测试音(${round(tonePeak)} vs 环境 ${round(noiseFloor)})`, noiseFloor, tonePeak }
    }
    return {
      status: 'fail',
      detail: `未听到测试音(${round(tonePeak)} vs 环境 ${round(noiseFloor)})——扬声器可能损坏或音量为 0`,
      noiseFloor,
      tonePeak,
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return { status: 'unknown', detail: `无法验证扬声器:${message.slice(0, 80)}` }
  } finally {
    try { await bridge.stop() } catch { /* best-effort teardown */ }
  }
}
