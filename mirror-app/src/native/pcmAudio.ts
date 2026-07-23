// Native PCM streaming audio bridge for v3 (direct realtime WS).
//
// The realtime path needs continuous mic PCM16 @16kHz -> base64 frames sent to Qwen, and
// gapless PCM16 @24kHz playback of the audio deltas. React Native core has no raw-PCM audio
// API, so this is backed by the local Expo native module `modules/expo-pcm-audio` (Android:
// AudioRecord/AudioTrack; iOS: AVAudioEngine). It requires a custom dev build — Expo Go and web
// cannot load it, so requireOptionalNativeModule returns null there and we fall back to a stub
// that throws a clear message. The WS/orchestration path is verified independently by
// server/smoke-direct-ws.mjs (direct-WS + ephemeral-token + orchestration, minus device audio).

import ExpoPcmAudio from '../../modules/expo-pcm-audio'

export type PcmAudioBridge = {
  /** Begin mic capture; onChunk receives base64 PCM16 mono @16kHz frames. */
  start: (onChunk: (base64Pcm16: string) => void) => Promise<void>
  /** Enqueue base64 PCM16 mono @24kHz for gapless playback. */
  play: (base64Pcm16: string) => void
  /** Immediately discard queued speaker audio when the user interrupts Aria. */
  clearPlayback: () => void
  /** Suppress mic capture during assistant playback (half-duplex). */
  setCaptureMuted: (muted: boolean) => void
  /** Unplayed playback backlog in ms; used to un-mute the mic only after playback drains. */
  getPlaybackBacklogMs: () => number
  /**
   * True only when the running build can actually measure the playback backlog. When false, a
   * getPlaybackBacklogMs() of 0 means "unknown", NOT "drained" — callers must fall back to a floor
   * wait derived from the enqueued audio duration so the full-playback guarantee is never defeated
   * by a stub returning 0 (implementation baseline §6).
   */
  isBacklogMeasurable: () => boolean
  stop: () => Promise<void>
}

/** True when the native PCM module is present in the running build (dev build on a device). */
export function isNativePcmAvailable(): boolean {
  return ExpoPcmAudio != null
}

const CAPTURE_SAMPLE_RATE = 16000

export function createPcmAudioBridge(): PcmAudioBridge {
  const native = ExpoPcmAudio
  if (!native) {
    const notWired = () => {
      throw new Error(
        'Native PCM audio module not in this build. v3 (direct realtime WS) needs the ' +
          'modules/expo-pcm-audio native module + a custom dev build (not Expo Go / web). ' +
          'Run `npx expo run:android` (or an EAS dev build), or use v1 (relay) / v2 (http).',
      )
    }
    return {
      start: async () => notWired(),
      play: () => notWired(),
      clearPlayback: () => {},
      setCaptureMuted: () => {},
      getPlaybackBacklogMs: () => 0,
      isBacklogMeasurable: () => false,
      stop: async () => {},
    }
  }

  let subscription: { remove: () => void } | null = null

  return {
    start: async (onChunk) => {
      subscription?.remove()
      subscription = native.addListener('onAudioChunk', (event: { data: string }) => {
        if (event?.data) onChunk(event.data)
      })
      await native.start(CAPTURE_SAMPLE_RATE)
    },
    play: (base64Pcm16) => native.play(base64Pcm16),
    clearPlayback: () => native.clearPlayback(),
    setCaptureMuted: (muted) => native.setCaptureMuted(muted),
    getPlaybackBacklogMs: () => native.getPlaybackBacklogMs?.() ?? 0,
    isBacklogMeasurable: () => typeof native.getPlaybackBacklogMs === 'function',
    stop: async () => {
      subscription?.remove()
      subscription = null
      await native.stop()
    },
  }
}
