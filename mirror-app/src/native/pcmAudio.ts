// Native PCM streaming audio bridge for v3 (direct realtime WS).
//
// The realtime path needs continuous mic PCM16 @16kHz -> base64 frames sent to Qwen, and
// gapless PCM16 @24kHz playback of the audio deltas. React Native core has no raw-PCM audio
// API, so this requires a native module + a custom dev build (Expo Go cannot do it).
//
// This file defines the interface the v3 hook depends on. The concrete implementation is a
// device-only task: install a PCM module (e.g. `npx expo install react-native-audio-api`,
// or @fugood/react-native-audio-pcm-stream for capture + a native AudioTrack player) and wire
// it here. Until then the factory throws a clear message so the WS/orchestration path can be
// built and verified independently (see server/smoke-direct-ws.mjs, which already proves the
// direct-WS + ephemeral-token + orchestration path works end-to-end minus device audio).

export type PcmAudioBridge = {
  /** Begin mic capture; onChunk receives base64 PCM16 mono @16kHz frames. */
  start: (onChunk: (base64Pcm16: string) => void) => Promise<void>
  /** Enqueue base64 PCM16 mono @24kHz for gapless playback. */
  play: (base64Pcm16: string) => void
  /** Suppress mic capture during assistant playback (half-duplex). */
  setCaptureMuted: (muted: boolean) => void
  stop: () => Promise<void>
}

export function createPcmAudioBridge(): PcmAudioBridge {
  const notWired = () => {
    throw new Error(
      'Native PCM audio not wired. v3 (direct realtime WS) needs a native PCM module + dev build. ' +
        'Install one (e.g. react-native-audio-api) and implement createPcmAudioBridge in src/native/pcmAudio.ts. ' +
        'Use v1 (relay) or v2 (http) meanwhile.',
    )
  }
  return {
    start: async () => notWired(),
    play: () => notWired(),
    setCaptureMuted: () => {},
    stop: async () => {},
  }
}
