import { NativeModule, requireOptionalNativeModule } from 'expo'

// Events emitted from native. `onAudioChunk.data` is base64-encoded PCM16 mono at the capture
// sample rate (16 kHz), ready to be forwarded verbatim as Qwen realtime input_audio_buffer.append.
export type PcmAudioModuleEvents = {
  onAudioChunk: (event: { data: string }) => void
}

export declare class ExpoPcmAudioModule extends NativeModule<PcmAudioModuleEvents> {
  /** Start mic capture (PCM16 mono @ sampleRate) + open the streaming playback track. */
  start(sampleRate: number, useCommunicationMode: boolean): Promise<void>
  /** Stop capture + playback and release all native resources. */
  stop(): Promise<void>
  /** Enqueue base64 PCM16 mono @ 24 kHz for gapless playback (non-blocking; ordered). */
  play(base64Pcm16: string): void
  /** Drop any queued playback audio (barge-in / interrupt). */
  clearPlayback(): void
  /** Half-duplex: while true, captured mic frames are dropped instead of emitted. */
  setCaptureMuted(muted: boolean): void
  /** Unplayed playback backlog in ms (queued + buffered but not yet rendered). */
  getPlaybackBacklogMs(): number
}

// requireOptionalNativeModule returns null when the native side isn't in the build (Expo Go, web,
// or before a custom dev build). Callers (src/native/pcmAudio.ts) fall back gracefully to a stub.
export default requireOptionalNativeModule<ExpoPcmAudioModule>('ExpoPcmAudio')
