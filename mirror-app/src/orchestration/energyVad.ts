export type EnergyVadEvent = 'speech_started' | 'speech_stopped' | null

export type EnergyVadResult = {
  event: EnergyVadEvent
  rms: number
  threshold: number
}

export type EnergyVad = {
  feed: (pcm16: Int16Array) => EnergyVadResult
  reset: () => void
}

type Options = {
  sampleRate?: number
  speechStartRms?: number
  speechContinueRms?: number
  minSpeechMs?: number
  silenceMs?: number
  maxTurnMs?: number
}

/** Decode little-endian PCM16 emitted by expo-pcm-audio. */
export function decodeBase64Pcm16(base64: string): Int16Array {
  const decoder = (globalThis as unknown as { atob?: (value: string) => string }).atob
  if (!decoder) throw new Error('The JavaScript runtime does not provide base64 audio decoding.')
  const bytes = decoder(base64)
  const sampleCount = bytes.length >> 1
  const samples = new Int16Array(sampleCount)
  for (let index = 0; index < sampleCount; index += 1) {
    const lo = bytes.charCodeAt(index * 2) & 0xff
    const hi = bytes.charCodeAt(index * 2 + 1) & 0xff
    samples[index] = (((hi << 8) | lo) << 16) >> 16
  }
  return samples
}

export function pcm16Rms(samples: Int16Array): number {
  if (samples.length === 0) return 0
  let sumSquares = 0
  for (let index = 0; index < samples.length; index += 1) {
    const normalized = samples[index] / 32768
    sumSquares += normalized * normalized
  }
  return Math.sqrt(sumSquares / samples.length)
}

/**
 * Small deterministic VAD for manual Qwen turns. The native bridge delivers roughly 100 ms
 * frames, so two voiced frames start a turn and 1.2 s of low energy ends it. The native bridge
 * already enables NS/AGC; fixed thresholds avoid the failure mode where a long idle period slowly
 * raises an adaptive noise floor until ordinary speech is no longer detected.
 */
export function createEnergyVad(options: Options = {}): EnergyVad {
  const sampleRate = options.sampleRate ?? 16_000
  const speechStartRms = options.speechStartRms ?? 0.015
  const speechContinueRms = options.speechContinueRms ?? 0.008
  const minSpeechMs = options.minSpeechMs ?? 200
  const silenceMs = options.silenceMs ?? 1200
  const maxTurnMs = options.maxTurnMs ?? 30_000

  let speaking = false
  let candidateSpeechMs = 0
  let activeSpeechMs = 0
  let quietMs = 0

  const reset = () => {
    speaking = false
    candidateSpeechMs = 0
    activeSpeechMs = 0
    quietMs = 0
  }

  const feed = (samples: Int16Array): EnergyVadResult => {
    const rms = pcm16Rms(samples)
    const frameMs = samples.length > 0 ? (samples.length / sampleRate) * 1000 : 0
    const startThreshold = speechStartRms

    if (!speaking) {
      if (rms >= startThreshold) {
        candidateSpeechMs += frameMs
      } else {
        candidateSpeechMs = 0
      }
      if (candidateSpeechMs >= minSpeechMs) {
        speaking = true
        activeSpeechMs = candidateSpeechMs
        quietMs = 0
        return { event: 'speech_started', rms, threshold: startThreshold }
      }
      return { event: null, rms, threshold: startThreshold }
    }

    activeSpeechMs += frameMs
    const continueThreshold = speechContinueRms
    quietMs = rms < continueThreshold ? quietMs + frameMs : 0
    if (quietMs >= silenceMs || activeSpeechMs >= maxTurnMs) {
      const result = { event: 'speech_stopped' as const, rms, threshold: continueThreshold }
      reset()
      return result
    }
    return { event: null, rms, threshold: continueThreshold }
  }

  return { feed, reset }
}
