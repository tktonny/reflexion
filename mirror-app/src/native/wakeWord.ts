// Open-source on-device wake-word engine (openWakeWord pipeline) via onnxruntime-react-native.
//
// Pipeline (exact openWakeWord specs): 16 kHz 16-bit PCM -> melspectrogram.onnx ([1,samples] ->
// (frames,32), scaled x/10+2) -> embedding_model.onnx ([1,76,32,1] windows, stride 8 -> 96-dim) ->
// wakeword.onnx ([1,16,96] -> score); fire when score > THRESHOLD on TRIGGER_HITS consecutive frames.
//
// Models are BUNDLED in the APK (assets/wakeword/*.onnx, ~3.7 MB total) and loaded via expo-asset,
// so the wake word works out of the box with no device-side setup. If onnxruntime's native module
// isn't linked in the running build, create() returns null and callers fall back to tap-to-start.

import { Asset } from 'expo-asset'
import { Platform } from 'react-native'

// Static import is safe for bundling (the JS package is installed); the NATIVE calls throw if the
// module isn't linked into this build, which we catch in createWakeWordEngine().
import { InferenceSession, Tensor } from 'onnxruntime-react-native'

const MEL_BINS = 32
const EMB_WINDOW = 76 // mel frames per embedding window
const EMB_STRIDE = 8 // mel-frame step between successive embeddings
const WW_WINDOW = 16 // embeddings per wakeword prediction
const CHUNK = 1280 // samples per melspec call (80 ms @ 16 kHz)
const THRESHOLD = 0.5 // openWakeWord default
const TRIGGER_HITS = 3 // frames above threshold before firing (debounce)
const MEL_SCALE = (v: number) => v / 10 + 2

// Bundled ONNX assets (metro.config.js registers .onnx as an asset).
const MODEL_MODULES = {
  mel: require('../../assets/wakeword/melspectrogram.onnx'),
  emb: require('../../assets/wakeword/embedding_model.onnx'),
  ww: require('../../assets/wakeword/wakeword.onnx'),
}

export function isWakeWordRuntimeAvailable(): boolean {
  return Platform.OS !== 'web' && !!InferenceSession
}

/** Resolve a bundled ONNX asset to a local filesystem path for InferenceSession.create. */
async function assetPath(moduleId: number): Promise<string> {
  const asset = Asset.fromModule(moduleId)
  if (!asset.localUri) await asset.downloadAsync()
  return (asset.localUri ?? asset.uri).replace('file://', '')
}

export type WakeWordEngine = { feed: (pcm16: Int16Array) => Promise<void>; reset: () => void }

/** Build the engine, or null if the runtime/models are unavailable (caller then uses tap-to-start). */
export async function createWakeWordEngine(onDetected: () => void): Promise<WakeWordEngine | null> {
  if (!isWakeWordRuntimeAvailable()) return null
  let mel: InferenceSession, emb: InferenceSession, ww: InferenceSession
  try {
    mel = await InferenceSession.create(await assetPath(MODEL_MODULES.mel))
    emb = await InferenceSession.create(await assetPath(MODEL_MODULES.emb))
    ww = await InferenceSession.create(await assetPath(MODEL_MODULES.ww))
  } catch {
    return null // native onnxruntime not linked, or model load failed
  }

  const melIn = mel.inputNames[0]
  const melOut = mel.outputNames[0]
  const embIn = emb.inputNames[0]
  const embOut = emb.outputNames[0]
  const wwIn = ww.inputNames[0]
  const wwOut = ww.outputNames[0]

  let audio: number[] = [] // pending float32 samples (raw int16 values, NOT normalized)
  let melFrames: number[][] = [] // rows of 32 mel bins
  let nextEmbStart = 0 // index in melFrames for the next embedding window
  let embeds: number[][] = [] // rows of 96
  let hits = 0
  let running = false

  const reset = () => { audio = []; melFrames = []; nextEmbStart = 0; embeds = []; hits = 0 }

  async function step() {
    // 1) audio -> mel frames, in 1280-sample chunks
    while (audio.length >= CHUNK) {
      const chunk = audio.slice(0, CHUNK)
      audio = audio.slice(CHUNK)
      const out = await mel.run({ [melIn]: new Tensor('float32', Float32Array.from(chunk), [1, chunk.length]) })
      const data = out[melOut].data as Float32Array
      const frames = Math.floor(data.length / MEL_BINS)
      for (let f = 0; f < frames; f += 1) {
        const row = new Array<number>(MEL_BINS)
        for (let b = 0; b < MEL_BINS; b += 1) row[b] = MEL_SCALE(data[f * MEL_BINS + b])
        melFrames.push(row)
      }
    }
    // 2) mel windows [i:i+76] stride 8 -> 96-dim embeddings
    while (melFrames.length >= nextEmbStart + EMB_WINDOW) {
      const flat = new Float32Array(EMB_WINDOW * MEL_BINS)
      for (let i = 0; i < EMB_WINDOW; i += 1) {
        const row = melFrames[nextEmbStart + i]
        for (let b = 0; b < MEL_BINS; b += 1) flat[i * MEL_BINS + b] = row[b]
      }
      const out = await emb.run({ [embIn]: new Tensor('float32', flat, [1, EMB_WINDOW, MEL_BINS, 1]) })
      const e = out[embOut].data as Float32Array
      embeds.push(Array.from(e)) // 96-dim (drop leading singleton dims)
      nextEmbStart += EMB_STRIDE
      // 3) once we have 16 embeddings, score the latest window each time a new one arrives
      if (embeds.length >= WW_WINDOW) {
        const win = embeds.slice(embeds.length - WW_WINDOW)
        const wflat = new Float32Array(WW_WINDOW * win[0].length)
        for (let i = 0; i < WW_WINDOW; i += 1) wflat.set(win[i], i * win[0].length)
        const s = await ww.run({ [wwIn]: new Tensor('float32', wflat, [1, WW_WINDOW, win[0].length]) })
        const score = (s[wwOut].data as Float32Array)[0]
        if (score > THRESHOLD) { hits += 1; if (hits >= TRIGGER_HITS) { hits = 0; reset(); onDetected() } }
        else hits = 0
      }
    }
    // bound memory: drop consumed mel frames + cap the embedding buffer
    if (nextEmbStart > 0) { melFrames.splice(0, nextEmbStart); nextEmbStart = 0 }
    if (embeds.length > 120) embeds = embeds.slice(embeds.length - 120)
  }

  return {
    reset,
    feed: async (pcm16: Int16Array) => {
      for (let i = 0; i < pcm16.length; i += 1) audio.push(pcm16[i]) // raw int16 as float32
      if (running) return
      running = true
      try { await step() } finally { running = false }
    },
  }
}

/** Decode base64 PCM16 (little-endian) from expo-pcm-audio into an Int16Array. */
export function base64Pcm16ToInt16(b64: string): Int16Array {
  // RN (Hermes) provides a global atob; cast since it isn't in the RN TS lib types.
  const bin: string = (globalThis as unknown as { atob: (s: string) => string }).atob(b64)
  const n = bin.length >> 1
  const out = new Int16Array(n)
  for (let i = 0; i < n; i += 1) {
    const lo = bin.charCodeAt(i * 2) & 0xff
    const hi = bin.charCodeAt(i * 2 + 1) & 0xff
    out[i] = ((hi << 8) | lo) << 16 >> 16 // sign-extend
  }
  return out
}
