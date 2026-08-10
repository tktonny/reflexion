import { useEffect, useRef } from 'react'

import { createPcmAudioBridge, isNativePcmAvailable, type PcmAudioBridge } from '../native/pcmAudio'
import {
  base64Pcm16ToInt16,
  createWakeWordEngine,
  isWakeWordRuntimeAvailable,
  type WakeWordEngine,
} from '../native/wakeWord'
import { createEnergyVad, decodeBase64Pcm16 } from '../orchestration/energyVad'

/**
 * Listen for the open-source wake word while `active`, firing onDetected once per detection.
 * Reuses the existing expo-pcm-audio capture (16 kHz PCM) to feed the openWakeWord ONNX engine.
 *
 * Falls back to a simple energy-based VAD when onnxruntime is unavailable — any sustained speech
 * above a threshold triggers the wake callback, so the mirror isn't completely tap-only.
 */
export function useWakeWord(active: boolean, onDetected: () => void): void {
  const onDetectedRef = useRef(onDetected)
  onDetectedRef.current = onDetected

  const bridgeRef = useRef<PcmAudioBridge | null>(null)
  const firedRef = useRef(false)

  useEffect(() => {
    if (!active) return
    const runtimeOk = isWakeWordRuntimeAvailable()
    const pcmOk = isNativePcmAvailable()
    console.log('[wakeword] active=true runtime=', runtimeOk, 'pcm=', pcmOk)
    if (!runtimeOk || !pcmOk) {
      // Fallback: energy-based VAD when ONNX is unavailable — any sustained speech triggers.
      if (!pcmOk) {
        console.log('[wakeword] disabled — PCM missing')
        return
      }
      console.log('[wakeword] ONNX unavailable, using energy-VAD fallback')
      let cancelled = false
      firedRef.current = false
      const FALLBACK_RMS = 0.04
      const FALLBACK_MS = 1000
      const vad = createEnergyVad({ speechStartRms: FALLBACK_RMS, minSpeechMs: FALLBACK_MS, silenceMs: 400 })
      const bridge = createPcmAudioBridge({ communicationMode: false })
      bridgeRef.current = bridge
      void bridge.start((b64) => {
        if (firedRef.current) return
        const pcm = decodeBase64Pcm16(b64)
        const vadResult = vad.feed(pcm)
        if (vadResult.event === 'speech_started') {
          console.log('[wakeword] VAD fallback detected speech, rms=', vadResult.rms.toFixed(4))
          firedRef.current = true
          onDetectedRef.current()
        }
      }).then(() => {
        if (!cancelled) console.log('[wakeword] VAD mic bridge started')
      }).catch((err: any) => {
        console.warn('[wakeword] VAD mic bridge start failed:', err?.message || err)
      })
      return () => {
        cancelled = true
        void bridge.stop().catch(() => {})
      }
    }
    let cancelled = false
    firedRef.current = false

    void (async () => {
      console.log('[wakeword] creating engine...')
      const engine: WakeWordEngine | null = await createWakeWordEngine(() => {
        console.log('[wakeword] DETECTED! firing callback')
        if (firedRef.current) return
        firedRef.current = true
        onDetectedRef.current()
      })
      if (cancelled || !engine) {
        console.log('[wakeword] engine creation failed — cancelled=', cancelled, 'engine=', !!engine)
        return
      }
      console.log('[wakeword] engine created, starting mic bridge...')
      const bridge = createPcmAudioBridge({ communicationMode: false })
      bridgeRef.current = bridge
      try {
        await bridge.start((b64) => { void engine.feed(base64Pcm16ToInt16(b64)) })
        console.log('[wakeword] mic bridge started, listening for wake word')
      } catch (err: any) {
        console.warn('[wakeword] mic bridge start failed:', err?.message || err)
      }
    })()

    return () => {
      cancelled = true
      const b = bridgeRef.current
      bridgeRef.current = null
      void b?.stop().catch(() => {})
    }
  }, [active])
}
