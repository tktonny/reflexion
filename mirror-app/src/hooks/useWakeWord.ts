import { useEffect, useRef } from 'react'

import { createPcmAudioBridge, isNativePcmAvailable, type PcmAudioBridge } from '../native/pcmAudio'
import {
  base64Pcm16ToInt16,
  createWakeWordEngine,
  isWakeWordRuntimeAvailable,
  type WakeWordEngine,
} from '../native/wakeWord'

/**
 * Listen for the open-source wake word while `active`, firing onDetected once per detection.
 * Reuses the existing expo-pcm-audio capture (16 kHz PCM) to feed the openWakeWord ONNX engine.
 *
 * No-op unless native PCM + onnxruntime + the ONNX models are ALL present (see docs/WAKEWORD.md):
 * on any missing piece the caller keeps its tap-to-start / web-SpeechRecognition path. This is why
 * it degrades cleanly on the emulator (no mic) and on builds without onnxruntime linked.
 */
export function useWakeWord(active: boolean, onDetected: () => void): void {
  const onDetectedRef = useRef(onDetected)
  onDetectedRef.current = onDetected

  const bridgeRef = useRef<PcmAudioBridge | null>(null)
  const firedRef = useRef(false)

  useEffect(() => {
    if (!active || !isWakeWordRuntimeAvailable() || !isNativePcmAvailable()) return
    let cancelled = false
    firedRef.current = false

    void (async () => {
      const engine: WakeWordEngine | null = await createWakeWordEngine(() => {
        if (firedRef.current) return
        firedRef.current = true
        onDetectedRef.current()
      })
      if (cancelled || !engine) return
      const bridge = createPcmAudioBridge({ communicationMode: false })
      bridgeRef.current = bridge
      try {
        await bridge.start((b64) => { void engine.feed(base64Pcm16ToInt16(b64)) })
      } catch {
        // mic/permission unavailable — no wake word this session, tap still works.
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
