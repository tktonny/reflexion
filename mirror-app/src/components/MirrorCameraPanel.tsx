import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from 'react'
import { Pressable, StyleSheet, Text, View } from 'react-native'
import { CameraView, useCameraPermissions } from 'expo-camera'
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator'

// Live mirror preview + periodic frame sampling for the multimodal cognitive screening.
// While `active`, grabs a frame every SAMPLE_INTERVAL_MS (front camera), DOWNSCALES it to a small
// JPEG (~512px, tens of KB) and pushes it to a ring buffer; the parent reads them via the
// imperative handle at assessment time. Downscaling is essential: takePictureAsync captures at full
// sensor resolution (native) or full-res PNG (web ignores `quality`), so 6 raw frames would be
// multi-MB and could 413 / break the assessment. Works on web + native (expo-camera CameraView).

export type MirrorCameraHandle = {
  getFrames: () => string[]
  reset: () => void
  /**
   * Newest frame as RAW base64 JPEG (no data: prefix), or null if none captured yet. Used by the realtime
   * conversation to feed the model live video; returns the frame ONCE, so a caller polling faster than the
   * camera samples never sends the same picture twice.
   */
  takeLatestFrame: () => string | null
}

const SAMPLE_INTERVAL_MS = 8000
const MAX_FRAMES = 6
const FRAME_WIDTH = 512
const JPEG_QUALITY = 0.4

type Props = {
  active: boolean
  autoRequestPermission?: boolean
  /** Sampling period. The realtime path passes a much shorter one than the archival screening default. */
  intervalMs?: number
}

export const MirrorCameraPanel = forwardRef<MirrorCameraHandle, Props>(function MirrorCameraPanel({ active, autoRequestPermission = true, intervalMs }, ref) {
  const [permission, requestPermission] = useCameraPermissions()
  const camRef = useRef<CameraView | null>(null)
  const framesRef = useRef<string[]>([])
  // Newest frame, kept separately from the archival ring buffer and consumed once (see takeLatestFrame).
  const latestFrameRef = useRef<string | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const capturingRef = useRef(false)

  useImperativeHandle(
    ref,
    () => ({
      getFrames: () => framesRef.current.slice(),
      reset: () => { framesRef.current = []; latestFrameRef.current = null },
      takeLatestFrame: () => { const f = latestFrameRef.current; latestFrameRef.current = null; return f },
    }),
    [],
  )

  const captureFrame = useCallback(async () => {
    if (capturingRef.current || !camRef.current) return
    capturingRef.current = true
    try {
      // Capture the URI, then resize+recompress to a small JPEG so the frame is tens of KB on both
      // web and native (takePictureAsync's own `quality` does not reduce resolution, and web ignores it).
      const pic = await camRef.current.takePictureAsync({ quality: 1 })
      if (pic?.uri) {
        const small = await manipulateAsync(pic.uri, [{ resize: { width: FRAME_WIDTH } }], {
          compress: JPEG_QUALITY,
          format: SaveFormat.JPEG,
          base64: true,
        })
        if (small.base64) {
          // Raw base64 for the realtime feed; the archival buffer keeps the data: form it already used.
          latestFrameRef.current = small.base64
          const arr = framesRef.current
          arr.push(`data:image/jpeg;base64,${small.base64}`)
          if (arr.length > MAX_FRAMES) arr.shift()
        }
      }
    } catch {
      /* transient capture error (preview not ready / device busy) — skip this tick */
    } finally {
      capturingRef.current = false
    }
  }, [])

  useEffect(() => {
    if (active && autoRequestPermission && permission && !permission.granted && permission.canAskAgain) {
      void requestPermission()
    }
  }, [active, autoRequestPermission, permission, requestPermission])

  useEffect(() => {
    if (!active || !permission?.granted) {
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
      return
    }
    void captureFrame() // one immediately, then on interval
    timerRef.current = setInterval(() => { void captureFrame() }, (intervalMs ?? SAMPLE_INTERVAL_MS))
    return () => {
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
    }
  }, [active, permission?.granted, captureFrame])

  if (!permission) return <View style={styles.panel} />
  if (!permission.granted) {
    return (
      <View style={[styles.panel, styles.centered]}>
        <Text style={styles.hint}>Today’s conversation needs camera access</Text>
        <Pressable style={styles.btn} onPress={() => void requestPermission()}>
          <Text style={styles.btnText}>Allow camera</Text>
        </Pressable>
      </View>
    )
  }
  return (
    <View style={styles.panel}>
      <CameraView ref={camRef} style={styles.cam} facing="front" animateShutter={false} />
      {active ? <Text style={styles.rec}>● CAMERA ACTIVE</Text> : <Text style={styles.idle}>Mirror preview</Text>}
    </View>
  )
})

const styles = StyleSheet.create({
  panel: { height: 180, borderRadius: 12, overflow: 'hidden', backgroundColor: '#20201e', position: 'relative' },
  centered: { alignItems: 'center', justifyContent: 'center', gap: 10 },
  cam: { flex: 1 },
  rec: { position: 'absolute', top: 8, left: 10, color: '#FF6B6B', fontSize: 12, fontWeight: '900' },
  idle: { position: 'absolute', top: 8, left: 10, color: '#FFFFFF', fontSize: 12, fontWeight: '800', opacity: 0.7 },
  hint: { color: '#E7CFA6', fontSize: 14, fontWeight: '700' },
  btn: { backgroundColor: '#C89755', borderRadius: 8, paddingHorizontal: 16, paddingVertical: 10 },
  btnText: { color: '#FFFFFF', fontSize: 14, fontWeight: '900' },
})
