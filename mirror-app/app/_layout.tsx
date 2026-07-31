import { useEffect } from 'react'
import { Stack } from 'expo-router'
import { StatusBar } from 'expo-status-bar'

import { startDeviceHeartbeat, subscribeDeviceHeartbeat } from '../src/api/deviceHeartbeat'
import { flushPendingConversations } from '../src/storage/conversationQueue'
import { runHardwareChecks, runSpeakerProbe, speakerProbeIsStale } from '../src/lib/hardwareCheck'
import { mirrorColors } from '../src/theme/mirrorTheme'

export default function RootLayout() {
  // Auto hardware self-check on every launch. Logs the readiness report to the console so a
  // real mirror reports its own hardware status at startup (no physical device needed to wire it).
  useEffect(() => {
    let stopHeartbeat: (() => void) | undefined
    // Verify the speaker for REAL before reporting anything about it. This is the one moment we know the
    // audio device is idle (the wake-word listener and the conversation both come later), and it has to
    // run BEFORE runHardwareChecks so the report — which the heartbeat then captures for the whole
    // session — carries a measured verdict instead of an assumption. The probe is audible, so it
    // self-limits to once per 24h via its cache and never runs on web / without the native module.
    const probeIfStale = speakerProbeIsStale()
      .then((stale) => (stale ? runSpeakerProbe() : null))
      .catch(() => null)
    void probeIfStale.then(() => runHardwareChecks()).then((r) => {
      console.log(`[hardware] platform=${r.platform} recommendedMode=${r.recommendedMode} (${r.recommendedReason})`)
      for (const c of r.checks) console.log(`[hardware] ${c.status.toUpperCase().padEnd(7)} ${c.label}: ${c.detail}`)
      stopHeartbeat = startDeviceHeartbeat(r)
    })
    // Drain any sessions that were queued offline: once on launch, then again whenever the heartbeat
    // reports the backend is reachable again (network reconnect), with a light guard against overlap.
    let flushing = false
    let lastState = ''
    const runFlush = () => {
      if (flushing) return
      flushing = true
      void flushPendingConversations().catch(() => undefined).finally(() => { flushing = false })
    }
    runFlush()
    const unsubscribe = subscribeDeviceHeartbeat((state) => {
      if (state === 'online' && lastState !== 'online') runFlush()
      lastState = state
    })
    return () => { stopHeartbeat?.(); unsubscribe() }
  }, [])

  return (
    <>
      <StatusBar style="dark" />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: mirrorColors.cream },
          headerShadowVisible: false,
          headerTintColor: mirrorColors.text,
          headerTitleStyle: { fontWeight: '700' },
          contentStyle: { backgroundColor: mirrorColors.cream },
        }}
      >
        <Stack.Screen name="index" options={{ headerShown: false }} />
        <Stack.Screen name="conversation" options={{ headerShown: false }} />
        <Stack.Screen name="conversation-closing" options={{ headerShown: false }} />
        <Stack.Screen name="settings" options={{ headerShown: false }} />
        <Stack.Screen name="network-setup" options={{ headerShown: false }} />
        <Stack.Screen name="test-device" options={{ headerShown: false }} />
        <Stack.Screen name="realtime-test" options={{ headerShown: false }} />
        <Stack.Screen name="hardware-check" options={{ headerShown: false }} />
        <Stack.Screen name="visual-acceptance" options={{ headerShown: false }} />
      </Stack>
    </>
  )
}
