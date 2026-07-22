import { useEffect } from 'react'
import { Stack } from 'expo-router'
import { StatusBar } from 'expo-status-bar'

import { startDeviceHeartbeat } from '../src/api/deviceHeartbeat'
import { runHardwareChecks } from '../src/lib/hardwareCheck'
import { mirrorColors } from '../src/theme/mirrorTheme'

export default function RootLayout() {
  // Auto hardware self-check on every launch. Logs the readiness report to the console so a
  // real mirror reports its own hardware status at startup (no physical device needed to wire it).
  useEffect(() => {
    let stopHeartbeat: (() => void) | undefined
    void runHardwareChecks().then((r) => {
      console.log(`[hardware] platform=${r.platform} recommendedMode=${r.recommendedMode} (${r.recommendedReason})`)
      for (const c of r.checks) console.log(`[hardware] ${c.status.toUpperCase().padEnd(7)} ${c.label}: ${c.detail}`)
      stopHeartbeat = startDeviceHeartbeat(r)
    })
    return () => stopHeartbeat?.()
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
        <Stack.Screen name="test-device" options={{ headerShown: false }} />
        <Stack.Screen name="realtime-test" options={{ headerShown: false }} />
        <Stack.Screen name="hardware-check" options={{ headerShown: false }} />
        <Stack.Screen name="visual-acceptance" options={{ headerShown: false }} />
      </Stack>
    </>
  )
}
