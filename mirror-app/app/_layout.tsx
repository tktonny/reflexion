import { Stack } from 'expo-router'
import { StatusBar } from 'expo-status-bar'

export default function RootLayout() {
  return (
    <>
      <StatusBar style="dark" />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: '#f7fbfa' },
          headerShadowVisible: false,
          headerTintColor: '#173a40',
          headerTitleStyle: { fontWeight: '700' },
          contentStyle: { backgroundColor: '#f7fbfa' },
        }}
      >
        <Stack.Screen name="index" options={{ headerShown: false }} />
        <Stack.Screen name="conversation" options={{ headerShown: false }} />
        <Stack.Screen name="conversation-closing" options={{ headerShown: false }} />
        <Stack.Screen name="settings" options={{ headerShown: false }} />
        <Stack.Screen name="test-device" options={{ headerShown: false }} />
        <Stack.Screen name="realtime-test" options={{ headerShown: false }} />
      </Stack>
    </>
  )
}
