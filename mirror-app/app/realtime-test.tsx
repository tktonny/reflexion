import { useMemo } from 'react'
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'

import { useConversation } from '../src/hooks/useConversation'

// Standalone verification screen for the Qwen realtime voice loop.
// No pairing / MongoDB required — connects to the local Node relay with a demo patient.
// Open http://localhost:8081/realtime-test (Expo web) with the relay running on :8787.
export default function RealtimeTestScreen() {
  const {
    mode,
    statusKind,
    statusText,
    messages,
    startConversation,
    stopConversation,
    connecting,
    sessionActive,
    userSpeaking,
    recording,
    toggleRecording,
  } = useConversation({ patientId: 'demo-patient', language: 'en' })

  const busy = connecting || sessionActive
  const statusColor = useMemo(() => {
    switch (statusKind) {
      case 'error': return '#C97068'
      case 'speaking': return '#C89755'
      case 'listening': return '#4D9668'
      case 'processing': return '#8E7F6D'
      default: return '#686868'
    }
  }, [statusKind])

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.stage}>
        <Text style={styles.title}>Qwen Realtime — Verify</Text>
        <Text style={[styles.status, { color: statusColor }]}>
          {statusText}{userSpeaking ? ' 🎤' : ''}
        </Text>

        <Text style={styles.modeTag}>version: {mode}</Text>

        <Pressable
          onPress={() => (busy ? void stopConversation() : void startConversation())}
          style={[styles.button, busy ? styles.stopButton : styles.startButton]}
        >
          <Text style={styles.buttonText}>{busy ? 'Stop' : 'Start conversation'}</Text>
        </Pressable>

        {busy && mode === 'http' && toggleRecording ? (
          <Pressable onPress={toggleRecording} style={[styles.button, recording ? styles.stopButton : styles.talkButton]}>
            <Text style={styles.buttonText}>{recording ? '发送（结束说话）' : '🎤 开始说话'}</Text>
          </Pressable>
        ) : null}

        <ScrollView style={styles.transcript} contentContainerStyle={styles.transcriptInner}>
          {messages.length === 0 ? (
            <Text style={styles.hint}>Press start, allow the microphone, and speak. Aria opens the check-in.</Text>
          ) : (
            messages.map((m) => (
              <View key={m.id} style={styles.row}>
                <Text style={[styles.role, m.role === 'assistant' ? styles.roleAi : m.role === 'user' ? styles.roleUser : styles.roleSys]}>
                  {m.role === 'assistant' ? 'Aria' : m.role === 'user' ? 'You' : 'System'}
                </Text>
                <Text style={styles.line}>{m.text}{m.streaming ? '…' : ''}</Text>
              </View>
            ))
          )}
        </ScrollView>

        <Text style={styles.footnote}>Relay: {process.env.EXPO_PUBLIC_RELAY_WS_URL || 'ws://<host>:8787'} · patient=demo-patient</Text>
      </View>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  safe: { backgroundColor: '#FFF9F1', flex: 1 },
  stage: { flex: 1, padding: 20, gap: 14, maxWidth: 640, width: '100%', alignSelf: 'center' },
  title: { color: '#282828', fontSize: 26, fontWeight: '900' },
  status: { fontSize: 16, fontWeight: '800' },
  button: { alignItems: 'center', borderRadius: 10, justifyContent: 'center', minHeight: 52 },
  startButton: { backgroundColor: '#C89755' },
  talkButton: { backgroundColor: '#4D9668' },
  stopButton: { backgroundColor: '#C97068' },
  modeTag: { color: '#8E7F6D', fontSize: 12, fontWeight: '800' },
  buttonText: { color: '#FFFFFF', fontSize: 16, fontWeight: '900' },
  transcript: { backgroundColor: '#FFFBF4', borderColor: '#F1E5D2', borderRadius: 10, borderWidth: 1, flex: 1 },
  transcriptInner: { gap: 12, padding: 14 },
  hint: { color: '#8E7F6D', fontSize: 15, fontWeight: '700' },
  row: { gap: 2 },
  role: { fontSize: 12, fontWeight: '900', textTransform: 'uppercase' },
  roleAi: { color: '#C89755' },
  roleUser: { color: '#4D9668' },
  roleSys: { color: '#BBAFA0' },
  line: { color: '#282828', fontSize: 16, fontWeight: '600', lineHeight: 22 },
  footnote: { color: '#BBAFA0', fontSize: 11, fontWeight: '700' },
})
