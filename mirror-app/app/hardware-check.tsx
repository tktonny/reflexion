import { useCallback, useEffect, useState } from 'react'
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { router } from 'expo-router'

import { runHardwareChecks, type CheckStatus, type HardwareReport } from '../src/lib/hardwareCheck'

// Startup hardware / capability self-check screen. Auto-runs on mount and on every launch
// (see app/_layout). Web results are real now; native hardware results appear once built to a
// device — the checks run automatically regardless, so a real mirror reports its own readiness.
export default function HardwareCheckScreen() {
  const [report, setReport] = useState<HardwareReport | null>(null)
  const [running, setRunning] = useState(false)

  const run = useCallback(async () => {
    setRunning(true)
    try {
      setReport(await runHardwareChecks())
    } finally {
      setRunning(false)
    }
  }, [])

  useEffect(() => { void run() }, [run])

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>硬件自检</Text>
        <Text style={styles.subtitle}>
          启动时自动运行 · 平台 {report?.platform ?? '…'}
        </Text>

        {running && !report ? (
          <View style={styles.loading}>
            <ActivityIndicator color="#4D9668" />
            <Text style={styles.loadingText}>正在探测硬件…</Text>
          </View>
        ) : null}

        {report ? (
          <>
            <View style={styles.card}>
              {report.checks.map((c) => (
                <View key={c.key} style={styles.row}>
                  <Text style={[styles.dot, { color: colorFor(c.status) }]}>{glyphFor(c.status)}</Text>
                  <View style={styles.rowText}>
                    <Text style={styles.rowLabel}>{c.label}</Text>
                    <Text style={styles.rowDetail}>{c.detail}</Text>
                  </View>
                </View>
              ))}
            </View>

            <View style={styles.recCard}>
              <Text style={styles.recTitle}>建议对话版本</Text>
              <Text style={styles.recMode}>{report.recommendedMode.toUpperCase()}</Text>
              <Text style={styles.recReason}>{report.recommendedReason}</Text>
              <Text style={styles.recConfigured}>
                当前配置(EXPO_PUBLIC_CONVERSATION_MODE): {report.configuredMode}
                {report.recommendedMode !== 'none' && report.recommendedMode !== report.configuredMode
                  ? ' ⚠️ 与建议不一致'
                  : ' ✓'}
              </Text>
            </View>
          </>
        ) : null}

        <Pressable style={[styles.button, running && styles.buttonDisabled]} onPress={run} disabled={running}>
          <Text style={styles.buttonText}>{running ? '检查中…' : '重新检查'}</Text>
        </Pressable>
        <Pressable style={[styles.button, styles.buttonGhost]} onPress={() => router.push('/realtime-test')}>
          <Text style={[styles.buttonText, styles.buttonGhostText]}>前往对话测试</Text>
        </Pressable>
      </ScrollView>
    </SafeAreaView>
  )
}

function colorFor(s: CheckStatus): string {
  switch (s) {
    case 'ok': return '#4D9668'
    case 'warn': return '#C89755'
    case 'fail': return '#C97068'
    default: return '#8E7F6D'
  }
}
function glyphFor(s: CheckStatus): string {
  switch (s) {
    case 'ok': return '✓'
    case 'warn': return '!'
    case 'fail': return '✕'
    default: return '?'
  }
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#f7fbfa' },
  content: { padding: 24, gap: 16 },
  title: { fontSize: 28, fontWeight: '700', color: '#173a40' },
  subtitle: { fontSize: 14, color: '#5c6b6b' },
  loading: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 24 },
  loadingText: { color: '#5c6b6b' },
  card: { backgroundColor: '#ffffff', borderRadius: 16, padding: 8, gap: 2 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingVertical: 12, paddingHorizontal: 12 },
  dot: { fontSize: 20, fontWeight: '800', width: 24, textAlign: 'center' },
  rowText: { flex: 1 },
  rowLabel: { fontSize: 16, fontWeight: '600', color: '#173a40' },
  rowDetail: { fontSize: 13, color: '#5c6b6b', marginTop: 2 },
  recCard: { backgroundColor: '#eef6f2', borderRadius: 16, padding: 16, gap: 4 },
  recTitle: { fontSize: 13, color: '#5c6b6b', fontWeight: '600' },
  recMode: { fontSize: 24, fontWeight: '800', color: '#173a40' },
  recReason: { fontSize: 14, color: '#3f5150' },
  recConfigured: { fontSize: 13, color: '#5c6b6b', marginTop: 6 },
  button: { backgroundColor: '#4D9668', borderRadius: 14, paddingVertical: 16, alignItems: 'center' },
  buttonDisabled: { opacity: 0.6 },
  buttonText: { color: '#ffffff', fontSize: 16, fontWeight: '700' },
  buttonGhost: { backgroundColor: 'transparent', borderWidth: 1, borderColor: '#4D9668' },
  buttonGhostText: { color: '#4D9668' },
})
