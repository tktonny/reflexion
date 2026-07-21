import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { router } from 'expo-router'

import { useConversation } from '../src/hooks/useConversation'
import {
  assessConversation,
  transcriptFromMessages,
  type ScreeningAssessment,
} from '../src/api/assess'
import { resolveOwnerIds, saveCheckin } from '../src/api/saveCheckin'
import { MirrorCameraPanel, type MirrorCameraHandle } from '../src/components/MirrorCameraPanel'

// Standalone test screen: real voice interaction + live transcript + a post-session
// cognitive-screening JUDGMENT. No pairing/MongoDB required (demo patient).
// Web: open /realtime-test in Chrome (mode relay needs `npm run relay`).
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
    ended,
    recording,
    toggleRecording,
  } = useConversation({ patientId: 'demo-patient', language: 'en' })

  const [assessment, setAssessment] = useState<ScreeningAssessment | null>(null)
  const [assessing, setAssessing] = useState(false)
  const [assessError, setAssessError] = useState('')
  const [saveNote, setSaveNote] = useState('')
  const sessionStartRef = useRef<Date | null>(null)
  const cameraRef = useRef<MirrorCameraHandle | null>(null)
  const finalizingRef = useRef(false)

  const busy = connecting || sessionActive
  const hasTurns = messages.some((m) => m.role === 'user' || m.role === 'assistant')

  const statusColor = useMemo(() => {
    switch (statusKind) {
      case 'error': return '#C97068'
      case 'speaking': return '#C89755'
      case 'listening': return '#4D9668'
      case 'processing': return '#8E7F6D'
      default: return '#686868'
    }
  }, [statusKind])

  const runAssessment = useCallback(async (): Promise<ScreeningAssessment | null> => {
    const transcript = transcriptFromMessages(messages)
    if (!transcript.trim()) { setAssessError('没有可评估的对话内容'); return null }
    setAssessing(true)
    setAssessError('')
    const frames = cameraRef.current?.getFrames() ?? []
    const r = await assessConversation(transcript, 'en', frames)
    setAssessing(false)
    if (r.success) { setAssessment(r.assessment); return r.assessment }
    setAssessError(`评估失败: ${r.reason}`)
    return null
  }, [messages])

  const onStart = useCallback(() => {
    setAssessment(null)
    setAssessError('')
    setSaveNote('')
    finalizingRef.current = false
    cameraRef.current?.reset()
    sessionStartRef.current = new Date()
    void startConversation()
  }, [startConversation])

  // Finalize once: stop the session, run the screening (transcript + frames), save. Shared by the
  // manual 结束并评估 button AND the auto-end path (Aria's goodbye -> `ended`), guarded so it runs once.
  const finalize = useCallback(async () => {
    if (finalizingRef.current) return
    finalizingRef.current = true
    await stopConversation()
    const a = await runAssessment()
    const ids = await resolveOwnerIds()
    const res = await saveCheckin({
      messages,
      startedAt: sessionStartRef.current ?? new Date(),
      endedAt: new Date(),
      nurseId: ids.nurseId,
      patientId: ids.patientId,
      deviceId: ids.deviceId,
      authToken: ids.authToken,
      language: ids.language,
      assessment: a,
    })
    setSaveNote(res.saved ? '✓ 已保存到后台(Conversation + 判断)' : `未入库: ${res.reason}(已进离线队列)`)
  }, [messages, runAssessment, stopConversation])

  const onEnd = finalize

  // Auto-finalize when the assistant delivers its closing goodbye (hands-free — no button needed).
  useEffect(() => {
    if (ended) void finalize()
  }, [ended, finalize])

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.stage}>
        <Text style={styles.title}>Reflexion 检查 · 测试</Text>
        <View style={styles.statusRow}>
          <Text style={[styles.status, { color: statusColor }]}>{statusText}{userSpeaking ? ' 🎤' : ''}</Text>
          <Text style={styles.modeTag}>版本: {mode}</Text>
        </View>
        <Pressable onPress={() => router.push('/hardware-check')}>
          <Text style={styles.linkText}>🔧 硬件自检</Text>
        </Pressable>

        <MirrorCameraPanel ref={cameraRef} active={sessionActive} />

        <Pressable
          onPress={() => (busy ? void onEnd() : onStart())}
          style={[styles.button, busy ? styles.stopButton : styles.startButton]}
        >
          <Text style={styles.buttonText}>{busy ? '结束并评估' : '开始对话'}</Text>
        </Pressable>

        {busy && mode === 'http' && toggleRecording ? (
          <Pressable onPress={toggleRecording} style={[styles.button, recording ? styles.stopButton : styles.talkButton]}>
            <Text style={styles.buttonText}>{recording ? '发送（结束说话）' : '🎤 开始说话'}</Text>
          </Pressable>
        ) : null}

        {!busy && hasTurns ? (
          <Pressable onPress={() => void runAssessment()} style={[styles.button, styles.assessButton]} disabled={assessing}>
            <Text style={styles.assessButtonText}>{assessing ? '评估中…' : '重新评估这段对话'}</Text>
          </Pressable>
        ) : null}

        <ScrollView style={styles.transcript} contentContainerStyle={styles.transcriptInner}>
          {messages.length === 0 ? (
            <Text style={styles.hint}>点"开始对话",允许麦克风,然后说话。Aria 会先开场,按 4 阶段推进。结束后给出判断。</Text>
          ) : (
            messages.map((m) => (
              <View key={m.id} style={styles.msgRow}>
                <Text style={[styles.role, m.role === 'assistant' ? styles.roleAi : m.role === 'user' ? styles.roleUser : styles.roleSys]}>
                  {m.role === 'assistant' ? 'Aria' : m.role === 'user' ? '你' : '系统'}
                </Text>
                <Text style={styles.line}>{m.text}{m.streaming ? '…' : ''}</Text>
              </View>
            ))
          )}

          {assessing ? (
            <View style={styles.assessing}><ActivityIndicator color="#C89755" /><Text style={styles.hint}>生成判断结果…</Text></View>
          ) : null}
          {assessError ? <Text style={styles.errText}>{assessError}</Text> : null}
          {assessment ? <AssessmentCard a={assessment} /> : null}
        </ScrollView>

        {saveNote ? <Text style={styles.saveNote}>{saveNote}</Text> : null}
        <Text style={styles.footnote}>relay: {process.env.EXPO_PUBLIC_RELAY_WS_URL || 'ws://<host>:8787'} · patient=demo-patient</Text>
      </View>
    </SafeAreaView>
  )
}

function AssessmentCard({ a }: { a: ScreeningAssessment }) {
  const tierColor = a.risk_tier === 'high' ? '#C97068' : a.risk_tier === 'medium' ? '#C89755' : '#4D9668'
  const pct = typeof a.risk_score === 'number' ? `${Math.round(a.risk_score * 100)}%` : '—'
  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>判断结果（研究筛查,非诊断）</Text>
      <View style={styles.badges}>
        <View style={[styles.badge, { backgroundColor: tierColor }]}>
          <Text style={styles.badgeText}>{(a.screening_classification || 'unknown').toUpperCase()}</Text>
        </View>
        <Text style={[styles.score, { color: tierColor }]}>风险 {pct} · {a.risk_tier || '—'}</Text>
      </View>
      {a.summary ? <Text style={styles.summary}>{a.summary}</Text> : null}
      <Section title="发现" items={a.findings} />
      <Section title="支持风险的证据" items={a.evidence_for_risk} color="#C97068" />
      <Section title="不支持风险的证据" items={a.evidence_against_risk} color="#4D9668" />
      <Section title="视觉观察（参与度/情绪/警觉,非诊断)" items={a.visual_observations ?? []} color="#5B8DBE" />
    </View>
  )
}

function Section({ title, items, color = '#686868' }: { title: string; items: string[]; color?: string }) {
  if (!items || items.length === 0) return null
  return (
    <View style={styles.section}>
      <Text style={[styles.sectionTitle, { color }]}>{title}</Text>
      {items.map((it, i) => (
        <Text key={i} style={styles.bullet}>• {it}</Text>
      ))}
    </View>
  )
}

const styles = StyleSheet.create({
  safe: { backgroundColor: '#FFF9F1', flex: 1 },
  stage: { flex: 1, padding: 20, gap: 12, maxWidth: 680, width: '100%', alignSelf: 'center' },
  title: { color: '#282828', fontSize: 24, fontWeight: '900' },
  statusRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  status: { fontSize: 16, fontWeight: '800' },
  modeTag: { color: '#8E7F6D', fontSize: 12, fontWeight: '800' },
  linkText: { color: '#4D9668', fontSize: 13, fontWeight: '700', marginTop: 4 },
  button: { alignItems: 'center', borderRadius: 10, justifyContent: 'center', minHeight: 52 },
  startButton: { backgroundColor: '#C89755' },
  talkButton: { backgroundColor: '#4D9668' },
  stopButton: { backgroundColor: '#C97068' },
  assessButton: { backgroundColor: '#FFFFFF', borderColor: '#C89755', borderWidth: 1, minHeight: 44 },
  assessButtonText: { color: '#C89755', fontSize: 15, fontWeight: '900' },
  buttonText: { color: '#FFFFFF', fontSize: 16, fontWeight: '900' },
  transcript: { backgroundColor: '#FFFBF4', borderColor: '#F1E5D2', borderRadius: 10, borderWidth: 1, flex: 1 },
  transcriptInner: { gap: 12, padding: 14 },
  hint: { color: '#8E7F6D', fontSize: 15, fontWeight: '700' },
  msgRow: { gap: 2 },
  role: { fontSize: 12, fontWeight: '900', textTransform: 'uppercase' },
  roleAi: { color: '#C89755' },
  roleUser: { color: '#4D9668' },
  roleSys: { color: '#BBAFA0' },
  line: { color: '#282828', fontSize: 16, fontWeight: '600', lineHeight: 22 },
  assessing: { flexDirection: 'row', gap: 8, alignItems: 'center', paddingVertical: 8 },
  errText: { color: '#C97068', fontSize: 14, fontWeight: '800' },
  card: { backgroundColor: '#FFF6EA', borderColor: '#E7CFA6', borderWidth: 1, borderRadius: 10, padding: 14, gap: 8, marginTop: 8 },
  cardTitle: { color: '#282828', fontSize: 15, fontWeight: '900' },
  badges: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  badge: { borderRadius: 6, paddingHorizontal: 8, paddingVertical: 4 },
  badgeText: { color: '#FFFFFF', fontSize: 13, fontWeight: '900' },
  score: { fontSize: 15, fontWeight: '900' },
  summary: { color: '#282828', fontSize: 15, fontWeight: '600', lineHeight: 21 },
  section: { gap: 2 },
  sectionTitle: { fontSize: 13, fontWeight: '900', marginTop: 4 },
  bullet: { color: '#3a3a3a', fontSize: 14, fontWeight: '600', lineHeight: 20 },
  saveNote: { color: '#4D9668', fontSize: 13, fontWeight: '800' },
  footnote: { color: '#BBAFA0', fontSize: 11, fontWeight: '700' },
})
