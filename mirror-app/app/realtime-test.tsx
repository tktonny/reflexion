import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { router } from 'expo-router'

import { useConversation } from '../src/hooks/useConversation'
import { VERSION_LABELS } from '../src/config/conversationMode'
import {
  assessConversation,
  transcriptFromMessages,
  type ScreeningAssessment,
} from '../src/api/assess'
import { resolveOwnerIds, saveCheckin } from '../src/api/saveCheckin'
import { beginMirrorSession } from '../src/api/sessionSync'
import { MirrorCameraPanel, type MirrorCameraHandle } from '../src/components/MirrorCameraPanel'
import { isCognitiveAssessmentEligible } from '../src/orchestration/conversationPurpose'

// Standalone test screen: real voice interaction + live transcript + a post-session
// cognitive-screening JUDGMENT. No pairing/MongoDB required (demo patient).
// Web: open /realtime-test in Chrome (mode relay needs `npm run relay`).
export default function RealtimeTestScreen() {
  const [persona, setPersona] = useState<'screening' | 'companion'>('companion')
  const [pushToTalk, setPushToTalk] = useState(false)
  const [pendingStart, setPendingStart] = useState(false)
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
    turnState,
    ended,
    recording,
    beginPushToTalk,
    endPushToTalk,
    toggleRecording,
  } = useConversation({ patientId: 'demo-patient', language: 'en', persona, pushToTalk })

  const [assessment, setAssessment] = useState<ScreeningAssessment | null>(null)
  const [assessing, setAssessing] = useState(false)
  const [assessError, setAssessError] = useState('')
  const [saveNote, setSaveNote] = useState('')
  const sessionStartRef = useRef<Date | null>(null)
  const cameraRef = useRef<MirrorCameraHandle | null>(null)
  const finalizingRef = useRef(false)
  const messagesRef = useRef(messages)
  messagesRef.current = messages

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

  const onStart = useCallback(async () => {
    setAssessment(null)
    setAssessError('')
    setSaveNote('')
    finalizingRef.current = false
    cameraRef.current?.reset()
    sessionStartRef.current = new Date()
    await beginMirrorSession(persona === 'screening' ? 'daily_checkin' : 'companion', 'en')
    await startConversation()
  }, [persona, startConversation])

  // Finalize once: stop the session, run the screening (transcript + frames), save. Shared by the
  // manual 结束并评估 button AND the auto-end path (Aria's goodbye -> `ended`), guarded so it runs once.
  const finalize = useCallback(async () => {
    if (finalizingRef.current) return
    finalizingRef.current = true
    await stopConversation()
    const finalMessages = messagesRef.current
    const transcript = transcriptFromMessages(finalMessages)
    let a: ScreeningAssessment | null = null
    if (isCognitiveAssessmentEligible(persona, finalMessages)) {
      setAssessing(true)
      setAssessError('')
      const result = await assessConversation(transcript, 'en', cameraRef.current?.getFrames() ?? [])
      setAssessing(false)
      if (result.success) { a = result.assessment; setAssessment(result.assessment) }
      else setAssessError(`评估失败: ${result.reason}`)
    }
    const ids = await resolveOwnerIds()
    const res = await saveCheckin({
      messages: finalMessages,
      startedAt: sessionStartRef.current ?? new Date(),
      endedAt: new Date(),
      nurseId: ids.nurseId,
      patientId: ids.patientId,
      deviceId: ids.deviceId,
      language: ids.language,
      assessment: a,
    })
    setSaveNote(res.saved ? '✓ 已保存到后台(Conversation + 判断)' : `未入库: ${res.reason}(已进离线队列)`)
  }, [persona, stopConversation])

  const onEnd = finalize

  // Auto-finalize when the assistant delivers its closing goodbye (hands-free — no button needed).
  useEffect(() => {
    if (ended) void finalize()
  }, [ended, finalize])

  // Two entry buttons set the persona, then start once the hook has re-bound to it on the next
  // render (React state isn't synchronous, so the start is deferred via pendingStart).
  useEffect(() => {
    if (pendingStart && !busy) { setPendingStart(false); onStart() }
  }, [pendingStart, busy, onStart])

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.stage}>
        <Text style={styles.title}>Reflexion 检查 · 测试</Text>
        <View style={styles.statusRow}>
          <Text style={[styles.status, { color: statusColor }]}>{statusText}{userSpeaking ? ' 🎤' : ''}</Text>
          <Text style={styles.modeTag}>版本: {VERSION_LABELS[mode] ?? mode} · {persona === 'companion' ? '日常助手' : '认知检查'}{turnState ? ` · ${turnState}` : ''}</Text>
        </View>
        <Pressable onPress={() => router.push('/hardware-check')}>
          <Text style={styles.linkText}>🔧 硬件自检</Text>
        </Pressable>

        <MirrorCameraPanel ref={cameraRef} active={sessionActive} />

        {busy ? (
          <Pressable onPress={() => void onEnd()} style={[styles.button, styles.stopButton]}>
            <Text style={styles.buttonText}>{persona === 'screening' ? '结束并评估' : '结束对话'}</Text>
          </Pressable>
        ) : (
          <View style={{ gap: 10 }}>
            <Pressable onPress={() => setPushToTalk((v) => !v)} style={styles.pttToggle}>
              <Text style={styles.pttToggleText}>
                {pushToTalk
                  ? '🎙 手动 · 按住说话（模拟器/嘈杂环境，防回声）— 点此切免手'
                  : '🖐 免手 · 自动断句（真机推荐）— 点此切手动'}
              </Text>
            </Pressable>
            <View style={styles.startRow}>
              <Pressable onPress={() => { setPersona('companion'); setPendingStart(true) }} style={[styles.button, styles.talkButton, styles.halfButton]}>
                <Text style={styles.buttonText}>🗣 日常助手</Text>
              </Pressable>
              <Pressable onPress={() => { setPersona('screening'); setPendingStart(true) }} style={[styles.button, styles.startButton, styles.halfButton]}>
                <Text style={styles.buttonText}>🧠 认知检查</Text>
              </Pressable>
            </View>
          </View>
        )}

        {busy && beginPushToTalk && endPushToTalk ? (
          <Pressable
            disabled={!recording && statusKind !== 'listening'}
            onPressIn={beginPushToTalk}
            onPressOut={endPushToTalk}
            style={[styles.button, recording ? styles.stopButton : styles.talkButton]}
          >
            <Text style={styles.buttonText}>{recording ? '🎙 正在听 · 松开发送' : '🎤 按住说话'}</Text>
          </Pressable>
        ) : null}

        {!busy && hasTurns && persona === 'screening' ? (
          <Pressable onPress={() => void runAssessment()} style={[styles.button, styles.assessButton]} disabled={assessing}>
            <Text style={styles.assessButtonText}>{assessing ? '评估中…' : '重新评估这段对话'}</Text>
          </Pressable>
        ) : null}

        <ScrollView style={styles.transcript} contentContainerStyle={styles.transcriptInner}>
          {messages.length === 0 ? (
            <Text style={styles.hint}>
              {persona === 'screening'
                ? '开始认知检查后，Aria 会用 7 个自然问题推进，结束后生成研究筛查判断。'
                : '开始日常对话后，可以询问天气、日程、用药提醒或随意聊天；不会生成认知风险判断。'}
            </Text>
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
  startRow: { flexDirection: 'row', gap: 10 },
  pttToggle: { alignItems: 'center', backgroundColor: '#FFF4E2', borderColor: '#F0DEC1', borderRadius: 8, borderWidth: 1, justifyContent: 'center', minHeight: 40, paddingHorizontal: 12 },
  pttToggleText: { color: '#8E7F6D', fontSize: 13, fontWeight: '800', textAlign: 'center' },
  halfButton: { flex: 1 },
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
