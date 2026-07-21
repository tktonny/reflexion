import AsyncStorage from '@react-native-async-storage/async-storage'
import { Ionicons } from '@expo/vector-icons'
import { router } from 'expo-router'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ActivityIndicator,
  Animated,
  Easing,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'

import { getApiUrl } from './apiUrl'
import {
  ACTIVE_MIRROR_ID_STORAGE_KEY,
  ACTIVE_NURSE_ID_STORAGE_KEY,
  ACTIVE_PATIENT_ID_STORAGE_KEY,
  DEVICE_AUTH_TOKEN_STORAGE_KEY,
  DEVICE_ID_STORAGE_KEY,
  MIRROR_LANGUAGE_STORAGE_KEY,
  MIRROR_TIMEZONE_STORAGE_KEY,
  NURSE_PATIENT_CONFIG_STORAGE_KEY,
} from '../src/constants/nursePatientConfig'
import { type ChatMessage } from '../src/hooks/conversationTypes'
import { useConversation } from '../src/hooks/useConversation'
import { DEFAULT_LANGUAGE } from '../src/config/conversationMode'
import { assessCheckin, transcriptFromMessages, type ScreeningAssessment } from '../src/api/assess'
import { resolveOwnerIds, saveCheckin } from '../src/api/saveCheckin'
import { MirrorCameraPanel, type MirrorCameraHandle } from '../src/components/MirrorCameraPanel'
import { useWakeWord } from '../src/hooks/useWakeWord'
import { getStoredMirrorProfile, persistNursePatientIds } from '../src/storage/mirrorStorage'

type DevicePairingStatus =
  | {
      success: true
      paired: true
      deviceId: string
      authToken: string
      nurseId?: string
      patientId?: string
      language?: string
      timezone?: string
      nursePatientConfig: unknown
    }
  | { success: true; paired: false }
  | { success: false; reason: string }

type MirrorState = 'idle' | 'starting' | 'listening' | 'response' | 'error'

type SpeechRecognitionResultLike = {
  isFinal: boolean
  length: number
  [index: number]: { transcript: string }
}

type SpeechRecognitionEventLike = {
  resultIndex: number
  results: {
    length: number
    [index: number]: SpeechRecognitionResultLike
  }
}

type SpeechRecognitionLike = {
  continuous: boolean
  interimResults: boolean
  lang: string
  onend: (() => void) | null
  onerror: ((event: { error?: string }) => void) | null
  onresult: ((event: SpeechRecognitionEventLike) => void) | null
  start: () => void
  stop: () => void
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike

function getSpeechRecognitionConstructor(): SpeechRecognitionConstructor | null {
  if (Platform.OS !== 'web' || typeof window === 'undefined') return null
  const speechWindow = window as typeof window & {
    SpeechRecognition?: SpeechRecognitionConstructor
    webkitSpeechRecognition?: SpeechRecognitionConstructor
  }
  return speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition ?? null
}

function getGreeting(date: Date) {
  const hour = date.getHours()
  if (hour < 12) return 'good morning'
  if (hour < 18) return 'good afternoon'
  return 'good evening'
}

function formatDate(date: Date) {
  return new Intl.DateTimeFormat('en-US', {
    day: 'numeric',
    month: 'long',
    weekday: 'long',
  }).format(date)
}

function formatTime(date: Date) {
  return new Intl.DateTimeFormat('en-US', {
    hour: 'numeric',
    minute: '2-digit',
  }).format(date)
}

function latestAssistantMessage(messages: ChatMessage[]) {
  return messages.filter((message) => message.role === 'assistant').at(-1)?.text.trim() ?? ''
}

export default function ConversationScreen() {
  const [language, setLanguage] = useState<string>(DEFAULT_LANGUAGE)
  // companion = everyday voice assistant (default, the common daily interaction); screening = the
  // structured cognitive check-in. Set by the idle-screen entries before starting.
  const [persona, setPersona] = useState<'screening' | 'companion'>('companion')
  const [pendingStart, setPendingStart] = useState(false)
  const {
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
  } = useConversation({ language, persona })

  const [now, setNow] = useState(() => new Date())
  const [patientName, setPatientName] = useState('Margaret')
  const [nurseName, setNurseName] = useState('your nurse')
  const [visibleAssistantText, setVisibleAssistantText] = useState('')
  const [endingConversation, setEndingConversation] = useState(false)
  const [checkingPairing, setCheckingPairing] = useState(true)
  const [wakeListening, setWakeListening] = useState(false)
  const [wakeError, setWakeError] = useState('')
  const responseOpacity = useRef(new Animated.Value(0)).current
  const endingRef = useRef(false)
  const wakeRecognitionRef = useRef<SpeechRecognitionLike | null>(null)
  const wakeTriggeredRef = useRef(false)
  const shouldRunWakeListenerRef = useRef(false)
  const cameraRef = useRef<MirrorCameraHandle | null>(null)
  const sessionStartRef = useRef<Date | null>(null)

  const assistantText = latestAssistantMessage(messages)
  const greeting = getGreeting(now)
  const busy = connecting || sessionActive

  const mirrorState: MirrorState = useMemo(() => {
    if (statusKind === 'error' && !endingConversation) return 'error'
    if (connecting) return 'starting'
    if (visibleAssistantText || statusKind === 'speaking') return 'response'
    if (sessionActive || statusKind === 'listening' || statusKind === 'processing') return 'listening'
    return 'idle'
  }, [connecting, endingConversation, sessionActive, statusKind, visibleAssistantText])

  useEffect(() => {
    const timerId = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(timerId)
  }, [])

  useEffect(() => {
    let mounted = true

    async function verifyAndLoadProfile() {
      const validPairing = await verifyStoredPairing()
      if (!mounted) return
      if (!validPairing) {
        await clearStoredMirrorConnection()
        router.replace('/')
        return
      }

      const profile = await getStoredMirrorProfile()
      if (!mounted) return
      setPatientName(profile.patientName ?? 'Margaret')
      setNurseName(profile.nurseName ?? 'your nurse')
      const storedLang = await AsyncStorage.getItem(MIRROR_LANGUAGE_STORAGE_KEY)
      if (mounted && storedLang && storedLang.trim()) setLanguage(storedLang.trim())
      setCheckingPairing(false)
    }

    void verifyAndLoadProfile()
    return () => {
      mounted = false
    }
  }, [])

  useEffect(() => {
    if (!assistantText) return
    setVisibleAssistantText(assistantText)
    Animated.timing(responseOpacity, {
      duration: 260,
      easing: Easing.out(Easing.cubic),
      toValue: 1,
      useNativeDriver: true,
    }).start()
  }, [assistantText, responseOpacity])

  useEffect(() => {
    if (checkingPairing || busy || endingConversation) {
      shouldRunWakeListenerRef.current = false
      stopWakeListener()
      return
    }

    shouldRunWakeListenerRef.current = true
    wakeTriggeredRef.current = false
    startWakeListener()

    return () => {
      shouldRunWakeListenerRef.current = false
      stopWakeListener()
    }
  }, [busy, checkingPairing, endingConversation])

  async function handleStart() {
    if (busy || checkingPairing) return
    shouldRunWakeListenerRef.current = false
    stopWakeListener()
    sessionStartRef.current = new Date()
    cameraRef.current?.reset()
    await startConversation()
  }

  // Pick a persona then start on the next render (state isn't synchronous, so the hook must re-bind
  // to the chosen persona before startConversation reads it). The wake word / plain tap use companion.
  function startWith(p: 'screening' | 'companion') {
    if (busy || checkingPairing) return
    setPersona(p)
    setPendingStart(true)
  }

  // End the check-in: stop the session, run the two-stage screening (transcript + camera frames),
  // persist the Conversation (+judgment; offline-queues on failure), THEN navigate to the closing
  // screen. Shared by the manual "End Chat" button and the auto-end path (Aria's goodbye -> `ended`).
  const finalize = useCallback(async () => {
    if (endingRef.current) return
    endingRef.current = true
    setEndingConversation(true)
    try {
      await stopConversation()
      const transcript = transcriptFromMessages(messages)
      let assessment: ScreeningAssessment | null = null
      if (transcript.trim()) {
        const frames = cameraRef.current?.getFrames() ?? []
        const r = await assessCheckin(transcript, language, frames)
        if (r.success) assessment = r.assessment
      }
      const ids = await resolveOwnerIds()
      await saveCheckin({
        messages,
        startedAt: sessionStartRef.current ?? new Date(),
        endedAt: new Date(),
        nurseId: ids.nurseId,
        patientId: ids.patientId,
        deviceId: ids.deviceId,
        authToken: ids.authToken,
        language: ids.language ?? language,
        assessment,
      })
      router.replace({ pathname: '/conversation-closing', params: { nurseName } })
    } catch {
      endingRef.current = false
      setEndingConversation(false)
    }
  }, [language, messages, nurseName, stopConversation])

  const handleEnd = finalize

  // Auto-finalize when the assistant delivers its closing goodbye (hands-free).
  useEffect(() => {
    if (ended) void finalize()
  }, [ended, finalize])

  // Start once the chosen persona has been applied to the hook on the next render.
  useEffect(() => {
    if (pendingStart && !busy) { setPendingStart(false); void handleStart() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingStart, busy])

  // Native open-source wake word while idle -> opens the everyday companion. No-op unless the
  // onnxruntime runtime + ONNX models are present (see docs/WAKEWORD.md); web uses SpeechRecognition.
  useWakeWord(
    Platform.OS !== 'web' && !checkingPairing && !busy && !endingConversation,
    () => startWith('companion'),
  )

  function startWakeListener() {
    if (Platform.OS !== 'web' || wakeRecognitionRef.current) return
    const SpeechRecognition = getSpeechRecognitionConstructor()
    if (!SpeechRecognition) {
      setWakeListening(false)
      setWakeError('Voice start is unavailable in this browser.')
      return
    }

    const recognition = new SpeechRecognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-US'
    recognition.onresult = (event) => {
      let transcript = ''
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index]
        for (let alt = 0; alt < result.length; alt += 1) {
          transcript += ` ${result[alt]?.transcript ?? ''}`
        }
      }

      if (/\bhello\b/i.test(transcript) && !wakeTriggeredRef.current) {
        wakeTriggeredRef.current = true
        shouldRunWakeListenerRef.current = false
        stopWakeListener()
        startWith('companion') // wake word opens the everyday assistant
      }
    }
    recognition.onerror = (event) => {
      setWakeListening(false)
      setWakeError(event.error ? `Voice start error: ${event.error}` : 'Voice start paused.')
    }
    recognition.onend = () => {
      wakeRecognitionRef.current = null
      setWakeListening(false)
      if (shouldRunWakeListenerRef.current && !wakeTriggeredRef.current) {
        window.setTimeout(() => startWakeListener(), 350)
      }
    }

    try {
      recognition.start()
      wakeRecognitionRef.current = recognition
      setWakeListening(true)
      setWakeError('')
    } catch {
      wakeRecognitionRef.current = null
      setWakeListening(false)
    }
  }

  function stopWakeListener() {
    const recognition = wakeRecognitionRef.current
    if (!recognition) {
      setWakeListening(false)
      return
    }
    wakeRecognitionRef.current = null
    recognition.onend = null
    recognition.onresult = null
    recognition.onerror = null
    try {
      recognition.stop()
    } catch {
      /* noop */
    }
    setWakeListening(false)
  }

  if (checkingPairing) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <View style={styles.stage}>
          <View style={styles.mirrorCard}>
            <View style={styles.fillCenter}>
              <ActivityIndicator color={colors.taupe} />
              <Text style={styles.verifyingText}>Checking mirror connection...</Text>
            </View>
          </View>
        </View>
      </SafeAreaView>
    )
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.stage}>
        <View style={styles.mirrorCard}>
          {busy ? (
            <View style={styles.cameraCorner}>
              <MirrorCameraPanel ref={cameraRef} active={sessionActive} />
            </View>
          ) : null}

          {mirrorState === 'idle' ? (
            <IdleScreen
              date={formatDate(now)}
              greeting={greeting}
              onStart={() => startWith('companion')}
              onScreening={() => startWith('screening')}
              patientName={patientName}
              time={formatTime(now)}
              wakeError={wakeError}
              wakeListening={wakeListening}
            />
          ) : null}

          {mirrorState === 'starting' ? (
            <StartScreen greeting={greeting} patientName={patientName} />
          ) : null}

          {mirrorState === 'listening' ? (
            <ListeningScreen userSpeaking={userSpeaking} />
          ) : null}

          {mirrorState === 'response' ? (
            <ResponseScreen
              assistantText={visibleAssistantText || 'How did you sleep last night?'}
              disabled={endingConversation}
              onEnd={() => void handleEnd()}
              opacity={responseOpacity}
              raiseEnd={!!toggleRecording}
              userSpeaking={userSpeaking}
            />
          ) : null}

          {mirrorState === 'error' ? (
            <ErrorScreen message={statusText} onRetry={() => void handleStart()} />
          ) : null}

          {/* Fallback (v2 turn-based) is push-to-talk — show a record control so the degraded path
              is usable on the hands-free screen. Absent in the omni (v3) path (no toggleRecording). */}
          {toggleRecording && busy && !endingConversation && mirrorState !== 'error' ? (
            <Pressable
              style={[styles.recordButton, recording && styles.recordButtonActive]}
              onPress={() => toggleRecording?.()}
            >
              <Ionicons name={recording ? 'send' : 'mic'} size={22} color="#FFFFFF" />
              <Text style={styles.recordButtonText}>{recording ? '发送（结束说话）' : '按此说话'}</Text>
            </Pressable>
          ) : null}
        </View>
      </View>
    </SafeAreaView>
  )
}

async function verifyStoredPairing() {
  const deviceId = await AsyncStorage.getItem(DEVICE_ID_STORAGE_KEY)
  if (!deviceId) return false

  const [authToken, config] = await Promise.all([
    AsyncStorage.getItem(DEVICE_AUTH_TOKEN_STORAGE_KEY),
    AsyncStorage.getItem(NURSE_PATIENT_CONFIG_STORAGE_KEY),
  ])
  const hasLocalPairing = Boolean(authToken && config)

  try {
    const response = await fetch(getApiUrl('/api/mirror-pairing/device-status'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ deviceId }),
    })
    const status = (await response.json()) as DevicePairingStatus
    if (status.success && status.paired) {
      await persistPairedMirror(status)
      return true
    }
    // AUTHORITATIVE "not paired" only when the server clearly says so — then clear.
    if (status.success && !status.paired) return false
    // Server error / unexpected shape = treat as unreachable: keep the local pairing (offline).
    return hasLocalPairing
  } catch {
    // Network unreachable — do NOT wipe pairing on a transient blip; run offline if paired locally.
    return hasLocalPairing
  }
}

async function persistPairedMirror(status: Extract<DevicePairingStatus, { paired: true }>) {
  await AsyncStorage.multiSet([
    [DEVICE_ID_STORAGE_KEY, status.deviceId],
    [ACTIVE_MIRROR_ID_STORAGE_KEY, status.deviceId],
    [DEVICE_AUTH_TOKEN_STORAGE_KEY, status.authToken],
    [NURSE_PATIENT_CONFIG_STORAGE_KEY, JSON.stringify(status.nursePatientConfig)],
    [MIRROR_LANGUAGE_STORAGE_KEY, status.language || DEFAULT_LANGUAGE],
    [MIRROR_TIMEZONE_STORAGE_KEY, status.timezone || Intl.DateTimeFormat().resolvedOptions().timeZone],
  ])
  if (status.nurseId && status.patientId) {
    await AsyncStorage.multiSet([
      [ACTIVE_NURSE_ID_STORAGE_KEY, status.nurseId],
      [ACTIVE_PATIENT_ID_STORAGE_KEY, status.patientId],
    ])
  } else {
    await persistNursePatientIds(status.nursePatientConfig, status.deviceId)
  }
}

async function clearStoredMirrorConnection() {
  await AsyncStorage.multiRemove([
    ACTIVE_MIRROR_ID_STORAGE_KEY,
    ACTIVE_NURSE_ID_STORAGE_KEY,
    ACTIVE_PATIENT_ID_STORAGE_KEY,
    DEVICE_AUTH_TOKEN_STORAGE_KEY,
    NURSE_PATIENT_CONFIG_STORAGE_KEY,
    MIRROR_LANGUAGE_STORAGE_KEY,
    MIRROR_TIMEZONE_STORAGE_KEY,
  ])
}

function IdleScreen({
  date,
  greeting,
  onStart,
  onScreening,
  patientName,
  time,
  wakeError,
  wakeListening,
}: {
  date: string
  greeting: string
  onStart: () => void
  onScreening: () => void
  patientName: string
  time: string
  wakeError: string
  wakeListening: boolean
}) {
  // Tapping anywhere (or the "Hello" wake word) opens the everyday companion; the pill starts the
  // structured cognitive check-in.
  return (
    <Pressable style={styles.fillCenter} onPress={onStart}>
      <View style={styles.idleCopy}>
        <Text style={styles.greeting}>Good {greeting.replace('good ', '')},</Text>
        <Text style={styles.patientName}>{patientName}</Text>
      </View>
      <View style={styles.dateBlock}>
        <Text style={styles.date}>{date}</Text>
        <Text style={styles.time}>{time}</Text>
      </View>
      <View style={styles.startPrompt}>
        <Text style={styles.promptMain}>Say “Hello”</Text>
        <Text style={styles.promptSub}>to chat with Aria</Text>
        <Text style={styles.wakeStatus}>{wakeListening ? 'Listening for Hello…' : wakeError || 'Or tap anywhere to start'}</Text>
      </View>
      <Pressable onPress={onScreening} style={styles.screeningPill} hitSlop={10}>
        <Ionicons name="pulse-outline" size={18} color={colors.goldDark} />
        <Text style={styles.screeningPillText}>认知检查 · Daily check</Text>
      </Pressable>
    </Pressable>
  )
}

function StartScreen({ greeting, patientName }: { greeting: string; patientName: string }) {
  return (
    <View style={styles.fillCenter}>
      <AriaAvatar size={160} />
      <View style={styles.startCopy}>
        <Text style={styles.startText}>Hi {patientName},</Text>
        <Text style={styles.startText}>{greeting}.</Text>
        <Text style={styles.startText}>Let’s have a short chat.</Text>
      </View>
      <Waveform active muted />
      <ActivityIndicator color={colors.taupe} />
    </View>
  )
}

function ListeningScreen({ userSpeaking }: { userSpeaking: boolean }) {
  return (
    <View style={styles.fillCenter}>
      <View style={styles.glow}>
        <AriaAvatar size={138} />
      </View>
      <View style={styles.micButton}>
        <Ionicons name="mic" size={34} color={colors.goldDark} />
      </View>
      <Text style={styles.listeningText}>Listening...</Text>
      <Waveform active={userSpeaking} />
    </View>
  )
}

function ResponseScreen({
  assistantText,
  disabled,
  onEnd,
  opacity,
  raiseEnd = false,
  userSpeaking,
}: {
  assistantText: string
  disabled: boolean
  onEnd: () => void
  opacity: Animated.Value
  raiseEnd?: boolean
  userSpeaking: boolean
}) {
  return (
    <View style={styles.responseScreen}>
      <Text style={styles.ariaTitle}>Aria</Text>
      <Animated.Text style={[styles.responseText, { opacity }]}>{assistantText}</Animated.Text>
      <View style={styles.responseAvatar}>
        <AriaAvatar size={112} />
      </View>
      <Waveform active={userSpeaking} cool />
      <Pressable disabled={disabled} style={[styles.endButton, raiseEnd && styles.endButtonRaised, disabled && styles.disabledButton]} onPress={onEnd}>
        <Text style={styles.endButtonText}>{disabled ? 'Ending...' : 'End Chat'}</Text>
        <Ionicons name="close" size={20} color={colors.text} />
      </Pressable>
    </View>
  )
}

function ErrorScreen({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <View style={styles.fillCenter}>
      <View style={styles.micButton}>
        <Ionicons name="cloud-offline-outline" size={34} color={colors.goldDark} />
      </View>
      <Text style={styles.listeningText}>连接出现问题</Text>
      <Text style={styles.wakeStatus}>{message || '请稍后重试'}</Text>
      <Pressable style={styles.endButton} onPress={onRetry}>
        <Text style={styles.endButtonText}>重试</Text>
      </Pressable>
    </View>
  )
}

function AriaAvatar({ size }: { size: number }) {
  const faceSize = size * 0.44
  return (
    <View style={[styles.avatar, { borderRadius: size / 2, height: size, width: size }]}>
      <View
        style={[
          styles.avatarHair,
          {
            borderRadius: size * 0.26,
            height: size * 0.58,
            top: size * 0.18,
            width: size * 0.56,
          },
        ]}
      />
      <View
        style={[
          styles.avatarFace,
          {
            borderRadius: faceSize / 2,
            height: faceSize,
            marginTop: size * 0.12,
            width: faceSize,
          },
        ]}
      >
        <View style={styles.avatarEyes}>
          <View style={styles.avatarEye} />
          <View style={styles.avatarEye} />
        </View>
        <View style={styles.avatarSmile} />
      </View>
    </View>
  )
}

function Waveform({
  active,
  cool = false,
  muted = false,
}: {
  active: boolean
  cool?: boolean
  muted?: boolean
}) {
  const pulse = useRef(new Animated.Value(0)).current

  useEffect(() => {
    if (!active) {
      pulse.stopAnimation()
      pulse.setValue(0)
      return
    }

    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, {
          duration: 360,
          easing: Easing.inOut(Easing.quad),
          toValue: 1,
          useNativeDriver: true,
        }),
        Animated.timing(pulse, {
          duration: 360,
          easing: Easing.inOut(Easing.quad),
          toValue: 0,
          useNativeDriver: true,
        }),
      ]),
    )
    loop.start()
    return () => loop.stop()
  }, [active, pulse])

  const baseBars = [6, 11, 18, 9, 25, 13, 32, 15, 23, 10, 17, 8, 5]
  const scale = pulse.interpolate({
    inputRange: [0, 1],
    outputRange: [0.72, 1.35],
  })

  return (
    <View style={styles.waveform}>
      {baseBars.map((height, index) => (
        <Animated.View
          key={`${height}-${index}`}
          style={[
            styles.waveBar,
            {
              backgroundColor: cool ? colors.blue : colors.goldDark,
              height,
              opacity: muted ? 0.38 : active ? 1 : 0.42,
              transform: [{ scaleY: active && index % 2 === 0 ? scale : 1 }],
            },
          ]}
        />
      ))}
    </View>
  )
}

const colors = {
  background: '#FFF9F1',
  card: '#FFFBF4',
  sand: '#F6F1E8',
  beige: '#EDE5D6',
  gold: '#E7CFA6',
  goldDark: '#C89755',
  taupe: '#BBAFA0',
  text: '#282828',
  secondary: '#686868',
  blue: '#5F87D8',
}

const styles = StyleSheet.create({
  safeArea: {
    backgroundColor: colors.background,
    flex: 1,
  },
  stage: {
    alignItems: 'center',
    backgroundColor: colors.background,
    flex: 1,
    justifyContent: 'center',
    padding: 18,
  },
  mirrorCard: {
    backgroundColor: colors.card,
    borderColor: '#F1E5D2',
    borderRadius: 8,
    borderWidth: 1,
    height: '100%',
    maxHeight: 760,
    maxWidth: 430,
    minHeight: 620,
    overflow: 'hidden',
    shadowColor: '#D8C6A8',
    shadowOffset: { height: 10, width: 0 },
    shadowOpacity: 0.22,
    shadowRadius: 22,
    width: '100%',
  },
  fillCenter: {
    alignItems: 'center',
    flex: 1,
    gap: 28,
    justifyContent: 'center',
    padding: 28,
  },
  cameraCorner: {
    borderRadius: 12,
    height: 118,
    overflow: 'hidden',
    position: 'absolute',
    right: 14,
    top: 14,
    width: 88,
    zIndex: 5,
  },
  idleCopy: {
    alignItems: 'center',
    gap: 6,
  },
  greeting: {
    color: colors.text,
    fontSize: 26,
    fontWeight: '500',
  },
  patientName: {
    color: colors.text,
    fontSize: 40,
    fontWeight: '500',
  },
  dateBlock: {
    alignItems: 'center',
    gap: 8,
  },
  date: {
    color: colors.text,
    fontSize: 22,
    fontWeight: '700',
  },
  time: {
    color: colors.text,
    fontSize: 32,
    fontWeight: '800',
  },
  startPrompt: {
    alignItems: 'center',
    gap: 4,
  },
  promptMain: {
    color: colors.text,
    fontSize: 24,
    fontWeight: '900',
  },
  promptSub: {
    color: colors.text,
    fontSize: 22,
    fontWeight: '800',
  },
  wakeStatus: {
    color: colors.taupe,
    fontSize: 14,
    fontWeight: '800',
    marginTop: 12,
    textAlign: 'center',
  },
  screeningPill: {
    alignItems: 'center',
    backgroundColor: '#FFF4E2',
    borderColor: '#F0DEC1',
    borderRadius: 22,
    borderWidth: 1,
    bottom: 34,
    flexDirection: 'row',
    gap: 8,
    paddingHorizontal: 18,
    paddingVertical: 12,
    position: 'absolute',
  },
  screeningPillText: {
    color: colors.goldDark,
    fontSize: 15,
    fontWeight: '900',
  },
  startCopy: {
    alignItems: 'center',
    gap: 2,
  },
  startText: {
    color: colors.text,
    fontSize: 24,
    fontWeight: '900',
    lineHeight: 32,
    textAlign: 'center',
  },
  glow: {
    alignItems: 'center',
    backgroundColor: '#FDF6EC',
    borderColor: '#F7FAFA',
    borderRadius: 102,
    borderWidth: 8,
    height: 196,
    justifyContent: 'center',
    shadowColor: '#C8D8F0',
    shadowOpacity: 0.65,
    shadowRadius: 24,
    width: 196,
  },
  micButton: {
    alignItems: 'center',
    backgroundColor: '#FFFDF9',
    borderRadius: 42,
    height: 78,
    justifyContent: 'center',
    shadowColor: '#D6C5AA',
    shadowOpacity: 0.26,
    shadowRadius: 14,
    width: 78,
  },
  listeningText: {
    color: colors.secondary,
    fontSize: 24,
    fontWeight: '900',
  },
  verifyingText: {
    color: colors.secondary,
    fontSize: 22,
    fontWeight: '800',
    textAlign: 'center',
  },
  responseScreen: {
    flex: 1,
    padding: 34,
    paddingTop: 78,
  },
  ariaTitle: {
    color: colors.text,
    fontSize: 24,
    fontWeight: '800',
    marginBottom: 28,
  },
  responseText: {
    color: colors.text,
    fontSize: 28,
    fontWeight: '800',
    lineHeight: 38,
    minHeight: 120,
  },
  responseAvatar: {
    alignItems: 'center',
    marginTop: 12,
  },
  endButton: {
    alignItems: 'center',
    alignSelf: 'center',
    backgroundColor: '#FFFFFF',
    borderColor: '#EEE0CC',
    borderRadius: 8,
    borderWidth: 1,
    bottom: 34,
    flexDirection: 'row',
    gap: 20,
    justifyContent: 'center',
    minWidth: 210,
    paddingHorizontal: 24,
    paddingVertical: 18,
    position: 'absolute',
    shadowColor: '#D8C6A8',
    shadowOpacity: 0.22,
    shadowRadius: 12,
  },
  endButtonText: {
    color: colors.text,
    fontSize: 18,
    fontWeight: '800',
  },
  disabledButton: {
    opacity: 0.55,
  },
  endButtonRaised: {
    bottom: 104,
  },
  recordButton: {
    alignItems: 'center',
    alignSelf: 'center',
    backgroundColor: '#4D9668',
    borderRadius: 8,
    bottom: 34,
    flexDirection: 'row',
    gap: 12,
    justifyContent: 'center',
    minWidth: 210,
    paddingHorizontal: 24,
    paddingVertical: 18,
    position: 'absolute',
    zIndex: 10,
  },
  recordButtonActive: {
    backgroundColor: '#C97068',
  },
  recordButtonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '800',
  },
  avatar: {
    alignItems: 'center',
    backgroundColor: '#FDF0E0',
    borderColor: '#FFFFFF',
    borderWidth: 4,
    justifyContent: 'center',
  },
  avatarHair: {
    backgroundColor: '#7A4D33',
    position: 'absolute',
  },
  avatarFace: {
    alignItems: 'center',
    backgroundColor: '#F2C4A5',
    justifyContent: 'center',
  },
  avatarEyes: {
    flexDirection: 'row',
    gap: 12,
  },
  avatarEye: {
    backgroundColor: colors.text,
    borderRadius: 4,
    height: 7,
    width: 7,
  },
  avatarSmile: {
    borderBottomColor: '#9A5F53',
    borderBottomWidth: 2,
    borderRadius: 12,
    height: 10,
    marginTop: 10,
    width: 24,
  },
  waveform: {
    alignItems: 'center',
    alignSelf: 'center',
    flexDirection: 'row',
    gap: 4,
    height: 48,
    justifyContent: 'center',
    marginTop: 18,
    width: 250,
  },
  waveBar: {
    borderRadius: 99,
    width: 3,
  },
})
