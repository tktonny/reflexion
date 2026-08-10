import { router, useLocalSearchParams } from 'expo-router'
import { createElement, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Platform, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native'

import { BootLoadingScreen, PairingScreen, type BootCheck, type PairingDetails, type PairingSetup } from './index'
import PairedSuccessScreen from './paired-success'
import { MirrorExperience, type MirrorHomeStatus, type MirrorVisualState } from '../src/components/mirror/MirrorExperience'
import { MirrorIcon } from '../src/components/mirror/MirrorIcon'
import { WifiSetupView } from '../src/components/mirror/WifiSetupView'
import { mirrorColors as c, mirrorFonts as f } from '../src/theme/mirrorTheme'
import {
  createDemoCheckInSnapshot,
  DEMO_CHECKIN_PROMPTS,
  applyDemoCheckInResponse,
  nextDemoCheckInTopic,
  startDemoCheckIn,
  stopDemoCheckIn,
  topicLabel,
  type DemoCheckInScenario,
  type DemoCheckInSnapshot,
  type DemoCheckInTopic,
} from '../src/demo/demoCheckinController'
import {
  DEMO_MESSAGES,
  DEMO_PATIENT,
  getDemoMirrorState,
  getDemoVoicePreference,
  resetDemoMirrorState,
  setDemoVoicePreference,
  updateDemoMirrorState,
  type DemoMirrorState,
} from '../src/demo/demoRepository'
import {
  chooseDemoVoice,
  demoVoiceId,
  getDemoSpeechRecognitionConstructor,
  getDemoVoiceOptions,
  getDemoWebAudioDiagnostics,
  isWebDemoRuntime,
  safeDemoAudioError,
  type DemoSpeechRecognition,
  type DemoVoiceOption,
} from '../src/demo/demoWebAudio'

const HOME_STATUSES: Array<{ value: MirrorHomeStatus; label: string }> = [
  { value: 'ready', label: 'Home ready' },
  { value: 'partial', label: 'Home partial' },
  { value: 'complete', label: 'Home complete' },
  { value: 'away', label: 'Away' },
  { value: 'paused', label: 'Paused' },
]

const CHECKIN_SCENARIOS: Array<{ value: DemoCheckInScenario; label: string; note: string }> = [
  { value: 'standard', label: 'Standard daily check-in', note: 'General → recall → planning → warm close' },
  { value: 'full', label: 'Full branch fixture', note: 'Routine, reminiscence and family message enabled' },
  { value: 'multi-topic', label: 'One response covers topics', note: 'One answer covers general, recall and planning' },
  { value: 'loved-one-question', label: 'Loved one asks a question', note: 'Answer a weather question, then return to the check-in' },
  { value: 'natural-follow-up', label: 'Natural follow-up', note: 'Use one gentle follow-up, then continue the core check-in' },
  { value: 'family-voice-reply', label: 'Family message → voice reply', note: 'Open MIR-11 and send an explicit voice reply' },
  { value: 'partial', label: 'Partial completion', note: 'Stop after general and recall; resume the remainder' },
  { value: 'clarification', label: 'One unclear response', note: 'One gentle reprompt, then continue' },
  { value: 'complete', label: 'Complete fixture', note: 'Free talk starts; daily state stays complete' },
]

type DemoCanonicalScreen = 'home' | 'boot' | 'wifi' | 'readiness' | 'pairing' | 'paired'
type DemoMicStatus = 'idle' | 'requesting' | 'ready' | 'capturing' | 'detected' | 'unavailable'
type DemoSpeakerStatus = 'idle' | 'speaking' | 'working' | 'unavailable'
type DemoRecognitionStatus = 'available' | 'listening' | 'captured' | 'fallback' | 'unavailable'

const DEMO_BOOT_CHECKS: BootCheck[] = [
  { key: 'wifi', label: 'Wi-Fi connected', ok: true },
  { key: 'internet', label: 'Internet available', ok: true },
  { key: 'backend', label: 'Reflexion service reachable', ok: true },
  { key: 'authenticated', label: 'Mirror authenticated', ok: true },
  { key: 'paired', label: 'Mirror paired with household', ok: true },
  { key: 'microphone', label: 'Microphone ready', ok: true },
  { key: 'timezone', label: 'Time and timezone ready', ok: true },
]

const DEMO_PAIRING: PairingDetails = {
  deviceId: 'demo-local-mirror',
  pairingId: 'demo-pairing-session',
  pairingCode: '482615',
  pairingToken: 'demo-local-pairing-token',
  qrPayload: JSON.stringify({ type: 'reflexion_device_pairing_v2', pairingToken: 'demo-local-pairing-token' }),
  expiresAt: new Date(Date.now() + 15 * 60 * 1000).toISOString(),
}

const DEMO_PAIRING_SETUP: PairingSetup = {
  current: 'READY_TO_PAIR',
  completed: ['WIFI_CONNECTED', 'INTERNET_AVAILABLE', 'SERVICE_REACHABLE', 'AUTHENTICATION_READY', 'READY_TO_PAIR'],
  failure: null,
}

export default function MirrorDemoScreen() {
  const params = useLocalSearchParams<{ screen?: string }>()
  const [demo, setDemo] = useState<DemoMirrorState | null>(null)
  const [visualState, setVisualState] = useState<MirrorVisualState>('ambient')
  const [activePrompt, setActivePrompt] = useState('')
  const [activeTopic, setActiveTopic] = useState<DemoCheckInTopic | null>(null)
  const [menuOpen, setMenuOpen] = useState(false)
  const [recognizedTranscript, setRecognizedTranscript] = useState('')
  const [voices, setVoices] = useState<DemoVoiceOption[]>([])
  const [selectedVoiceId, setSelectedVoiceId] = useState('')
  const webAudio = useMemo(() => getDemoWebAudioDiagnostics(), [])
  const [micStatus, setMicStatus] = useState<DemoMicStatus>('idle')
  const [micReason, setMicReason] = useState('')
  const [micLevel, setMicLevel] = useState(0)
  const [micTestActive, setMicTestActive] = useState(false)
  const [speakerStatus, setSpeakerStatus] = useState<DemoSpeakerStatus>('idle')
  const [speakerReason, setSpeakerReason] = useState('')
  const [recognitionStatus, setRecognitionStatus] = useState<DemoRecognitionStatus>(webAudio.speechRecognitionSupported ? 'available' : 'unavailable')
  const canonicalScreen = canonicalScreenFor(params.screen)
  const audioContextRef = useRef<AudioContext | null>(null)
  const micStreamRef = useRef<MediaStream | null>(null)
  const micAnalyserRef = useRef<AnalyserNode | null>(null)
  const micFrameRef = useRef<number | null>(null)
  const recognitionRef = useRef<DemoSpeechRecognition | null>(null)
  const speechGenerationRef = useRef(0)
  const handleResponseRef = useRef<(text: string) => void>(() => undefined)

  useEffect(() => { void getDemoMirrorState().then(setDemo) }, [])

  const time = useMemo(() => new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(new Date()), [])
  const date = useMemo(() => new Intl.DateTimeFormat(undefined, { weekday: 'long', month: 'long', day: 'numeric' }).format(new Date()), [])
  const speechLocale = useMemo(() => {
    const configured = String(process.env.EXPO_PUBLIC_DEMO_LANGUAGE || '').toLowerCase()
    const browserLanguage = typeof navigator !== 'undefined' ? navigator.language.toLowerCase() : ''
    return configured.startsWith('zh') || browserLanguage.startsWith('zh') ? 'zh-CN' : 'en-US'
  }, [])
  const message = DEMO_MESSAGES.find((item) => item.messageId === demo?.activeMessageId) || DEMO_MESSAGES[0]

  const refreshVoices = useCallback(() => {
    if (!isWebDemoRuntime()) return
    const options = getDemoVoiceOptions()
    setVoices(options)
    setSelectedVoiceId((current) => {
      if (current && options.some((voice) => voice.id === current)) return current
      const preferred = chooseDemoVoice(speechLocale)
      return preferred ? demoVoiceId(preferred) : ''
    })
  }, [speechLocale])

  useEffect(() => {
    if (!isWebDemoRuntime()) return
    const synthesis = window.speechSynthesis
    refreshVoices()
    void getDemoVoicePreference().then((preference) => { if (preference) setSelectedVoiceId(preference) })
    const onVoicesChanged = () => refreshVoices()
    synthesis.addEventListener?.('voiceschanged', onVoicesChanged)
    const timer = window.setTimeout(refreshVoices, 250)
    return () => {
      synthesis.removeEventListener?.('voiceschanged', onVoicesChanged)
      window.clearTimeout(timer)
    }
  }, [refreshVoices])

  const stopRecognition = useCallback(() => {
    const recognition = recognitionRef.current
    recognitionRef.current = null
    if (!recognition) return
    recognition.onend = null
    recognition.onresult = null
    recognition.onerror = null
    try { recognition.stop() } catch { /* best effort */ }
  }, [])

  const stopMicCapture = useCallback(() => {
    if (micFrameRef.current !== null && typeof window !== 'undefined') {
      window.cancelAnimationFrame(micFrameRef.current)
      micFrameRef.current = null
    }
    micAnalyserRef.current?.disconnect()
    micAnalyserRef.current = null
    micStreamRef.current?.getTracks().forEach((track) => track.stop())
    micStreamRef.current = null
    setMicLevel(0)
    setMicStatus((current) => current === 'unavailable' ? current : 'ready')
  }, [])

  const ensureWebAudioContext = useCallback(async () => {
    if (!isWebDemoRuntime()) return null
    if (audioContextRef.current) {
      await audioContextRef.current.resume()
      return audioContextRef.current
    }
    const browserWindow = window as Window & { webkitAudioContext?: typeof AudioContext }
    const AudioContextConstructor = window.AudioContext ?? browserWindow.webkitAudioContext
    if (!AudioContextConstructor) throw new Error('Web Audio is not supported by this browser.')
    const context = new AudioContextConstructor()
    await context.resume()
    audioContextRef.current = context
    return context
  }, [])

  const startMicCapture = useCallback(async () => {
    if (!isWebDemoRuntime() || micStreamRef.current) return
    if (!webAudio.secureContext) {
      setMicStatus('unavailable')
      setMicReason('Microphone access requires HTTPS or localhost.')
      return
    }
    if (!webAudio.microphoneSupported) {
      setMicStatus('unavailable')
      setMicReason('This browser does not expose navigator.mediaDevices.getUserMedia.')
      return
    }
    setMicStatus('requesting')
    setMicReason('')
    try {
      const context = await ensureWebAudioContext()
      if (!context) return
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const analyser = context.createAnalyser()
      analyser.fftSize = 512
      context.createMediaStreamSource(stream).connect(analyser)
      micStreamRef.current = stream
      micAnalyserRef.current = analyser
      setMicStatus('capturing')
      const samples = new Uint8Array(analyser.fftSize)
      const measure = () => {
        if (micAnalyserRef.current !== analyser) return
        analyser.getByteTimeDomainData(samples)
        let sum = 0
        for (const sample of samples) {
          const normalized = (sample - 128) / 128
          sum += normalized * normalized
        }
        const level = Math.min(1, Math.sqrt(sum / samples.length) * 3.2)
        setMicLevel(level)
        if (level > 0.035) setMicStatus('detected')
        micFrameRef.current = window.requestAnimationFrame(measure)
      }
      micFrameRef.current = window.requestAnimationFrame(measure)
    } catch (error) {
      setMicStatus('unavailable')
      setMicReason(safeDemoAudioError(error, 'Permission was denied or the microphone could not start.'))
    }
  }, [ensureWebAudioContext, webAudio.microphoneSupported, webAudio.secureContext])

  const cancelSpeech = useCallback(() => {
    speechGenerationRef.current += 1
    if (isWebDemoRuntime()) window.speechSynthesis?.cancel()
    setSpeakerStatus((current) => current === 'speaking' ? 'idle' : current)
  }, [])

  const speakWebText = useCallback((text: string, onFinished?: () => void, onStarted?: () => void) => {
    if (!isWebDemoRuntime()) { onFinished?.(); return false }
    if (!webAudio.speechSynthesisSupported) {
      setSpeakerStatus('unavailable')
      setSpeakerReason('This browser does not support speech synthesis.')
      onFinished?.()
      return false
    }
    const synthesis = window.speechSynthesis
    const generation = ++speechGenerationRef.current
    synthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = speechLocale
    utterance.rate = 0.85
    utterance.volume = 1
    const voice = chooseDemoVoice(speechLocale, selectedVoiceId)
    if (voice) utterance.voice = voice
    setSpeakerStatus('working')
    setSpeakerReason('')
    let finished = false
    utterance.onstart = () => {
      if (finished || generation !== speechGenerationRef.current) return
      setSpeakerStatus('speaking')
      onStarted?.()
    }
    utterance.onend = () => {
      if (finished || generation !== speechGenerationRef.current) return
      finished = true
      setSpeakerStatus('working')
      onFinished?.()
    }
    utterance.onerror = (event) => {
      if (finished || generation !== speechGenerationRef.current) return
      finished = true
      setSpeakerStatus('unavailable')
      setSpeakerReason(event.error || 'The browser speech engine could not speak.')
      onFinished?.()
    }
    try {
      synthesis.speak(utterance)
      return true
    } catch (error) {
      finished = true
      setSpeakerStatus('unavailable')
      setSpeakerReason(safeDemoAudioError(error, 'The browser speech engine could not start.'))
      onFinished?.()
      return false
    }
  }, [selectedVoiceId, speechLocale, webAudio.speechSynthesisSupported])

  const testSpeaker = useCallback(() => {
    void ensureWebAudioContext().catch(() => undefined)
    speakWebText('Hello Margaret. This is the local Aria voice preview.')
  }, [ensureWebAudioContext, speakWebText])

  const startRecognition = useCallback(() => {
    if (!isWebDemoRuntime()) return
    const Recognition = getDemoSpeechRecognitionConstructor()
    if (!Recognition) { setRecognitionStatus('unavailable'); return }
    stopRecognition()
    const recognition = new Recognition()
    recognition.continuous = false
    recognition.interimResults = true
    recognition.lang = speechLocale
    recognition.onresult = (event) => {
      let finalTranscript = ''
      let interimTranscript = ''
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index]
        const transcript = result[0]?.transcript || ''
        if (result.isFinal) finalTranscript += transcript
        else interimTranscript += transcript
      }
      if (finalTranscript.trim()) {
        setRecognizedTranscript(finalTranscript.trim())
        setRecognitionStatus('captured')
        stopRecognition()
        handleResponseRef.current(finalTranscript.trim())
      } else if (interimTranscript.trim()) setRecognizedTranscript(interimTranscript.trim())
    }
    recognition.onerror = (event) => {
      if (recognitionRef.current !== recognition) return
      setRecognitionStatus('fallback')
      setMicReason(event.error || 'Speech recognition is unavailable; use Continue demo.')
      stopRecognition()
    }
    recognition.onend = () => {
      if (recognitionRef.current === recognition) {
        recognitionRef.current = null
        setRecognitionStatus('fallback')
        setMicReason('Speech recognition ended; use Continue demo.')
      }
    }
    recognitionRef.current = recognition
    try {
      recognition.start()
      setRecognitionStatus('listening')
    } catch (error) {
      recognitionRef.current = null
      setRecognitionStatus('fallback')
      setMicReason(safeDemoAudioError(error, 'Speech recognition could not start; use Continue demo.'))
    }
  }, [speechLocale, stopRecognition])

  useEffect(() => {
    if (!isWebDemoRuntime()) return
    const captureWanted = visualState === 'listening' || micTestActive
    if (captureWanted) void startMicCapture()
    else stopMicCapture()
    if (visualState === 'listening') startRecognition()
    else stopRecognition()
  }, [micTestActive, startMicCapture, startRecognition, stopMicCapture, stopRecognition, visualState])

  useEffect(() => () => {
    cancelSpeech()
    stopRecognition()
    stopMicCapture()
    try { audioContextRef.current?.close() } catch { /* best effort */ }
    audioContextRef.current = null
  }, [cancelSpeech, stopMicCapture, stopRecognition])

  const releaseDemoAudio = () => {
    cancelSpeech()
    stopRecognition()
    stopMicCapture()
    setMicTestActive(false)
  }

  if (!demo) return <View style={styles.loading}><Text style={styles.loadingText}>Opening local demo…</Text></View>

  const persistCheckIn = (checkIn: DemoCheckInSnapshot, homeStatus?: MirrorHomeStatus) => {
    setDemo((current) => current ? { ...current, checkIn, ...(homeStatus ? { homeStatus } : {}) } : current)
    void updateDemoMirrorState({ checkIn, ...(homeStatus ? { homeStatus } : {}) })
  }

  const speakPrompt = (text: string, onFinished?: () => void) => {
    setActivePrompt(text)
    setVisualState('thinking')
    void ensureWebAudioContext().catch(() => undefined)
    speakWebText(
      text,
      onFinished || (() => setVisualState('listening')),
      () => setVisualState('speaking'),
    )
  }

  const beginConversation = () => {
    cancelSpeech()
    setRecognizedTranscript('')
    setMenuOpen(false)
    if (canonicalScreen !== 'home') router.replace('/demo')
    const result = startDemoCheckIn(demo.checkIn)
    persistCheckIn(result.snapshot)
    setActiveTopic(result.topic || null)
    speakPrompt(result.assistantText, () => setVisualState('listening'))
  }

  const scriptedResponse = (snapshot: DemoCheckInSnapshot): string => {
    if (snapshot.scenario === 'clarification' && snapshot.clarificationFor !== null) return 'I am feeling good and ready for the day.'
    if (snapshot.scenario === 'clarification' && snapshot.responseCount === 0) return 'hmm'
    if (snapshot.scenario === 'multi-topic' && snapshot.responseCount === 0) return 'I am feeling good. Yesterday I visited Mei, and today I plan to read.'
    if (snapshot.scenario === 'loved-one-question' && snapshot.responseCount === 1) return 'What is the weather like?'
    if (snapshot.scenario === 'partial' && snapshot.responseCount === 0) return 'I am feeling good today.'
    if (snapshot.scenario === 'partial' && snapshot.responseCount === 1) return 'Yesterday I slept well.'
    switch (nextDemoCheckInTopic(snapshot)) {
      case 'general': return 'I am feeling good and ready for the day.'
      case 'recentRecall': return 'Yesterday I had a quiet day and slept comfortably.'
      case 'planning': return 'I plan to have breakfast and read this morning.'
      case 'routine': return 'Yes, I would like to take them now.'
      case 'reminiscence': return 'I remember a lovely day in the garden with my family.'
      case 'familyMessage': return 'Yes, I would love to hear Mei’s message.'
      default: return 'Thank you, Aria.'
    }
  }

  const handleResponse = (text: string) => {
    if (visualState !== 'listening' && visualState !== 'heard') return
    stopRecognition()
    stopMicCapture()
    setRecognizedTranscript(text)
    const result = applyDemoCheckInResponse(demo.checkIn, text)
    const isComplete = result.kind === 'complete'
    persistCheckIn(result.snapshot, isComplete ? 'complete' : undefined)
    setActiveTopic(result.topic || null)
    if (isComplete) {
      setActivePrompt(result.assistantText)
      setVisualState('closing')
      void ensureWebAudioContext().catch(() => undefined)
      speakWebText(result.assistantText)
      return
    }
    speakPrompt(result.assistantText, () => setVisualState('listening'))
  }

  handleResponseRef.current = handleResponse

  const stopConversation = () => {
    const next = stopDemoCheckIn(demo.checkIn)
    persistCheckIn(next, next.state === 'Partially complete' ? 'partial' : 'complete')
    releaseDemoAudio()
    setActivePrompt('That’s alright. We can continue another time.')
    setActiveTopic(null)
    setVisualState('closing')
    void ensureWebAudioContext().catch(() => undefined)
    speakWebText('That’s alright. We can continue another time.')
  }

  const repeatConversation = () => {
    if (!activePrompt) return
    cancelSpeech()
    speakPrompt(activePrompt, () => setVisualState('listening'))
  }

  const returnHome = () => {
    releaseDemoAudio()
    setVisualState('ambient')
    if (canonicalScreen !== 'home') router.replace('/demo')
  }

  const openMessage = (messageId = message.messageId) => {
    releaseDemoAudio()
    setVisualState('ambient')
    setMenuOpen(false)
    void updateDemoMirrorState({ activeMessageId: messageId })
    router.push({ pathname: '/family-message', params: { demo: '1', messageId } })
  }

  const openCanonicalScreen = (screen: DemoCanonicalScreen) => {
    releaseDemoAudio()
    setActivePrompt('')
    setVisualState('ambient')
    setMenuOpen(false)
    router.replace(screen === 'home' ? '/demo' : `/demo?screen=${screen}`)
  }

  const openDemoRoute = (path: string) => {
    releaseDemoAudio()
    setVisualState('ambient')
    setMenuOpen(false)
    router.push(path)
  }

  const showListeningState = () => {
    releaseDemoAudio()
    setActivePrompt(activePrompt || DEMO_CHECKIN_PROMPTS.general)
    setMenuOpen(false)
    setVisualState('listening')
    if (canonicalScreen !== 'home') router.replace('/demo')
  }

  const showClosingState = () => {
    releaseDemoAudio()
    const complete = demo.checkIn.state === 'Complete'
    setActivePrompt(complete ? `Thank you, ${DEMO_PATIENT.name}.` : 'That’s alright. We can continue another time.')
    setVisualState('closing')
    if (canonicalScreen !== 'home') router.replace('/demo')
  }

  const setHomeStatus = (homeStatus: MirrorHomeStatus) => {
    setDemo((current) => current ? { ...current, homeStatus } : current)
    void updateDemoMirrorState({ homeStatus })
    releaseDemoAudio()
    setVisualState('ambient')
  }

  const setScenario = (scenario: DemoCheckInScenario) => {
    const checkIn = createDemoCheckInSnapshot(scenario)
    persistCheckIn(checkIn, checkIn.state === 'Complete' ? 'complete' : 'ready')
    setActivePrompt('')
    setActiveTopic(null)
    releaseDemoAudio()
    setVisualState('ambient')
  }

  const setVoice = (voiceId: string) => {
    setSelectedVoiceId(voiceId)
    if (voiceId) void setDemoVoicePreference(voiceId)
  }

  const mainContent = canonicalScreen === 'boot' ? (
    <BootLoadingScreen checks={DEMO_BOOT_CHECKS} />
  ) : canonicalScreen === 'wifi' ? (
    <WifiSetupView onRetry={() => router.replace('/demo')} />
  ) : canonicalScreen === 'readiness' ? (
    <PairingScreen error="" onBackToReadiness={() => undefined} onRefresh={() => undefined} onRetry={() => undefined} onShowPairing={() => router.replace('/demo?screen=pairing')} pairing={DEMO_PAIRING} setup={DEMO_PAIRING_SETUP} showPairing={false} />
  ) : canonicalScreen === 'pairing' ? (
    <PairingScreen error="" onBackToReadiness={() => router.replace('/demo?screen=readiness')} onRefresh={() => undefined} onRetry={() => undefined} onShowPairing={() => undefined} pairing={DEMO_PAIRING} setup={DEMO_PAIRING_SETUP} showPairing />
  ) : canonicalScreen === 'paired' ? (
    <PairedSuccessScreen demoName={DEMO_PATIENT.name} />
  ) : (
    <MirrorExperience
      assistantText={visualState === 'thinking' || visualState === 'speaking' || visualState === 'listening' ? activePrompt : visualState === 'closing' ? activePrompt : undefined}
      date={date}
      greeting={DEMO_PATIENT.greeting}
      homeMessage={{ kind: message.kind, preview: message.preview, sender: message.senderName }}
      homeStatus={demo.homeStatus}
      homeWidgets={[{ icon: 'medical-outline', label: 'Medication reminder', value: demo.routineResponse === 'complete' ? 'Morning tablets · done' : 'Morning tablets' }]}
      onBegin={beginConversation}
      onEnd={returnHome}
      onOpenConsent={() => router.push('/consent?demo=1')}
      onOpenMessage={() => openMessage()}
      onOpenResearch={() => router.push('/research?demo=1')}
      onOpenRoutine={() => router.push('/routine?demo=1')}
      onOpenStatus={() => router.push('/status?demo=1')}
      onRepeat={repeatConversation}
      onStop={stopConversation}
      patientName={DEMO_PATIENT.name}
      microphoneLevel={isWebDemoRuntime() ? micLevel : undefined}
      state={visualState}
      time={time}
    />
  )

  const currentTopic = nextDemoCheckInTopic(demo.checkIn)
  const selectedVoice = voices.find((voice) => voice.id === selectedVoiceId)
  const voicePicker = Platform.OS === 'web' ? createElement('select', {
    'aria-label': 'Browser voice',
    onChange: (event: { target?: { value?: string } }) => setVoice(event.target?.value || ''),
    style: { backgroundColor: '#fffaf4', borderColor: c.sage, borderRadius: 8, borderWidth: 1, color: c.text, fontSize: 12, minHeight: 34, padding: 6, width: '100%' },
    value: selectedVoiceId,
  }, [createElement('option', { key: 'automatic', value: '' }, 'Automatic browser voice'), ...voices.map((voice) => createElement('option', { key: voice.id, value: voice.id }, `${voice.name} · ${voice.lang}${voice.default ? ' · default' : ''}`))]) : null

  return (
    <View style={styles.shell}>
      <View style={styles.contentLayer}>{mainContent}</View>
      <View pointerEvents="box-none" style={styles.overlay}>
        <View style={styles.banner}>
          <MirrorIcon color={c.goldDeep} name="flask-outline" size={17} />
          <Text style={styles.bannerText}>DEMO MODE · local only · no caregiver sync</Text>
          <Pressable accessibilityLabel="Open demo test menu" accessibilityRole="button" onPress={() => setMenuOpen((current) => !current)} style={styles.menuButton}>
            <MirrorIcon color={c.sageDeep} name={menuOpen ? 'close' : 'menu'} size={19} />
            <Text style={styles.menuText}>{menuOpen ? 'Close' : 'Test menu'}</Text>
          </Pressable>
        </View>

        {canonicalScreen === 'home' && visualState !== 'ambient' && visualState !== 'closing' ? (
          <View style={styles.advanceCard}>
            <Text style={styles.advanceTitle}>DAILY CHECK-IN · {activeTopic ? topicLabel(activeTopic) : demo.checkIn.state}</Text>
            <Text style={styles.advanceCopy}>
              {visualState === 'thinking'
                ? 'Aria is thinking about what you said.'
                : visualState === 'speaking'
                ? speakerStatus === 'unavailable' ? 'Speech unavailable; use Continue demo.' : 'Aria is speaking locally.'
                : recognizedTranscript
                  ? `Heard: “${recognizedTranscript}”`
                  : !webAudio.speechRecognitionSupported || recognitionStatus === 'fallback' || recognitionStatus === 'unavailable'
                    ? `Demo response: “${scriptedResponse(demo.checkIn)}”`
                    : 'Speak naturally; the browser will continue when it hears you.'}
            </Text>
            {visualState !== 'thinking' && (visualState === 'speaking' || !webAudio.speechRecognitionSupported || recognitionStatus === 'fallback' || recognitionStatus === 'unavailable') ? (
              <Pressable accessibilityRole="button" onPress={visualState === 'speaking' ? () => { cancelSpeech(); setVisualState('listening') } : () => handleResponse(scriptedResponse(demo.checkIn))} style={styles.advanceButton}>
                <Text style={styles.advanceButtonText}>{visualState === 'speaking' ? 'Start local listening' : 'Continue check-in'}</Text>
              </Pressable>
            ) : null}
          </View>
        ) : null}

        {visualState === 'closing' && canonicalScreen === 'home' ? (
          <View style={styles.returnHomeCard}>
            <Text style={styles.returnHomeCopy}>{demo.checkIn.state === 'Partially complete' ? 'Your progress is saved.' : 'The daily check-in is complete.'}</Text>
            <Pressable accessibilityRole="button" onPress={returnHome} style={styles.advanceButton}><Text style={styles.advanceButtonText}>Return home</Text></Pressable>
          </View>
        ) : null}

        {menuOpen ? (
          <ScrollView style={styles.menuPanel} contentContainerStyle={styles.menuPanelContent}>
            <Text style={styles.menuHeading}>Mirror test menu</Text>
            <Text style={styles.menuNote}>Development-only local fixtures; no caregiver sync or production requests.</Text>
            {isWebDemoRuntime() ? (
              <View style={styles.audioPanel}>
                <Text style={styles.audioHeading}>AUDIO TEST · WEB DEMO ONLY</Text>
                <Text style={styles.audioStatus}>Secure context: {webAudio.secureContext ? 'YES' : 'NO — open the HTTPS URL or localhost'}</Text>
                <Text style={styles.audioStatus}>Microphone API: {webAudio.microphoneSupported ? 'available' : 'unavailable'}</Text>
                <Text style={styles.audioStatus}>Speech synthesis: {webAudio.speechSynthesisSupported ? 'available' : 'unavailable'}</Text>
                <Text style={styles.audioStatus}>Speech recognition: {webAudio.speechRecognitionSupported ? recognitionStatus : 'unavailable — use Continue check-in'}</Text>
                <Text style={styles.audioStatus}>Mic status: {micStatus}{micReason ? ` · ${micReason}` : ''}</Text>
                <Text style={styles.audioStatus}>Live level: {micLevel.toFixed(3)}</Text>
                <Text style={styles.audioLabel}>Browser voice</Text>
                {voicePicker}
                <Text style={styles.audioStatus}>{selectedVoice ? `Selected: ${selectedVoice.name} · ${selectedVoice.lang}` : voices.length ? 'Automatic voice selection' : 'Loading browser voices…'}</Text>
                <View style={styles.audioActions}>
                  <Pressable accessibilityRole="button" onPress={() => setMicTestActive((current) => !current)} style={styles.audioButton}><Text style={styles.audioButtonText}>{micTestActive ? 'Stop microphone test' : 'Test microphone'}</Text></Pressable>
                  <Pressable accessibilityRole="button" onPress={testSpeaker} style={styles.audioButton}><Text style={styles.audioButtonText}>Preview voice</Text></Pressable>
                </View>
                <Text style={styles.audioStatus}>Speaker status: {speakerStatus}{speakerReason ? ` · ${speakerReason}` : ''}</Text>
                {recognizedTranscript ? <Text style={styles.audioTranscript}>Development transcript: {recognizedTranscript}</Text> : null}
                <Text style={styles.audioNote}>Speech rate is configured for a calm demo pace. Browser voice preference is stored locally.</Text>
              </View>
            ) : null}

            <Text style={styles.menuSectionHeading}>FINAL MIR SCREENS</Text>
            <DemoMenuButton label="MIR-01 Boot / loading" onPress={() => openCanonicalScreen('boot')} />
            <DemoMenuButton label="MIR-02 Wi-Fi setup" onPress={() => openCanonicalScreen('wifi')} />
            <DemoMenuButton label="MIR-03 Readiness & pairing status" onPress={() => openCanonicalScreen('readiness')} />
            <DemoMenuButton label="MIR-04 Pairing QR & code" onPress={() => openCanonicalScreen('pairing')} />
            <DemoMenuButton label="MIR-05 Paired successfully" onPress={() => openCanonicalScreen('paired')} />
            <DemoMenuButton label="MIR-06 Idle Home" onPress={() => openCanonicalScreen('home')} />
            <DemoMenuButton label="MIR-07 Aria speaking" onPress={beginConversation} />
            <DemoMenuButton label="MIR-08 Aria listening" onPress={showListeningState} />
            <DemoMenuButton label="MIR-09 Session close" onPress={showClosingState} />
            <DemoMenuButton label="MIR-10 Routine reminder" onPress={() => openDemoRoute('/routine?demo=1')} />
            <DemoMenuButton label="MIR-11 Family message + voice reply" onPress={() => openMessage('demo-text-message')} />
            <DemoMenuButton label="MIR-12 Consent & control" onPress={() => openDemoRoute('/consent?demo=1')} />
            <DemoMenuButton label="MIR-13 Device status & help" onPress={() => openDemoRoute('/status?demo=1')} />
            <DemoMenuButton label="MIR-14 Research participation" onPress={() => openDemoRoute('/research?demo=1')} />

            <Text style={styles.menuSectionHeading}>CHECK-IN CONTROLLER INSPECTOR</Text>
            <Text style={styles.inspectorState}>Check-in state: {demo.checkIn.state}</Text>
            <Text style={styles.inspectorState}>Current topic: {currentTopic ? topicLabel(currentTopic) : 'None'}</Text>
            <Text style={styles.inspectorState}>Last user turn: {demo.checkIn.lastUserTurn?.text || 'None yet'}</Text>
            <Text style={styles.inspectorState}>Topics detected: {demo.checkIn.lastUserTurn?.topicsDetected.map(topicLabel).join(', ') || 'None'}</Text>
            <Text style={styles.inspectorState}>Direct question: {demo.checkIn.lastUserTurn?.directQuestion ? 'yes' : 'no'}</Text>
            <Text style={styles.inspectorState}>Follow-up used: {demo.checkIn.lastUserTurn?.followUpUsed ? 'yes' : 'no'}</Text>
            <Text style={styles.inspectorState}>Next core topic: {demo.checkIn.lastUserTurn?.nextCoreTopic ? topicLabel(demo.checkIn.lastUserTurn.nextCoreTopic) : 'None'}</Text>
            {(Object.keys(demo.checkIn.topics) as Array<keyof DemoCheckInSnapshot['topics']>).map((topic) => (
              <View key={topic} style={styles.inspectorRow}><Text style={styles.inspectorTopic}>{topicLabel(topic)}</Text><Text style={styles.inspectorValue}>{demo.checkIn.topics[topic]}</Text></View>
            ))}
            <View style={styles.scenarioGrid}>
              {CHECKIN_SCENARIOS.map((scenario) => <Pressable key={scenario.value} onPress={() => setScenario(scenario.value)} style={[styles.scenarioButton, demo.checkIn.scenario === scenario.value && styles.scenarioButtonActive]}><Text style={[styles.scenarioButtonText, demo.checkIn.scenario === scenario.value && styles.scenarioButtonTextActive]}>{scenario.label}</Text><Text style={[styles.scenarioButtonNote, demo.checkIn.scenario === scenario.value && styles.scenarioButtonTextActive]}>{scenario.note}</Text></Pressable>)}
            </View>

            <Text style={styles.menuSectionHeading}>HOME FIXTURES</Text>
            <View style={styles.statusGrid}>{HOME_STATUSES.map((item) => <Pressable key={item.value} onPress={() => setHomeStatus(item.value)} style={[styles.statusButton, demo.homeStatus === item.value && styles.statusButtonActive]}><Text style={[styles.statusButtonText, demo.homeStatus === item.value && styles.statusButtonTextActive]}>{item.label}</Text></Pressable>)}</View>
            <DemoMenuButton label="Start Demo Conversation" onPress={beginConversation} />
            <DemoMenuButton label="Text message" onPress={() => openMessage('demo-text-message')} />
            <DemoMenuButton label="Photo message" onPress={() => openMessage('demo-photo-message')} />
            <DemoMenuButton label="Voice message" onPress={() => openMessage('demo-voice-message')} />
            <Pressable accessibilityRole="button" onPress={() => { void resetDemoMirrorState(); router.replace('/') }} style={styles.exitButton}><Text style={styles.exitButtonText}>Exit Demo Mode</Text></Pressable>
          </ScrollView>
        ) : null}
      </View>
    </View>
  )
}

function DemoMenuButton({ label, onPress }: { label: string; onPress: () => void }) {
  return <Pressable accessibilityRole="button" onPress={onPress} style={styles.menuRow}><Text style={styles.menuRowText}>{label}</Text><MirrorIcon color={c.sageDeep} name="chevron-forward" size={18} /></Pressable>
}

function canonicalScreenFor(value: unknown): DemoCanonicalScreen {
  return ['boot', 'wifi', 'readiness', 'pairing', 'paired'].includes(String(value)) ? value as DemoCanonicalScreen : 'home'
}

const styles = StyleSheet.create({
  shell: { backgroundColor: c.cream, flex: 1 },
  contentLayer: { flex: 1 },
  loading: { alignItems: 'center', backgroundColor: c.cream, flex: 1, justifyContent: 'center' },
  loadingText: { color: c.sageDeep, fontFamily: f.body, fontSize: 18 },
  overlay: { left: 0, position: 'absolute', right: 0, top: 7 },
  banner: { alignItems: 'center', alignSelf: 'center', backgroundColor: 'rgba(255,255,255,0.94)', borderColor: c.gold, borderRadius: 20, borderWidth: 1, flexDirection: 'row', gap: 7, maxWidth: '94%', paddingHorizontal: 12, paddingVertical: 7 },
  bannerText: { color: c.goldDeep, flexShrink: 1, fontFamily: f.bodyMedium, fontSize: 11, letterSpacing: 0.4 },
  menuButton: { alignItems: 'center', borderLeftColor: c.lineWarm, borderLeftWidth: 1, flexDirection: 'row', gap: 4, marginLeft: 3, paddingLeft: 9 },
  menuText: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 11 },
  advanceCard: { alignSelf: 'center', backgroundColor: 'rgba(255,255,255,0.95)', borderColor: c.sage, borderRadius: 17, borderWidth: 1, marginTop: 10, maxWidth: 520, padding: 10, width: '84%' },
  advanceTitle: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 11, letterSpacing: 0.5, textAlign: 'center' },
  advanceCopy: { color: c.textSecondary, fontFamily: f.body, fontSize: 13, marginTop: 3, textAlign: 'center' },
  advanceButton: { alignItems: 'center', backgroundColor: c.sageDeep, borderRadius: 18, marginTop: 7, minHeight: 34, justifyContent: 'center', paddingHorizontal: 15 },
  advanceButtonText: { color: c.white, fontFamily: f.bodyMedium, fontSize: 12 },
  returnHomeCard: { alignSelf: 'center', backgroundColor: 'rgba(255,255,255,0.95)', borderColor: c.sage, borderRadius: 17, borderWidth: 1, marginTop: 10, maxWidth: 370, padding: 10, width: '78%' },
  returnHomeCopy: { color: c.textSecondary, fontFamily: f.body, fontSize: 13, textAlign: 'center' },
  menuPanel: { alignSelf: 'center', backgroundColor: 'rgba(251,248,243,0.98)', borderColor: c.lineWarm, borderRadius: 21, borderWidth: 1, marginTop: 10, maxHeight: '84%', maxWidth: 680, width: '92%' },
  menuPanelContent: { padding: 15 },
  menuHeading: { color: c.text, fontFamily: f.display, fontSize: 24, textAlign: 'center' },
  menuNote: { color: c.textSecondary, fontFamily: f.body, fontSize: 12, marginBottom: 10, marginTop: 3, textAlign: 'center' },
  menuSectionHeading: { color: c.goldDeep, fontFamily: f.bodyMedium, fontSize: 11, letterSpacing: 0.7, marginBottom: 3, marginTop: 12, textAlign: 'center' },
  audioPanel: { backgroundColor: c.cream, borderColor: c.sage, borderRadius: 16, borderWidth: 1, marginBottom: 11, padding: 12 },
  audioHeading: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 12, letterSpacing: 0.6, textAlign: 'center' },
  audioLabel: { color: c.text, fontFamily: f.bodyMedium, fontSize: 12, marginTop: 9 },
  audioStatus: { color: c.textSecondary, fontFamily: f.body, fontSize: 12, marginTop: 4 },
  audioActions: { flexDirection: 'row', flexWrap: 'wrap', gap: 7, marginTop: 9 },
  audioButton: { backgroundColor: c.sageDeep, borderRadius: 16, minHeight: 34, justifyContent: 'center', paddingHorizontal: 12 },
  audioButtonText: { color: c.white, fontFamily: f.bodyMedium, fontSize: 12 },
  audioTranscript: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 12, marginTop: 7 },
  audioNote: { color: c.textSecondary, fontFamily: f.body, fontSize: 11, marginTop: 8 },
  inspectorState: { color: c.text, fontFamily: f.bodyMedium, fontSize: 13, marginTop: 4 },
  inspectorRow: { alignItems: 'center', borderBottomColor: c.lineWarm, borderBottomWidth: StyleSheet.hairlineWidth, flexDirection: 'row', justifyContent: 'space-between', minHeight: 31, paddingHorizontal: 5 },
  inspectorTopic: { color: c.text, fontFamily: f.body, fontSize: 12 },
  inspectorValue: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 12 },
  scenarioGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 7, marginTop: 8 },
  scenarioButton: { borderColor: c.sage, borderRadius: 13, borderWidth: 1, flexGrow: 1, minWidth: 190, padding: 9 },
  scenarioButtonActive: { backgroundColor: c.sageDeep },
  scenarioButtonText: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 12 },
  scenarioButtonTextActive: { color: c.white },
  scenarioButtonNote: { color: c.textSecondary, fontFamily: f.body, fontSize: 10, marginTop: 3 },
  statusGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 7, justifyContent: 'center', marginBottom: 4 },
  statusButton: { borderColor: c.sage, borderRadius: 15, borderWidth: 1, paddingHorizontal: 11, paddingVertical: 8 },
  statusButtonActive: { backgroundColor: c.sageDeep },
  statusButtonText: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 12 },
  statusButtonTextActive: { color: c.white },
  menuRow: { alignItems: 'center', borderBottomColor: c.lineWarm, borderBottomWidth: StyleSheet.hairlineWidth, flexDirection: 'row', justifyContent: 'space-between', minHeight: 42, paddingHorizontal: 5 },
  menuRowText: { color: c.text, fontFamily: f.bodyMedium, fontSize: 15 },
  exitButton: { alignItems: 'center', borderColor: c.coral, borderRadius: 20, borderWidth: 1, marginTop: 13, minHeight: 42, justifyContent: 'center' },
  exitButtonText: { color: c.coral, fontFamily: f.bodyMedium, fontSize: 14 },
})
