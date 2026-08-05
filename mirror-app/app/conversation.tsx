import AsyncStorage from '@react-native-async-storage/async-storage'
import { Ionicons } from '@expo/vector-icons'
import { router, useLocalSearchParams } from 'expo-router'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Platform, Pressable, StyleSheet, Text, View } from 'react-native'

import { dataOrThrow, type DeviceConfiguration } from '../src/api/devicePairing'
import { fetchFamilyMessages, markFamilyMessageOpened, type FamilyMessage } from '../src/api/familyMessages'
import { loadDailyConversationPlan } from '../src/api/dailyConversationContext'
import { sendDeviceHeartbeat, subscribeDeviceHeartbeat } from '../src/api/deviceHeartbeat'
import { resolveOwnerIds, saveCheckin } from '../src/api/saveCheckin'
import { updatePatientMemoryFromChat } from '../src/api/patientMemory'
import { beginMirrorSession } from '../src/api/sessionSync'
import { MirrorExperience, type MirrorHomeWidget, type MirrorVisualState } from '../src/components/mirror/MirrorExperience'
import { getMirrorCopy } from '../src/components/mirror/mirrorStrings'
import { fetchCurrentWeather, type CurrentWeather, type WeatherLocation } from '../src/api/weather'
import { DEFAULT_LANGUAGE } from '../src/config/conversationMode'
import {
  ACTIVE_MIRROR_ID_STORAGE_KEY,
  ACTIVE_PATIENT_ID_STORAGE_KEY,
  DEVICE_ID_STORAGE_KEY,
  MIRROR_LANGUAGE_STORAGE_KEY,
  MIRROR_TIMEZONE_STORAGE_KEY,
  NURSE_PATIENT_CONFIG_STORAGE_KEY,
} from '../src/constants/nursePatientConfig'
import { type ChatMessage } from '../src/hooks/conversationTypes'
import { useConversation } from '../src/hooks/useConversation'
import { useWakeWord } from '../src/hooks/useWakeWord'
import { looksLikeGoodbye } from '../src/orchestration/orchestrator'
import { createDailyConversationPlan } from '../src/orchestration/deterministicSpeech'
import { clearDeviceCredential, deviceFetch, getDeviceCredential } from '../src/storage/deviceCredentials'
import { getStoredMirrorProfile } from '../src/storage/mirrorStorage'
import { mirrorColors as c, mirrorFonts as f } from '../src/theme/mirrorTheme'
import { DebugOverlay, dbg } from '../src/debug/debugOverlay'
import { isCheckinDoneToday, markCheckinDoneToday, wakeHourFrom } from '../src/storage/checkinState'

type ClosingStage = 'idle' | 'buffering' | 'goodbye' | 'saving'
type LocalProblem = 'offline' | 'microphone' | 'service' | null

// Show the raw failure reason on the error screen. Reuses the debug-overlay switch, so a test APK
// (EXPO_PUBLIC_DEBUG_OVERLAY=on) is self-diagnosing while a production mirror shows an elder only the
// warm copy. Set it independently with EXPO_PUBLIC_DEBUG_ERRORS=on if you want reasons without the HUD.
const DEBUG_REASON_ON_SCREEN =
  process.env.EXPO_PUBLIC_DEBUG_OVERLAY === 'on' || process.env.EXPO_PUBLIC_DEBUG_ERRORS === 'on'

type SpeechRecognitionResultLike = {
  isFinal: boolean
  length: number
  [index: number]: { transcript: string }
}
type SpeechRecognitionEventLike = {
  resultIndex: number
  results: { length: number; [index: number]: SpeechRecognitionResultLike }
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

// Ambient clock/greeting localized to the patient's conversation language (baseline §7 Phase 2).
// Falls back to English for languages without an explicit greeting table.
const LANGUAGE_LOCALES: Record<string, string> = {
  english: 'en-US', mandarin: 'zh-CN', cantonese: 'zh-HK', minnan: 'zh-Hant',
  malay: 'ms-MY', hindi: 'hi-IN', urdu: 'ur-PK', tamil: 'ta-IN',
}
const GREETINGS: Record<string, [string, string, string]> = {
  english: ['Good morning', 'Good afternoon', 'Good evening'],
  mandarin: ['早安', '午安', '晚上好'],
  cantonese: ['早晨', '午安', '晚上好'],
  minnan: ['早安', '午安', '暗安'],
  malay: ['Selamat pagi', 'Selamat petang', 'Selamat malam'],
  hindi: ['सुप्रभात', 'नमस्कार', 'शुभ संध्या'],
  urdu: ['صبح بخیر', 'آداب', 'شب بخیر'],
  tamil: ['காலை வணக்கம்', 'மதிய வணக்கம்', 'மாலை வணக்கம்'],
}
function localeFor(language: string) { return LANGUAGE_LOCALES[language] || 'en-US' }

function formatGreeting(date: Date, language: string) {
  const hour = date.getHours()
  const table = GREETINGS[language] || GREETINGS.english
  return hour < 12 ? table[0] : hour < 18 ? table[1] : table[2]
}

function formatDate(date: Date, language: string) {
  return new Intl.DateTimeFormat(localeFor(language), { day: 'numeric', month: 'long', weekday: 'long' }).format(date)
}

function formatTime(date: Date, language: string) {
  return new Intl.DateTimeFormat(localeFor(language), { hour: 'numeric', minute: '2-digit' }).format(date)
}

function latestMessage(messages: ChatMessage[], role: 'assistant' | 'user') {
  return messages.filter((message) => message.role === role).at(-1)?.text.trim() ?? ''
}

// Anything matching neither list falls through to 'service' — the "we could not tell what broke" bucket
// the elder sees as "Aria needs a moment.". Two genuine connectivity failures used to land there on
// wording alone: the reconnect watchdog says "timed out" (the old pattern only had "timeout"), and an
// aborted fetch throws "Aborted" — so a slow link was reported as a service fault. Gateway codes
// (api_502/503/504, 408, 429) are plainly connectivity too.
function classifyError(message: string): Exclude<LocalProblem, null> {
  if (/network|offline|connection|reach|socket|closed|timed?\s?out|timeout|fetch|abort|api_(408|429|50[234])/i.test(message)) return 'offline'
  if (/microphone|mic |audio start|permission|audiorecord/i.test(message)) return 'microphone'
  return 'service'
}

export default function ConversationScreen() {
  const params = useLocalSearchParams<{ start?: string }>()
  const requestedScreening = params.start === 'screening'
  const [language, setLanguage] = useState(DEFAULT_LANGUAGE)
  const [persona, setPersona] = useState<'screening' | 'companion'>(requestedScreening ? 'screening' : 'companion')
  const [pendingStart, setPendingStart] = useState(false)
  const [now, setNow] = useState(() => new Date())
  const [patientName, setPatientName] = useState('there')
  const [patientId, setPatientId] = useState<string | null>(null)
  const [dailyPlan, setDailyPlan] = useState(() => createDailyConversationPlan({ patientName: 'there' }))
  const [nurseName, setNurseName] = useState('your caregiver')
  const [endingConversation, setEndingConversation] = useState(false)
  const [closingStage, setClosingStage] = useState<ClosingStage>('idle')
  const [checkingPairing, setCheckingPairing] = useState(true)
  const [wakeListening, setWakeListening] = useState(false)
  const [wakeError, setWakeError] = useState('')
  const [localProblem, setLocalProblem] = useState<LocalProblem>(null)
  // The REASON behind localProblem. It used to be classified into a category and then thrown away, so
  // the only record of why a check-in died was a logcat line — every field failure had to be guessed at.
  // Kept here so a debug build can show it on screen (see DebugReason in MirrorExperience).
  const [problemDetail, setProblemDetail] = useState<string | null>(null)
  const [weather, setWeather] = useState<CurrentWeather | null>(null)
  const [weatherLocation, setWeatherLocation] = useState<WeatherLocation | null>(null)
  const [usualWakeTime, setUsualWakeTime] = useState<string | null>(null)
  const [familyMessages, setFamilyMessages] = useState<FamilyMessage[]>([])
  const [openFamilyMessageId, setOpenFamilyMessageId] = useState<string | null>(null)
  const lastAutoStartDateRef = useRef<string>('')

  const {
    statusKind,
    statusText,
    messages,
    startConversation,
    stopConversation,
    resetStatus,
    connecting,
    sessionActive,
    userSpeaking,
    bargeInActive,
    turnState,
    ended,
    endReason,
    recording,
    beginPushToTalk,
    endPushToTalk,
    getSessionTelemetry,
    getSessionAudio,
  } = useConversation({
    language, persona, patientName, dailyPlan,
    weather: weather ? `${weather.tempC}°C, ${weather.label}`.trim() : undefined,
  })

  const endingRef = useRef(false)
  const endHandledRef = useRef(false)
  const requestedStartHandledRef = useRef(false)
  const wakeRecognitionRef = useRef<SpeechRecognitionLike | null>(null)
  const wakeTriggeredRef = useRef(false)
  const preparingStartRef = useRef(false)
  const shouldRunWakeListenerRef = useRef(false)
  const sessionStartRef = useRef<Date | null>(null)
  const messagesRef = useRef(messages)
  messagesRef.current = messages

  const assistantText = latestMessage(messages, 'assistant')
  const userText = latestMessage(messages, 'user')
  const busy = connecting || sessionActive || endingConversation
  const mirrorCopy = getMirrorCopy(language)
  const progressText = persona === 'screening' ? mirrorCopy.todaysConversation : mirrorCopy.ariaListeningHeader

  const visualState: MirrorVisualState = useMemo(() => {
    if (localProblem === 'offline') return 'offline'
    if (localProblem === 'microphone') return 'microphone_error'
    if (localProblem === 'service') return 'service_error'
    if (endingConversation || turnState === 'closing') return closingStage === 'saving' ? 'saving' : 'closing'
    if (statusKind === 'error') return classifyError(statusText) === 'offline'
      ? 'offline'
      : classifyError(statusText) === 'microphone'
        ? 'microphone_error'
        : 'service_error'
    if (checkingPairing || connecting || turnState === 'connecting' || turnState === 'configuring') return 'connecting'
    if (turnState === 'user_speaking' || userSpeaking) return 'heard'
    if (turnState === 'thinking' || turnState === 'assistant_generating' || statusKind === 'processing') return 'thinking'
    if (turnState === 'assistant_playing' || turnState === 'playback_guard' || statusKind === 'speaking') return 'speaking'
    if (sessionActive || turnState === 'listening' || statusKind === 'listening') return 'listening'
    return 'ambient'
  }, [checkingPairing, closingStage, connecting, endingConversation, localProblem, sessionActive, statusKind, statusText, turnState, userSpeaking])

  // Ambient home widgets: real weather (open-meteo, if a location is configured) + next medication.
  const homeWidgets = useMemo<MirrorHomeWidget[]>(() => {
    const widgets: MirrorHomeWidget[] = []
    if (weather) widgets.push({ icon: 'partly-sunny-outline', label: 'Weather', value: `${weather.tempC}°C ${weather.label}`.trim() })
    const medication = dailyPlan.medicationReminder
    if (medication?.scheduledAt) {
      const when = new Date(medication.scheduledAt)
      if (!Number.isNaN(when.getTime())) widgets.push({ icon: 'medkit-outline', label: 'Medication', value: formatTime(when, language) })
    }
    return widgets.slice(0, 2)
  }, [weather, dailyPlan, language])

  useEffect(() => {
    const timerId = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(timerId)
  }, [])

  // Family messages are a separate, one-way notification channel. Polling retrieves only messages for
  // this paired device's patient; it never generates a reply or adds anything to conversation data.
  useEffect(() => {
    if (checkingPairing) return
    let mounted = true
    const load = async () => {
      try {
        const credential = await getDeviceCredential()
        if (!credential) return
        const messages = await fetchFamilyMessages(credential.deviceId)
        if (mounted) setFamilyMessages(messages)
      } catch {
        // A message refresh must never interrupt the older adult's conversation or turn an offline
        // notification poll into a scary error screen.
      }
    }
    void load()
    const timerId = setInterval(load, 30_000)
    return () => { mounted = false; clearInterval(timerId) }
  }, [checkingPairing])

  useEffect(() => {
    let mounted = true
    async function verifyAndLoadProfile() {
      const pairing = await verifyStoredPairing()
      if (!mounted) return
      if (!pairing.valid) {
        await clearStoredMirrorConnection()
        router.replace('/')
        return
      }
      const profile = await getStoredMirrorProfile()
      if (!mounted) return
      const nextPatientName = profile.patientName ?? pairing.configuration?.patient?.displayName ?? 'there'
      setPatientName(nextPatientName)
      setNurseName(profile.nurseName ?? 'your caregiver')
      if (pairing.configuration?.patient) {
        setPatientId(pairing.configuration.patient.patientId)
        // Ambient-home inputs from the care plan (untyped Records): weather location + usual wake time.
        const carePlan = pairing.configuration.patient.carePlan
        const location = (carePlan?.communicationPreferences as { location?: WeatherLocation } | undefined)?.location
        if (location && (location.city || (typeof location.latitude === 'number' && typeof location.longitude === 'number'))) {
          setWeatherLocation(location)
        }
        const wake = (carePlan?.dailyRoutine as { usualWakeTime?: unknown } | undefined)?.usualWakeTime
        if (typeof wake === 'string' && /^\d{1,2}:\d{2}$/.test(wake.trim())) setUsualWakeTime(wake.trim())
        const plan = await loadDailyConversationPlan({
          patientId: pairing.configuration.patient.patientId,
          patientName: nextPatientName,
        })
        if (!mounted) return
        setDailyPlan(plan)
      } else {
        setDailyPlan(createDailyConversationPlan({ patientName: nextPatientName }))
      }
      const storedLanguage = await AsyncStorage.getItem(MIRROR_LANGUAGE_STORAGE_KEY)
      const storedPatientId = await AsyncStorage.getItem(ACTIVE_PATIENT_ID_STORAGE_KEY)
      if (mounted && storedPatientId?.trim()) setPatientId(storedPatientId.trim())
      if (mounted && storedLanguage?.trim()) setLanguage(storedLanguage.trim())
      if (!pairing.online) setLocalProblem('offline')
      setCheckingPairing(false)
    }
    void verifyAndLoadProfile()
    return () => { mounted = false }
  }, [])

  useEffect(() => {
    if (checkingPairing || !requestedScreening || requestedStartHandledRef.current) return
    requestedStartHandledRef.current = true
    startWith('screening')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [checkingPairing, requestedScreening])

  useEffect(() => {
    if (!endingConversation && turnState !== 'closing') return
    if (closingStage === 'idle') setClosingStage('buffering')
    if (assistantText && looksLikeGoodbye(assistantText)) setClosingStage('goodbye')
  }, [assistantText, closingStage, endingConversation, turnState])

  useEffect(() => {
    if (checkingPairing || busy || localProblem) {
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [busy, checkingPairing, localProblem])

  useEffect(() => subscribeDeviceHeartbeat((heartbeatState) => {
    if (heartbeatState === 'offline' && !busy && !checkingPairing) setLocalProblem('offline')
    if (heartbeatState === 'online') setLocalProblem((problem) => problem === 'offline' ? null : problem)
  }), [busy, checkingPairing])

  // Fetch real weather for the ambient home widget when a location is configured; refresh every 30 min.
  useEffect(() => {
    if (!weatherLocation) return
    let mounted = true
    const load = async () => {
      const result = await fetchCurrentWeather(weatherLocation)
      if (mounted && result) setWeather(result)
    }
    void load()
    const timerId = setInterval(load, 30 * 60 * 1000)
    return () => { mounted = false; clearInterval(timerId) }
  }, [weatherLocation])

  // Scheduled auto-start: at the patient's usual wake time, gently begin a daily check-in — but only
  // when idle (ambient) and not already started today. Wake word / tap still work independently.
  useEffect(() => {
    if (!usualWakeTime) return
    const normalized = usualWakeTime.length === 4 ? `0${usualWakeTime}` : usualWakeTime
    const timerId = setInterval(() => {
      if (busy || checkingPairing || localProblem || endingConversation || visualState !== 'ambient') return
      const current = new Date()
      const hhmm = `${String(current.getHours()).padStart(2, '0')}:${String(current.getMinutes()).padStart(2, '0')}`
      const today = current.toDateString()
      if (hhmm === normalized && lastAutoStartDateRef.current !== today) {
        lastAutoStartDateRef.current = today
        // Only proactively start the check-in if today's isn't already done (the elder may have woken early
        // and checked in manually). Never auto-start an unsolicited free chat.
        void (async () => { if (!(await isCheckinDoneToday(wakeHourFrom(usualWakeTime)))) startWith('screening') })()
      }
    }, 30_000)
    return () => clearInterval(timerId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [usualWakeTime, busy, checkingPairing, localProblem, endingConversation, visualState])

  const handleStart = useCallback(async () => {
    if (busy || checkingPairing) return
    shouldRunWakeListenerRef.current = false
    stopWakeListener()
    setLocalProblem(null)
    sessionStartRef.current = new Date()
    try {
      await beginMirrorSession(
        persona === 'screening' ? 'daily_checkin' : 'companion',
        language,
        persona === 'screening' ? dailyPlan : undefined,
      )
      await startConversation()
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unable to reach the Reflexion service.'
      setProblemDetail(`start: ${message}`)
      setLocalProblem(classifyError(message))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [busy, checkingPairing, dailyPlan, language, persona, startConversation])

  function startWith(nextPersona: 'screening' | 'companion') {
    if (busy || checkingPairing || preparingStartRef.current) return
    if (nextPersona === 'screening') {
      preparingStartRef.current = true
      void (async () => {
        try {
          const refreshedPlan = patientId
            ? await loadDailyConversationPlan({ patientId, patientName, now: new Date() })
            : createDailyConversationPlan({ patientName })
          setDailyPlan(refreshedPlan)
          setPersona('screening')
          setPendingStart(true)
        } finally {
          preparingStartRef.current = false
        }
      })()
      return
    }
    setPersona('companion')
    setPendingStart(true)
  }

  // The mirror decides the conversation kind so the elder never chooses: the FIRST talk of the day (or a
  // forced long-press) runs the check-in; once today's check-in is complete, further talks are free chat.
  function startAuto(forceCheckin: boolean) {
    if (busy || checkingPairing || preparingStartRef.current) return
    void (async () => {
      const runCheckin = forceCheckin || !(await isCheckinDoneToday(wakeHourFrom(usualWakeTime)))
      startWith(runCheckin ? 'screening' : 'companion')
    })()
  }

  useEffect(() => {
    if (!pendingStart || busy) return
    setPendingStart(false)
    void handleStart()
  }, [busy, handleStart, pendingStart])

  // Entry map: the mirror auto-decides. Wake word "Hello Aria" and a short TAP both AUTO-select — the
  // first conversation of the day is the check-in, and once it's done they become free chat. A long-PRESS
  // forces a check-in anytime. The elder never has to pick "check-in vs free talk".
  useWakeWord(
    Platform.OS !== 'web' && !checkingPairing && !busy && !localProblem,
    () => startAuto(false),
  )

  // Debug HUD feeds (no-ops unless EXPO_PUBLIC_DEBUG_OVERLAY=on): pairing ids once on mount, and a poll
  // of the LLM median response latency while a conversation is running (snapshot() is a pure read).
  useEffect(() => {
    let active = true
    void getDeviceCredential().then((cred) => {
      if (active && cred) dbg.patch({ pairing: { deviceId: cred.deviceId, patientId: cred.patientId } })
    }).catch(() => undefined)
    return () => { active = false }
  }, [])
  useEffect(() => {
    if (!busy) return
    const id = setInterval(() => {
      const telemetry = getSessionTelemetry?.()
      if (telemetry) dbg.patch({ llmMedianMs: telemetry.medianResponseLatencyMs })
    }, 2000)
    return () => clearInterval(id)
  }, [busy, getSessionTelemetry])

  const finalize = useCallback(async () => {
    if (endingRef.current) return
    endingRef.current = true
    setEndingConversation(true)
    setClosingStage((stage) => stage === 'goodbye' ? stage : 'buffering')
    try {
      await stopConversation()
      // Capture after stop: the closing goodbye turn + its playback drain are now recorded, and
      // cleanup does not reset the telemetry recorder.
      const telemetry = getSessionTelemetry?.() ?? null
      const sessionAudio = getSessionAudio?.() ?? null
      // Production telemetry breadcrumb (baseline §7 Phase 1 Req 4) — the per-turn detail rides the
      // uploaded capture_metric events; this one summary line is intentionally NOT __DEV__-gated.
      if (telemetry) {
        console.info(
          `[session-telemetry] turns=${telemetry.patientTurns} speechMs=${telemetry.patientSpeechMs} ` +
            `ariaMs=${telemetry.ariaSpeechMs} reprompts=${telemetry.repromptCount} ` +
            `reconnects=${telemetry.reconnectCount} medianLatencyMs=${telemetry.medianResponseLatencyMs ?? 'n/a'}`,
        )
      }
      setClosingStage('saving')
      const ids = await resolveOwnerIds()
      const saveResult = await saveCheckin({
        messages: messagesRef.current,
        startedAt: sessionStartRef.current ?? new Date(),
        endedAt: new Date(),
        nurseId: ids.nurseId,
        patientId: ids.patientId,
        deviceId: ids.deviceId,
        language: ids.language ?? language,
        telemetry,
        sessionAudio,
      })
      // A real check-in (screening + at least one patient turn) marks today's check-in complete, so the
      // mirror routes later conversations to free chat until the next wake-bounded day.
      if (persona === 'screening' && messagesRef.current.some((message) => message.role === 'user')) {
        void markCheckinDoneToday(wakeHourFrom(usualWakeTime))
      }
      // Summarize this conversation into durable, non-clinical facts and merge them into backend memory
      // so Aria remembers the patient next time. Fire-and-forget — never delay the closing screen for an
      // LLM round-trip; a failure just leaves the prior memory untouched.
      if (ids.patientId) {
        void updatePatientMemoryFromChat(messagesRef.current).catch(() => undefined)
      }
      router.replace({
        pathname: '/conversation-closing',
        params: { nurseName, sync: saveResult.saved ? 'synced' : 'queued', language },
      })
    } catch (error) {
      endingRef.current = false
      setEndingConversation(false)
      setClosingStage('idle')
      const saveMessage = error instanceof Error ? error.message : 'Unable to save the conversation.'
      setProblemDetail(`save: ${saveMessage}`)
      setLocalProblem(classifyError(saveMessage))
    }
  }, [language, nurseName, persona, usualWakeTime, stopConversation])

  // A realtime failure before the patient ever answered (endReason='error', zero user turns) is a
  // startup/connection error, NOT a completed check-in. Don't run finalize (which would save a bogus
  // turns-0 check-in and show the "check-in saved / goodbye" screen). Instead tear the session down and
  // bounce through '/' so the mirror returns to a fresh ambient state and keeps listening for the wake
  // word — no fake goodbye, no polluted caregiver data.
  const abandonFailedStart = useCallback(async () => {
    try { await stopConversation() } catch { /* transport already torn down */ }
    router.replace('/')
  }, [stopConversation])

  useEffect(() => {
    if (!ended) { endHandledRef.current = false; return }
    if (endHandledRef.current) return
    endHandledRef.current = true
    const userTurns = messagesRef.current.filter((message) => message.role === 'user').length
    // Any zero-user-turn end is a failed/degenerate session, never a real check-in — whether it ended as
    // 'error' or a self-marched 'goodbye' (e.g. broken ASR let the flow run to completion with no
    // answers). Don't save a bogus empty check-in into the caregiver pipeline or announce a fake
    // goodbye; tear down and reset to ambient instead.
    if (userTurns === 0) { void abandonFailedStart(); return }
    void finalize()
  }, [ended, endReason, abandonFailedStart, finalize])

  async function retryProblem() {
    const pairing = await verifyStoredPairing()
    if (!pairing.valid) {
      await clearStoredMirrorConnection()
      router.replace('/')
      return
    }
    if (!pairing.online) {
      setProblemDetail('retry: device offline (pairing check)')
      setLocalProblem('offline')
      return
    }
    try {
      await sendDeviceHeartbeat()
    } catch (error) {
      setProblemDetail(`retry: heartbeat failed — ${error instanceof Error ? error.message : 'unknown'}`)
      setLocalProblem('offline')
      return
    }
    // Also un-latch an error the HOOK is holding. Without this, "Try again" only cleared the problems
    // this screen owned, so on any realtime/TTS failure the button visibly did nothing.
    resetStatus?.()
    setProblemDetail(null)
    setLocalProblem(null)
  }

  function startWakeListener() {
    if (Platform.OS !== 'web' || wakeRecognitionRef.current) return
    const SpeechRecognition = getSpeechRecognitionConstructor()
    if (!SpeechRecognition) {
      setWakeListening(false)
      setWakeError(mirrorCopy.tapToStartWake)
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
        for (let alternative = 0; alternative < result.length; alternative += 1) {
          transcript += ` ${result[alternative]?.transcript ?? ''}`
        }
      }
      if (/\b(?:hello|hi)\s+(?:aria|arya)\b/i.test(transcript) && !wakeTriggeredRef.current) {
        wakeTriggeredRef.current = true
        shouldRunWakeListenerRef.current = false
        stopWakeListener()
        startWith('screening')
      }
    }
    recognition.onerror = () => {
      setWakeListening(false)
      setWakeError(mirrorCopy.tapToStartWake)
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
    try { recognition.stop() } catch { /* no-op */ }
    setWakeListening(false)
  }

  const newestFamilyMessage = familyMessages.at(0)
  async function openFamilyMessage(message: FamilyMessage) {
    setOpenFamilyMessageId(message.messageId)
    try {
      const credential = await getDeviceCredential()
      if (!credential) return
      const opened = await markFamilyMessageOpened(credential.deviceId, message.messageId)
      setFamilyMessages((current) => current.map((item) => item.messageId === opened.messageId ? opened : item))
    } catch {
      // The content is already on screen after an explicit tap. A later poll can reconcile delivery state.
    }
  }

  return (
    <View style={styles.root}>
      <MirrorExperience
        assistantText={assistantText}
        bargeInActive={bargeInActive}
        date={formatDate(now, language)}
        greeting={formatGreeting(now, language)}
        language={language}
        homeWidgets={homeWidgets}
        microphoneActive={sessionActive}
        onBegin={() => startAuto(false)}
        onBeginCheckin={() => startAuto(true)}
        onEnd={() => void finalize()}
        onRetry={() => void retryProblem()}
        patientName={patientName}
        problemDetail={DEBUG_REASON_ON_SCREEN ? (problemDetail || (statusKind === 'error' ? `realtime: ${statusText}` : undefined)) : undefined}
        progressText={progressText}
        state={visualState}
        time={formatTime(now, language)}
        userText={userText}
        wakeError={wakeError}
        wakeListening={wakeListening}
      />

      {visualState === 'ambient' && newestFamilyMessage ? (
        <Pressable
          accessibilityLabel={openFamilyMessageId === newestFamilyMessage.messageId ? 'Close family message' : 'Open family message'}
          accessibilityRole="button"
          onPress={() => openFamilyMessageId === newestFamilyMessage.messageId
            ? setOpenFamilyMessageId(null)
            : void openFamilyMessage(newestFamilyMessage)}
          style={styles.familyMessageNotice}
        >
          <Ionicons name="heart-outline" size={25} color={c.sageDeep} />
          <View style={styles.familyMessageCopy}>
            <Text style={styles.familyMessageTitle}>A message from your family</Text>
            {openFamilyMessageId === newestFamilyMessage.messageId
              ? <Text style={styles.familyMessageBody}>{newestFamilyMessage.body}</Text>
              : <Text style={styles.familyMessageHint}>Tap to open when you’re ready.</Text>}
          </View>
          <Ionicons name={openFamilyMessageId === newestFamilyMessage.messageId ? 'chevron-down' : 'chevron-forward'} size={22} color={c.sageDeep} />
        </Pressable>
      ) : null}

      {beginPushToTalk && endPushToTalk && busy && !endingConversation && visualState !== 'service_error' ? (
        <Pressable
          accessibilityHint="Hold while speaking and release when finished"
          accessibilityLabel="Hold to speak"
          disabled={!recording && statusKind !== 'listening'}
          onPressIn={beginPushToTalk}
          onPressOut={endPushToTalk}
          style={({ pressed }) => [
            styles.pushToTalk,
            recording && styles.pushToTalkActive,
            pressed && styles.pushToTalkPressed,
          ]}
        >
          <Ionicons name={recording ? 'radio' : 'mic'} size={22} color={recording ? c.ink : c.linen} />
          <Text style={[styles.pushToTalkText, recording && styles.pushToTalkTextActive]}>
            {recording ? 'Listening — release when finished' : 'Hold to speak'}
          </Text>
        </Pressable>
      ) : null}

      <DebugOverlay />
    </View>
  )
}

async function verifyStoredPairing(): Promise<{ valid: boolean; online: boolean; configuration?: DeviceConfiguration }> {
  const credential = await getDeviceCredential()
  if (!credential) return { valid: false, online: false }
  try {
    const response = await deviceFetch(`/api/v1/devices/${encodeURIComponent(credential.deviceId)}/configuration`)
    if (response.status === 401 || response.status === 403 || response.status === 404) {
      return { valid: false, online: true }
    }
    const configuration = await dataOrThrow<DeviceConfiguration>(response)
    if (!configuration.patient) return { valid: false, online: true }
    await persistDeviceProfile(credential.deviceId, configuration)
    return { valid: true, online: true, configuration }
  } catch {
    return { valid: true, online: false }
  }
}

async function persistDeviceProfile(deviceId: string, configuration: DeviceConfiguration) {
  if (!configuration.patient) return
  await AsyncStorage.multiSet([
    [DEVICE_ID_STORAGE_KEY, deviceId],
    [ACTIVE_MIRROR_ID_STORAGE_KEY, deviceId],
    [ACTIVE_PATIENT_ID_STORAGE_KEY, configuration.patient.patientId],
    [NURSE_PATIENT_CONFIG_STORAGE_KEY, JSON.stringify({ patient: configuration.patient })],
    [MIRROR_LANGUAGE_STORAGE_KEY, configuration.patient.preferredLanguage || DEFAULT_LANGUAGE],
    [MIRROR_TIMEZONE_STORAGE_KEY, configuration.patient.timezone || Intl.DateTimeFormat().resolvedOptions().timeZone],
  ])
}

async function clearStoredMirrorConnection() {
  await clearDeviceCredential({ preserveBootstrap: true })
  await AsyncStorage.multiRemove([
    ACTIVE_MIRROR_ID_STORAGE_KEY,
    ACTIVE_PATIENT_ID_STORAGE_KEY,
    NURSE_PATIENT_CONFIG_STORAGE_KEY,
    MIRROR_LANGUAGE_STORAGE_KEY,
    MIRROR_TIMEZONE_STORAGE_KEY,
  ])
}

const styles = StyleSheet.create({
  root: { backgroundColor: c.cream, flex: 1 },
  familyMessageNotice: { alignItems: 'center', backgroundColor: 'rgba(255,252,244,0.97)', borderColor: c.line, borderRadius: 18, borderWidth: 1, bottom: 30, flexDirection: 'row', gap: 12, left: 38, maxWidth: 460, padding: 16, position: 'absolute', right: 38 },
  familyMessageCopy: { flex: 1 },
  familyMessageTitle: { color: c.ink, fontFamily: f.bodyMedium, fontSize: 17 },
  familyMessageHint: { color: c.sageDeep, fontFamily: f.body, fontSize: 14, marginTop: 3 },
  familyMessageBody: { color: c.ink, fontFamily: f.body, fontSize: 16, lineHeight: 22, marginTop: 5 },
  pushToTalk: { alignItems: 'center', backgroundColor: c.inkLift, borderColor: c.line, borderRadius: 28, borderWidth: 1, bottom: 28, flexDirection: 'row', gap: 9, left: 38, paddingHorizontal: 20, paddingVertical: 14, position: 'absolute' },
  pushToTalkActive: { backgroundColor: c.linen, borderColor: c.linen },
  pushToTalkPressed: { transform: [{ scale: 0.98 }] },
  pushToTalkText: { color: c.linen, fontFamily: f.bodyMedium, fontSize: 14 },
  pushToTalkTextActive: { color: c.ink },
})
