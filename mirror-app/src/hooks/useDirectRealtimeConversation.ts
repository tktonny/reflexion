import { useCallback, useEffect, useRef, useState } from 'react'
import { Platform } from 'react-native'

import { getBearer } from '../api/qwenToken'
import { qwenTTS } from '../api/qwenClient'
import {
  looksLikeGoodbye,
  looksLikeUserGoodbye,
} from '../orchestration/orchestrator'
import { buildLiveSessionUpdate, realtimeWsUrl } from '../orchestration/realtime'
import { createEnergyVad, decodeBase64Pcm16 } from '../orchestration/energyVad'
import {
  acknowledgementForLanguage,
  base64ToBytes,
  closingTextForLanguage,
  companionClosingTextForLanguage,
  createDailyConversationPlan,
  dailyConversationMetadataForPatientTurn,
  openingTextForLanguage,
  qwenWavToPcm24kChunks,
  screeningQuestionForTurn,
  takeYourTimeForLanguage,
} from '../orchestration/deterministicSpeech'
import { createSessionTelemetry } from '../orchestration/sessionTelemetry'
import { createSessionCheckinFlow, type DailyCheckinFlow } from '../orchestration/dailyCheckinFlow'
import { getStoredSessionMemory } from '../storage/mirrorStorage'
import {
  acquireConversationRuntime,
  type ConversationRuntimeLease,
} from '../orchestration/conversationRuntime'
import { createPushToTalkGesture } from '../orchestration/pushToTalkGesture'
import {
  choosePostPlaybackAction,
  createTurnTakingState,
  playbackDrainDecision,
  reduceTurnTaking,
  type TurnTakingEvent,
  type TurnTakingPhase,
} from '../orchestration/turnTaking'
import {
  detectLanguageSignal,
  voiceProfileForLanguageKey,
  voiceProfileForSession,
  type VoiceProfile,
} from '../orchestration/voice'
import { createPcmAudioBridge, type PcmAudioBridge } from '../native/pcmAudio'
import { randomId } from '../utils/id'
import type { ChatMessage, ConversationApi, ConversationOptions, StatusKind } from './conversationTypes'

type Options = ConversationOptions & {
  onUnavailable?: (reason: string) => void
}

// --- automatic echo suppression (hands-free) ---
// On devices with no hardware AEC (e.g. an emulator sharing mic+speaker), the assistant's own audio
// can leak into the mic during the un-mute tail and get transcribed as a "user" turn, causing Aria to
// talk to herself. We compare each incoming transcript to Aria's recent lines and drop close matches.
function normEcho(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9一-鿿]/g, '')
}
function bigramJaccard(a: string, b: string): number {
  const grams = (s: string) => {
    const g = new Set<string>()
    for (let i = 0; i < s.length - 1; i += 1) g.add(s.slice(i, i + 2))
    return g
  }
  const ga = grams(a)
  const gb = grams(b)
  if (ga.size === 0 || gb.size === 0) return 0
  let inter = 0
  for (const x of ga) if (gb.has(x)) inter += 1
  return inter / (ga.size + gb.size - inter)
}
function looksLikeEchoOfAria(userText: string, ariaLines: string[]): boolean {
  const u = normEcho(userText)
  if (u.length < 4) return false // too short to judge; let it through
  for (const line of ariaLines) {
    const a = normEcho(line)
    if (a.length < 4) continue
    if (a.includes(u) || u.includes(a)) return true
    if (bigramJaccard(u, a) >= 0.5) return true
  }
  return false
}

function providerResponseId(payload: any): string | null {
  const value = payload?.response_id ?? payload?.response?.id
  return typeof value === 'string' && value ? value : null
}

function isCancelledProviderEvent(payload: any, cancelledResponseId: string | null): boolean {
  return Boolean(cancelledResponseId && providerResponseId(payload) === cancelledResponseId)
}

// Approximate playback duration of a base64 PCM16 mono @24kHz chunk. Used ONLY as a fail-safe floor
// when the native backlog is unmeasurable (baseline §6), so a stub getPlaybackBacklogMs()===0 can
// never make the drain guard declare "drained" before the audio has actually had time to play.
function pcm24kBase64DurationMs(base64: string): number {
  if (!base64) return 0
  return Math.round((base64.length * 0.75) / 2 / 24)
}

// --- barge-in (talk-over) tuning ---------------------------------------------------------------
// The mirror plays Aria through a loud speaker. On devices whose hardware AEC is weak or absent that
// output leaks into the mic, and the energy VAD that powers barge-in mistakes Aria's OWN voice for the
// user starting to speak: it fires interruptAssistant() mid-sentence ("I heard you — I've paused"),
// cutting her off, and when it lands on the opening line it cancels that turn and can cascade the whole
// session to an early close. These knobs make barge-in reject echo. They are env-tunable so the floor
// can be calibrated on the real hardware without rebuilding the APK:
//   EXPO_PUBLIC_BARGEIN=off            → disable talk-over entirely (half-duplex: the mic stays muted
//                                        while Aria speaks). The most robust setting for a bad-AEC room.
//   EXPO_PUBLIC_BARGEIN_START_RMS=0.06 → loudness (0..1) the mic must reach to count as an interruption.
//   EXPO_PUBLIC_BARGEIN_MIN_MS=700     → how long that loudness must persist before we believe it.
//   EXPO_PUBLIC_BARGEIN_GRACE_MS=900   → ignore barge-in for this long after Aria STARTS an utterance,
//                                        where echo onset is loudest and a real interruption never is.
function numberFromEnv(raw: string | undefined, fallback: number): number {
  const value = Number(raw)
  return Number.isFinite(value) && value > 0 ? value : fallback
}
const BARGE_IN_ENABLED = (process.env.EXPO_PUBLIC_BARGEIN ?? 'on') !== 'off'
const BARGE_IN_START_RMS = numberFromEnv(process.env.EXPO_PUBLIC_BARGEIN_START_RMS, 0.06)
const BARGE_IN_MIN_MS = numberFromEnv(process.env.EXPO_PUBLIC_BARGEIN_MIN_MS, 700)
const BARGE_IN_GRACE_MS = numberFromEnv(process.env.EXPO_PUBLIC_BARGEIN_GRACE_MS, 900)

// Auto-reconnect a bounded number of times per turn-gap on a transient realtime WS drop (common on the
// cross-border China link, which otherwise dumps the user to the goodbye screen). Before the first
// patient turn we restart fresh; mid-conversation we RESUME (screening re-asks the current scripted
// question, companion reopens the mic — its provider-side context is gone, re-seeded from recent
// transcripts via session memory). The budget refreshes after each successful patient turn, so a long
// conversation survives several isolated blips without an unbounded loop inside one stall. Set the env
// var to 0 to disable reconnect entirely.
const MAX_RECONNECTS_PER_GAP = (() => {
  const value = Number(process.env.EXPO_PUBLIC_REALTIME_MAX_RECONNECTS)
  return Number.isInteger(value) && value >= 0 ? value : 1
})()

/**
 * Version 3 (Flavor A): NATIVE device opens a direct realtime WebSocket to Qwen (header auth
 * with a short-lived ticket minted for an authenticated `/api/v1/sessions/:id` session), running the on-device orchestration
 * (session.update / server-VAD / dynamic voice / wrap-up). No relay.
 *
 * Verified headlessly by server/smoke-direct-ws.mjs (token → direct WS → orchestration →
 * audio deltas). The only device-bound gap is native PCM capture/playback (src/native/pcmAudio.ts).
 * Web cannot set WS headers, so the selector routes web to the relay instead of this hook.
 */
export function useDirectRealtimeConversation(options: Options = {}): ConversationApi {
  const patientId = options.patientId ?? 'demo-patient'
  const language = options.language ?? 'en'
  const persona = options.persona ?? 'screening'
  const pushToTalk = options.pushToTalk ?? false
  const speechRate = typeof options.speechRate === 'number' ? options.speechRate : 0.85
  const dailyPlan = options.dailyPlan ?? createDailyConversationPlan({ patientName: options.patientName, reminiscenceWeekdays: [] })

  const [statusKind, setStatusKind] = useState<StatusKind>('idle')
  const [statusText, setStatusText] = useState('Ready to start')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [connecting, setConnecting] = useState(false)
  const [sessionActive, setSessionActive] = useState(false)
  const [userSpeaking, setUserSpeaking] = useState(false)
  const [bargeInActive, setBargeInActive] = useState(false)
  const [ended, setEnded] = useState(false)
  const [recording, setRecording] = useState(false)
  const [turnState, setTurnState] = useState<TurnTakingPhase>('idle')

  const instanceIdRef = useRef(randomId('direct'))
  const runtimeLeaseRef = useRef<ConversationRuntimeLease | null>(null)
  const endedRef = useRef(false)
  // Why the session ended: 'goodbye' (a real completed check-in) vs 'error' (failClosed). The screen
  // uses this to avoid saving/announcing a bogus check-in when a startup failure ends a turns-0 session.
  const endReasonRef = useRef<'goodbye' | 'error' | null>(null)
  // Bounded reconnect bookkeeping (see MAX_RECONNECTS_PER_GAP). isReconnectRef tells the reset block in
  // startConversation NOT to zero the counter on a reconnect-driven restart.
  const reconnectAttemptsRef = useRef(0)
  const isReconnectRef = useRef(false)
  const startConversationRef = useRef<(opts?: { resume?: boolean }) => Promise<void>>(async () => {})
  const socketRef = useRef<WebSocket | null>(null)
  const phase1InjectorRef = useRef<WebSocket | null>(null)
  const phase1InjectionChainRef = useRef<Promise<void>>(Promise.resolve())
  const phase1InjectionEpochRef = useRef(0)
  const phase1StopAtRef = useRef<TurnTakingPhase | null>(null)
  const phase1StopConversationRef = useRef<() => void>(() => {})
  // startConversation may be invoked twice before getBearer resolves (for example by a development
  // StrictMode effect). socketRef is still null during that window, so it cannot be the startup lock.
  const startingRef = useRef(false)
  const startAttemptRef = useRef(0)
  const audioRef = useRef<PcmAudioBridge | null>(null)
  const vadRef = useRef(createEnergyVad())
  // Barge-in is intentionally far more conservative than ordinary turn VAD. It runs while Aria's
  // speaker is playing, so its floor must sit ABOVE the residual echo a weak-AEC device leaks into the
  // mic — otherwise Aria interrupts herself. Floor + confirmation window are env-tunable (see the
  // BARGE_IN_* constants) so a noisy/bad-AEC room can be calibrated on the device.
  const bargeInVadRef = useRef(createEnergyVad({
    speechStartRms: BARGE_IN_START_RMS,
    speechContinueRms: 0.01,
    minSpeechMs: BARGE_IN_MIN_MS,
    silenceMs: 1200,
  }))
  // Wall-clock (ms) when Aria's current utterance began playing; used for the start-of-utterance
  // barge-in grace window where echo onset is strongest.
  const playbackStartedAtRef = useRef(0)
  const vadSpeakingRef = useRef(false)
  const audioPreRollRef = useRef<string[]>([])
  const bargeInPreRollRef = useRef<string[]>([])
  const bargeInActiveRef = useRef(false)
  const voiceRef = useRef<VoiceProfile>(voiceProfileForSession(language))
  const openingRequestedRef = useRef(false)
  // True on the first session.updated after a MID-CONVERSATION reconnect: skip the opening and instead
  // resume where we left off (re-ask the current screening question / reopen the mic for companion).
  const resumingRef = useRef(false)
  const responseAfterSessionUpdateRef = useRef<'normal' | 'closing' | null>(null)
  const streamIdRef = useRef<string | null>(null)
  const assistantTextRef = useRef('')
  const recentAriaRef = useRef<string[]>([]) // Aria's last few finalized lines, for echo suppression
  const userTranscriptsRef = useRef<string[]>([])
  const drainTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const transcriptTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const wrapupTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const wrappingUpRef = useRef(false)
  const turnTakingRef = useRef(createTurnTakingState())
  const responseRequestedRef = useRef(false)
  const responseActiveRef = useRef(false)
  const responseAudioReceivedRef = useRef(false)
  const responseCompletedRef = useRef(false)
  const activeProviderResponseIdRef = useRef<string | null>(null)
  const cancelledProviderResponseIdRef = useRef<string | null>(null)
  const currentResponseSourceRef = useRef<'realtime' | 'local' | null>(null)
  const currentResponseClosingRef = useRef(false)
  const goodbyeDetectedRef = useRef(false)
  const drainInProgressRef = useRef(false)
  const manualCloseRequestedRef = useRef(false)
  const stopPromiseRef = useRef<Promise<void> | null>(null)
  const stopResolveRef = useRef<(() => void) | null>(null)
  // Fallback plumbing: tell the supervisor (useConversation) when omni is unavailable so it can drop
  // to the turn-based (v2) stack. Only fires during STARTUP — once real audio/response has flowed
  // (hadResponseRef) a mid-session blip is surfaced as an error, not restarted in a new transport.
  const onUnavailableRef = useRef(options.onUnavailable)
  onUnavailableRef.current = options.onUnavailable
  const hadResponseRef = useRef(false)
  const openedRef = useRef(false)
  const connectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reportedUnavailableRef = useRef(false)
  // Deterministic Daily Conversation v2 stage order and natural ending.
  const turnCountRef = useRef(0)
  const closingRef = useRef(false)
  const recordingRef = useRef(false)
  const pushToTalkGestureRef = useRef(createPushToTalkGesture())
  const handlePlaybackCompleteRef = useRef<(closing: boolean) => void>(() => {})
  // Per-turn / per-session telemetry (baseline §3). Additive observation only — never gates control flow.
  const telemetryRef = useRef(createSessionTelemetry())
  const sessionStartedAtMsRef = useRef(0)
  const endedAtMsRef = useRef(0)
  // Per-question state machine (screening only) — authority for question order/completion/reprompt.
  const checkinFlowRef = useRef<DailyCheckinFlow | null>(null)
  // Soft cross-session continuity memory injected into Aria's instructions ("last time you mentioned…").
  const memoryRef = useRef<string[]>([])
  // Silence handling (doc: wait 7–10s, then a single gentle "Take your time", never rush).
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const silenceNudgedRef = useRef(false)
  const onSilenceRef = useRef<() => void>(() => {})
  // Enqueued playback duration of the current Aria response — the fail-safe floor for the drain guard.
  const enqueuedPlaybackMsRef = useRef(0)
  // Raw patient-speech PCM16 frames for the session audio artifact (recorded only while mic is open).
  const sessionAudioFramesRef = useRef<string[]>([])
  const CAPTURE_SAMPLE_RATE = 16000
  const MAX_AUDIO_FRAMES = 12_000 // bound memory (~a long check-in); older frames beyond this are dropped

  const updateStatus = useCallback((kind: StatusKind, text: string) => {
    setStatusKind(kind)
    setStatusText(text)
  }, [])

  const transition = useCallback((event: TurnTakingEvent) => {
    const previousViolationCount = turnTakingRef.current.violations.length
    const next = reduceTurnTaking(turnTakingRef.current, event)
    turnTakingRef.current = next
    setTurnState(next.phase)
    if (__DEV__ && phase1StopAtRef.current === next.phase) {
      phase1StopAtRef.current = null
      setTimeout(() => phase1StopConversationRef.current(), 0)
    }
    if (__DEV__) {
      console.info(
        `[turn-taking:${instanceIdRef.current}] #${next.sequence} ${event.type} -> ${next.phase} ` +
          `response=${next.responseInFlight ? 'active' : 'idle'} playback=${next.awaitingPlayback ? 'pending' : 'clear'} mic=${next.captureMuted ? 'muted' : 'open'}`,
      )
    }
    if (next.violations.length > previousViolationCount) {
      console.warn(`[turn-taking:${instanceIdRef.current}] ${next.violations.at(-1)}`)
    }
    return next
  }, [])

  const resolveStop = useCallback(() => {
    const resolve = stopResolveRef.current
    stopResolveRef.current = null
    stopPromiseRef.current = null
    resolve?.()
  }, [])

  const reportUnavailable = useCallback((reason: string) => {
    if (hadResponseRef.current || reportedUnavailableRef.current) return
    reportedUnavailableRef.current = true
    if (connectTimerRef.current) { clearTimeout(connectTimerRef.current); connectTimerRef.current = null }
    // Non-__DEV__ breadcrumb (like [session-telemetry]) so the reason for a v3→fallback is visible in a
    // production logcat — critical for telling a real device startup failure from an echo/barge-in one.
    console.info(`[realtime] unavailable → turn-based fallback: ${reason}`)
    onUnavailableRef.current?.(reason)
  }, [])

  // The closing response has completed AND its native playback has drained. There is deliberately
  // no normal-path timer here: provider generation completion is not speaker completion.
  const completeConversation = useCallback(() => {
    if (endedRef.current) return
    endedRef.current = true
    endReasonRef.current = 'goodbye'
    endedAtMsRef.current = Date.now()
    wrappingUpRef.current = true
    audioRef.current?.setCaptureMuted(true)
    transition({ type: 'finished' })
    updateStatus('processing', 'Goodbye complete. Saving the conversation...')
    setEnded(true)
    resolveStop()
  }, [resolveStop, transition, updateStatus])

  const failClosed = useCallback((reason: string) => {
    if (endedRef.current) return
    endedRef.current = true
    endReasonRef.current = 'error'
    endedAtMsRef.current = Date.now()
    wrappingUpRef.current = true
    audioRef.current?.setCaptureMuted(true)
    // Non-__DEV__ breadcrumb: failClosed ends the session and the screen then navigates to the closing
    // scene, masking the reason. Log it (with whether a turn had happened) so a production logcat shows
    // exactly why the mirror ended — the definitive signal for a non-echo "opening → immediate end".
    console.warn(`[realtime] failClosed (hadResponse=${hadResponseRef.current}, turns=${turnCountRef.current}): ${reason}`)
    transition({ type: 'failed', reason })
    updateStatus('error', reason)
    setEnded(true)
    resolveStop()
  }, [resolveStop, transition, updateStatus])

  const clearDrain = useCallback(() => {
    if (drainTimerRef.current) {
      clearTimeout(drainTimerRef.current)
      drainTimerRef.current = null
    }
    drainInProgressRef.current = false
  }, [])

  const clearTranscriptWait = useCallback(() => {
    if (transcriptTimerRef.current) {
      clearTimeout(transcriptTimerRef.current)
      transcriptTimerRef.current = null
    }
  }, [])

  const SILENCE_MS = 8000 // within the doc's 7–10s window
  const clearSilenceTimer = useCallback(() => {
    if (silenceTimerRef.current) { clearTimeout(silenceTimerRef.current); silenceTimerRef.current = null }
  }, [])
  const armSilenceTimer = useCallback(() => {
    clearSilenceTimer()
    // Companion gets at most one gentle nudge per silence gap; screening is governed by the flow's cap.
    if (persona === 'companion' && silenceNudgedRef.current) return
    silenceTimerRef.current = setTimeout(() => onSilenceRef.current(), SILENCE_MS)
  }, [clearSilenceTimer, persona])

  const send = useCallback((event: Record<string, unknown>) => {
    const s = socketRef.current
    if (s && s.readyState === WebSocket.OPEN) s.send(JSON.stringify(event))
  }, [])

  const configureNextResponse = useCallback((
    sessionUpdate: Record<string, unknown>,
    intent: 'normal' | 'closing',
  ) => {
    if (responseAfterSessionUpdateRef.current) {
      failClosed('A second response configuration was queued before the first was acknowledged.')
      return false
    }
    responseAfterSessionUpdateRef.current = intent
    if (__DEV__) console.info(`[turn-taking] queued ${intent} session update`)
    send(sessionUpdate)
    return true
  }, [failClosed, send])

  const appendAssistantStreaming = useCallback((text: string) => {
    if (!streamIdRef.current) {
      const id = randomId('assistant')
      streamIdRef.current = id
      setMessages((prev) => [...prev, { id, role: 'assistant', text, streaming: true }])
      return
    }
    const id = streamIdRef.current
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, text } : m)))
  }, [])

  const finalizeAssistant = useCallback((text: string) => {
    const id = streamIdRef.current
    streamIdRef.current = null
    assistantTextRef.current = ''
    const clean = text.trim()
    if (!clean) return
    recentAriaRef.current = [clean, ...recentAriaRef.current].slice(0, 4)
    if (id) setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, text: clean, streaming: false } : m)))
    else setMessages((prev) => [...prev, { id: randomId('assistant'), role: 'assistant', text: clean }])
  }, [])

  const markAssistantInterrupted = useCallback(() => {
    const streamingId = streamIdRef.current
    const partialText = assistantTextRef.current.trim()
    streamIdRef.current = null
    assistantTextRef.current = ''
    // A provider turn (e.g. the opening line) is only added to the echo-suppression list on
    // response.done. If it is interrupted before that — which on a bad-AEC device is usually Aria's own
    // echo tripping barge-in — its text never lands there, so the echo transcript is NOT recognised as
    // Aria and gets accepted as a real user turn. Register the partial line now so the echo is dropped.
    if (partialText) recentAriaRef.current = [partialText, ...recentAriaRef.current].slice(0, 4)
    setMessages((previous) => {
      if (streamingId) {
        return previous.map((message) => message.id === streamingId
          ? { ...message, text: partialText || message.text, streaming: false, interrupted: true }
          : message)
      }
      const lastAssistantIndex = previous.findLastIndex((message) => message.role === 'assistant')
      if (lastAssistantIndex < 0) return previous
      return previous.map((message, index) => index === lastAssistantIndex
        ? { ...message, interrupted: true }
        : message)
    })
  }, [])

  const applyVoice = useCallback((profile: VoiceProfile) => {
    voiceRef.current = profile
  }, [])

  const waitForPlaybackDrain = useCallback((onDrained: () => void) => {
    const bridge = audioRef.current
    clearDrain()
    drainInProgressRef.current = true
    const startedAt = Date.now()
    const THRESHOLD_MS = 40
    const MAX_WAIT_MS = 25_000

    const poll = () => {
      if (bridge !== audioRef.current) {
        drainTimerRef.current = null
        drainInProgressRef.current = false
        return
      }
      const elapsed = Date.now() - startedAt
      // When the native backlog is unmeasurable (stub / missing method) a reported 0 means "unknown",
      // not "drained" — fall back to the enqueued-audio floor so a full utterance always plays out.
      const measurable = bridge?.isBacklogMeasurable?.() ?? false
      const backlogMs = measurable
        ? (bridge?.getPlaybackBacklogMs?.() ?? 0)
        : Math.max(0, enqueuedPlaybackMsRef.current - elapsed)
      const decision = playbackDrainDecision(backlogMs, elapsed, THRESHOLD_MS, MAX_WAIT_MS)
      if (decision.timedOut) {
        drainTimerRef.current = null
        drainInProgressRef.current = false
        failClosed('Audio playback stalled; microphone kept muted for safety.')
        return
      }
      if (decision.drained) {
        drainTimerRef.current = null
        drainInProgressRef.current = false
        if (__DEV__) console.info(`[turn-taking] playback drained in ${Date.now() - startedAt}ms (backlog=${backlogMs}ms)`)
        telemetryRef.current.onAriaPlaybackFinished(Date.now())
        transition({ type: 'playback_drained' })
        onDrained()
        return
      }
      drainTimerRef.current = setTimeout(poll, Math.min(Math.max(backlogMs, 40), 250))
    }

    poll()
  }, [clearDrain, failClosed, transition])

  const playDeterministicResponse = useCallback(async (text: string, closing: boolean) => {
    const lease = runtimeLeaseRef.current
    if (!lease?.isCurrent()) return
    try {
      responseRequestedRef.current = false
      responseActiveRef.current = true
      responseCompletedRef.current = false
      responseAudioReceivedRef.current = false
      currentResponseClosingRef.current = closing
      currentResponseSourceRef.current = 'local'
      goodbyeDetectedRef.current = closing
      audioRef.current?.setCaptureMuted(true)
      transition({ type: 'response_created' })
      telemetryRef.current.onAriaResponseCreated(Date.now())
      updateStatus('speaking', 'Speaking...')

      const tts = await qwenTTS(text, { voice: voiceRef.current.voice, rate: speechRate })
      // Bail if the session was torn down OR failed while the TTS request was in flight — otherwise a
      // late-resolving opening would enqueue audio and re-open the mic after failClosed already ended it.
      if (!lease.isCurrent() || endedRef.current) return
      let wav: Uint8Array
      if (tts.url) {
        const response = await fetch(tts.url)
        if (!lease.isCurrent() || endedRef.current) return
        if (!response.ok) throw new Error(`Qwen TTS audio download failed (${response.status}).`)
        wav = new Uint8Array(await response.arrayBuffer())
      } else if (tts.audioBase64) {
        wav = base64ToBytes(tts.audioBase64)
      } else {
        throw new Error('Qwen TTS returned no audio.')
      }
      const chunks = qwenWavToPcm24kChunks(wav)
      if (chunks.length === 0) throw new Error('Qwen TTS returned empty audio.')

      finalizeAssistant(text)
      responseAudioReceivedRef.current = true
      transition({ type: 'audio_delta' })
      telemetryRef.current.onAriaFirstAudio(Date.now())
      playbackStartedAtRef.current = Date.now()
      enqueuedPlaybackMsRef.current = 0
      for (const chunk of chunks) {
        enqueuedPlaybackMsRef.current += pcm24kBase64DurationMs(chunk)
        audioRef.current?.play(chunk)
      }

      responseCompletedRef.current = true
      responseActiveRef.current = false
      // Keep capture frames local while the speaker is active. JS sends nothing upstream unless
      // conservative barge-in VAD confirms that the person has actually started speaking. With barge-in
      // disabled the mic stays muted for the whole utterance (true half-duplex — no echo can reach it).
      bargeInVadRef.current.reset()
      bargeInPreRollRef.current = []
      audioRef.current?.setCaptureMuted(pushToTalk || manualCloseRequestedRef.current || !BARGE_IN_ENABLED)
      transition({ type: 'response_done' })
      telemetryRef.current.onAriaResponseDone(Date.now())
      updateStatus('speaking', 'Finishing playback...')
      waitForPlaybackDrain(() => handlePlaybackCompleteRef.current(closing))
    } catch (error) {
      failClosed(error instanceof Error ? error.message : 'Deterministic assistant speech failed.')
    }
  }, [failClosed, finalizeAssistant, pushToTalk, speechRate, transition, updateStatus, waitForPlaybackDrain])

  // Ask for exactly one closing response. This is called only while no response or local playback
  // owns the turn, so the goodbye can never overlap the sentence that came before it.
  const requestGoodbye = useCallback((): boolean => {
    if (closingRef.current) return false
    if (responseRequestedRef.current || responseActiveRef.current || drainInProgressRef.current) return false
    closingRef.current = true
    wrappingUpRef.current = true
    clearDrain()
    audioRef.current?.setCaptureMuted(true)
    transition({ type: 'close_requested' })
    transition({ type: 'response_requested', closing: true })
    responseRequestedRef.current = true
    // If manual close happened mid-utterance, discard the unfinished input so semantic VAD cannot
    // create a second, competing response after the explicit goodbye response.
    send({ type: 'input_audio_buffer.clear' })
    const goodbye = persona === 'companion'
      ? companionClosingTextForLanguage(voiceRef.current.languageKey)
      : closingTextForLanguage(voiceRef.current.languageKey)
    void playDeterministicResponse(goodbye, true)
    return true
  }, [clearDrain, persona, playDeterministicResponse, send, transition])

  const resumeListening = useCallback(() => {
    bargeInActiveRef.current = false
    setBargeInActive(false)
    updateStatus('listening', pushToTalk ? '按住说话' : 'Listening...')
    if (pushToTalk) return

    // Let the physical speaker/room tail settle after native backlog reaches zero. If a manual close
    // arrives during this guard, cancel reopening and generate the goodbye while still muted.
    clearDrain()
    drainInProgressRef.current = true
    drainTimerRef.current = setTimeout(() => {
      drainTimerRef.current = null
      drainInProgressRef.current = false
      if (manualCloseRequestedRef.current) {
        requestGoodbye()
        return
      }
      if (audioRef.current && !wrappingUpRef.current) {
        vadRef.current.reset()
        vadSpeakingRef.current = false
        audioPreRollRef.current = []
        audioRef.current.setCaptureMuted(false)
        transition({ type: 'mic_reopened' })
        armSilenceTimer()
      }
    }, 1100)
  }, [armSilenceTimer, clearDrain, pushToTalk, requestGoodbye, transition, updateStatus])

  const rejectPendingInput = useCallback((reason: 'empty' | 'echo' | 'timeout' | 'failed') => {
    clearTranscriptWait()
    telemetryRef.current.onReprompt()
    if (responseRequestedRef.current) {
      responseRequestedRef.current = false
      transition({ type: 'input_rejected' })
    }
    if (__DEV__) console.info(`[turn-taking] rejected ${reason} input before response.create`)
    if (manualCloseRequestedRef.current) {
      requestGoodbye()
      return
    }
    updateStatus('listening', reason === 'timeout' ? 'I did not catch that. Please try again.' : 'Listening...')
    resumeListening()
  }, [clearTranscriptWait, requestGoodbye, resumeListening, transition, updateStatus])

  const finishUserTurn = useCallback(() => {
    if (turnTakingRef.current.captureMuted || responseRequestedRef.current || responseActiveRef.current) return
    clearSilenceTimer()
    vadSpeakingRef.current = false
    bargeInActiveRef.current = false
    setBargeInActive(false)
    audioPreRollRef.current = []
    transition({ type: 'user_speech_stopped' })
    responseRequestedRef.current = true
    audioRef.current?.setCaptureMuted(true)
    transition({ type: 'response_requested' })
    setUserSpeaking(false)
    setRecording(false)
    updateStatus('processing', 'Thinking...')
    clearTranscriptWait()
    // Manual Qwen mode: commit requests transcription but does not create a response. The
    // transcript handler first chooses the response mode, waits for session.updated, then creates.
    send({ type: 'input_audio_buffer.commit' })
    transcriptTimerRef.current = setTimeout(() => rejectPendingInput('timeout'), 6000)
  }, [clearSilenceTimer, clearTranscriptWait, rejectPendingInput, send, transition, updateStatus])

  const interruptAssistant = useCallback(() => {
    if (manualCloseRequestedRef.current || !runtimeLeaseRef.current?.isCurrent()) return false
    const state = turnTakingRef.current
    if (!state.responseInFlight && !state.awaitingPlayback) return false

    clearDrain()
    audioRef.current?.clearPlayback()
    // Realtime generation must be cancelled as well as local playback. A locally synthesized
    // deterministic question has already finished generating, so sending response.cancel there
    // would incorrectly cancel a future provider turn.
    if (
      currentResponseSourceRef.current === 'realtime' &&
      responseActiveRef.current &&
      !responseCompletedRef.current
    ) {
      cancelledProviderResponseIdRef.current = activeProviderResponseIdRef.current
      send({ type: 'response.cancel' })
    }
    send({ type: 'input_audio_buffer.clear' })

    responseAfterSessionUpdateRef.current = null
    responseRequestedRef.current = false
    responseActiveRef.current = false
    responseCompletedRef.current = false
    responseAudioReceivedRef.current = false
    currentResponseClosingRef.current = false
    currentResponseSourceRef.current = null
    goodbyeDetectedRef.current = false
    closingRef.current = false
    wrappingUpRef.current = false
    markAssistantInterrupted()

    transition({ type: 'assistant_interrupted' })
    telemetryRef.current.onInterrupt('user_barge_in')
    bargeInActiveRef.current = true
    setBargeInActive(true)
    vadSpeakingRef.current = true
    setUserSpeaking(true)
    updateStatus('listening', 'I heard you — Aria has paused.')
    return true
  }, [clearDrain, markAssistantInterrupted, send, transition, updateStatus])

  const handleCaptureChunk = useCallback((base64Pcm16: string) => {
    if (!runtimeLeaseRef.current?.isCurrent()) return
    // Record patient audio for the session WAV. Only while the mic is open (Aria's playback mutes
    // capture), so her voice is excluded, per the doc's "record patient speech only".
    if (!turnTakingRef.current.captureMuted && sessionAudioFramesRef.current.length < MAX_AUDIO_FRAMES) {
      sessionAudioFramesRef.current.push(base64Pcm16)
    }
    if (pushToTalk) {
      if (recordingRef.current) send({ type: 'input_audio_buffer.append', audio: base64Pcm16 })
      return
    }
    if (bargeInActiveRef.current) {
      send({ type: 'input_audio_buffer.append', audio: base64Pcm16 })
      try {
        const result = bargeInVadRef.current.feed(decodeBase64Pcm16(base64Pcm16))
        if (result.event === 'speech_stopped') finishUserTurn()
      } catch (error) {
        failClosed(error instanceof Error ? error.message : 'Unable to inspect microphone audio.')
      }
      return
    }

    const state = turnTakingRef.current
    const assistantOwnsAudio = state.responseInFlight || state.awaitingPlayback
    if (assistantOwnsAudio && !manualCloseRequestedRef.current) {
      // Reject echo-driven false interruptions: ignore the mic entirely while talk-over is disabled,
      // and during the start-of-utterance grace window (loudest echo onset, where no real user would
      // interrupt). Outside the window a genuine interruption still has to clear the raised RMS floor.
      if (!BARGE_IN_ENABLED || Date.now() - playbackStartedAtRef.current < BARGE_IN_GRACE_MS) return
      try {
        const result = bargeInVadRef.current.feed(decodeBase64Pcm16(base64Pcm16))
        bargeInPreRollRef.current.push(base64Pcm16)
        if (bargeInPreRollRef.current.length > 6) bargeInPreRollRef.current.shift()
        if (result.event === 'speech_started' && interruptAssistant()) {
          for (const chunk of bargeInPreRollRef.current) {
            send({ type: 'input_audio_buffer.append', audio: chunk })
          }
          bargeInPreRollRef.current = []
        }
      } catch (error) {
        failClosed(error instanceof Error ? error.message : 'Unable to inspect microphone audio.')
      }
      return
    }
    if (state.captureMuted || responseRequestedRef.current || responseActiveRef.current) return

    let result
    try {
      result = vadRef.current.feed(decodeBase64Pcm16(base64Pcm16))
    } catch (error) {
      failClosed(error instanceof Error ? error.message : 'Unable to inspect microphone audio.')
      return
    }

    if (vadSpeakingRef.current) {
      send({ type: 'input_audio_buffer.append', audio: base64Pcm16 })
    } else {
      // Preserve the beginning of the word while two frames confirm speech; discard endless idle
      // silence instead of letting Qwen's uncommitted input buffer grow forever.
      audioPreRollRef.current.push(base64Pcm16)
      if (audioPreRollRef.current.length > 8) audioPreRollRef.current.shift()
    }

    if (result.event === 'speech_started') {
      clearSilenceTimer()
      silenceNudgedRef.current = false
      vadSpeakingRef.current = true
      for (const chunk of audioPreRollRef.current) {
        send({ type: 'input_audio_buffer.append', audio: chunk })
      }
      audioPreRollRef.current = []
      transition({ type: 'user_speech_started' })
      telemetryRef.current.onUserSpeechStart(Date.now())
      setUserSpeaking(true)
      updateStatus('listening', 'Listening...')
      if (__DEV__) console.info(`[local-vad] speech started rms=${result.rms.toFixed(4)} threshold=${result.threshold.toFixed(4)}`)
      return
    }
    if (result.event === 'speech_stopped') {
      if (__DEV__) console.info(`[local-vad] speech stopped rms=${result.rms.toFixed(4)} threshold=${result.threshold.toFixed(4)}`)
      finishUserTurn()
    }
  }, [clearSilenceTimer, failClosed, finishUserTurn, interruptAssistant, pushToTalk, send, transition, updateStatus])

  const handlePlaybackComplete = useCallback((closingResponse: boolean) => {
    if (closingResponse && !goodbyeDetectedRef.current) {
      failClosed('Closing response finished without the required goodbye sentence.')
      return
    }
    const action = choosePostPlaybackAction({
      persona,
      closingResponse,
      manualCloseRequested: manualCloseRequestedRef.current,
      spontaneousGoodbye: goodbyeDetectedRef.current,
      dailyFlowComplete: false,
    })

    if (action === 'finish') { completeConversation(); return }
    if (action === 'request_goodbye') {
      if (!requestGoodbye() && !closingRef.current) failClosed('Unable to start the closing response.')
      return
    }
    resumeListening()
  }, [completeConversation, failClosed, persona, requestGoodbye, resumeListening])
  handlePlaybackCompleteRef.current = handlePlaybackComplete

  // Fired when the user has been silent past SILENCE_MS while it was their turn. A single gentle spoken
  // nudge, then move on: screening consults the per-question flow (reprompt → skip → complete); the
  // open companion nudges once per gap and then simply waits.
  const handleSilence = useCallback(() => {
    if (!runtimeLeaseRef.current?.isCurrent()) return
    const state = turnTakingRef.current
    if (state.captureMuted || responseRequestedRef.current || responseActiveRef.current
      || drainInProgressRef.current || wrappingUpRef.current || endedRef.current) return
    const nudge = takeYourTimeForLanguage(voiceRef.current.languageKey)
    if (persona === 'screening' && checkinFlowRef.current) {
      const action = checkinFlowRef.current.recordRepromptOrTimeout()
      if (action === 'reprompt') { void playDeterministicResponse(nudge, false); return }
      if (action === 'skip') {
        const next = checkinFlowRef.current.current()
        const prompt = next ? screeningQuestionForTurn(voiceRef.current.languageKey, next.order, dailyPlan) : null
        if (prompt) void playDeterministicResponse(prompt, false)
        else requestGoodbye()
        return
      }
      requestGoodbye()
      return
    }
    if (!silenceNudgedRef.current) {
      silenceNudgedRef.current = true
      void playDeterministicResponse(nudge, false)
    }
  }, [dailyPlan, persona, playDeterministicResponse, requestGoodbye])
  onSilenceRef.current = handleSilence

  const handleMessage = useCallback(
    (payload: any) => {
      if (!runtimeLeaseRef.current?.isCurrent()) return
      const type = String(payload?.type || '')

      if (type === 'error') {
        const providerMessage = String(payload?.error?.message || '')
        if (cancelledProviderResponseIdRef.current && /cancel|active response/i.test(providerMessage)) {
          cancelledProviderResponseIdRef.current = null
          return
        }
        // Qwen delivers app-level errors as in-band frames over the OPEN socket (not socket.onerror).
        clearDrain()
        if (!hadResponseRef.current) {
          // Omni rejected the session before any response: hand off to the fallback AND tear this
          // transport down (close -> onclose -> cleanup stops the PCM bridge). Do NOT un-mute, or the
          // still-open v3 mic would capture concurrently with the v2 fallback.
          reportUnavailable('ws_error_frame')
          try { socketRef.current?.close() } catch {}
          return
        }
        failClosed(String(payload?.error?.message || 'Realtime provider error.'))
        return
      }
      if (type === 'session.created') return
      if (type === 'session.updated') {
        if (resumingRef.current) {
          resumingRef.current = false
          // Reconnected mid-conversation: resume instead of re-opening. Screening re-asks the current
          // scripted question (its flow position survived in checkinFlowRef); companion just reopens the
          // mic and continues (its provider context is gone but re-seeded via session memory above).
          if (persona === 'screening') {
            const pending = checkinFlowRef.current?.current()
            const prompt = pending
              ? screeningQuestionForTurn(voiceRef.current.languageKey, pending.order, dailyPlan)
              : null
            if (prompt) void playDeterministicResponse(prompt, false)
            else requestGoodbye()
          } else {
            resumeListening()
          }
        } else if (!openingRequestedRef.current) {
          openingRequestedRef.current = true
          if (persona === 'screening') {
            // Screening opens with the EXACT scripted warm-up: strict stage 1, no LLM drift, no
            // round-trip. Mark the session live (hadResponseRef) so a later WS drop reconnects/fails
            // rather than silently restarting under turn-based — the screening path produces no realtime
            // provider response (only ASR), so hadResponseRef would otherwise never flip.
            hadResponseRef.current = true
            void playDeterministicResponse(
              openingTextForLanguage(voiceRef.current.languageKey, dailyPlan.patientName),
              false,
            )
          } else {
            responseRequestedRef.current = true
            audioRef.current?.setCaptureMuted(true)
            transition({ type: 'response_requested' })
            send({ type: 'response.create' })
          }
        } else if (responseAfterSessionUpdateRef.current) {
          const intent = responseAfterSessionUpdateRef.current
          responseAfterSessionUpdateRef.current = null
          if (__DEV__) console.info(`[turn-taking] acknowledged ${intent} session update; creating response`)
          send({ type: 'response.create' })
        }
        return
      }
      if (type === 'conversation.item.input_audio_transcription.completed') {
        clearTranscriptWait()
        const transcript = String(payload?.transcript || '').trim()
        if (!transcript) {
          rejectPendingInput('empty')
          return
        }
        // With create_response=false, echo/blank input is rejected before a response exists. This
        // removes the race where response.cancel arrived after Qwen had already started speaking.
        if (transcript && looksLikeEchoOfAria(transcript, recentAriaRef.current)) {
          rejectPendingInput('echo')
          return
        }

        if (!responseRequestedRef.current) {
          responseRequestedRef.current = true
          audioRef.current?.setCaptureMuted(true)
          transition({ type: 'response_requested' })
        }
        setMessages((prev) => [...prev, { id: randomId('user'), role: 'user', text: transcript }])
        userTranscriptsRef.current.push(transcript)
        turnCountRef.current += 1
        // A completed turn proves the link is healthy again: refresh the reconnect budget so each
        // isolated stall gets its own allowance instead of one budget for the whole session.
        reconnectAttemptsRef.current = 0
        const patientTurnMeta = persona === 'screening' ? dailyConversationMetadataForPatientTurn(turnCountRef.current, dailyPlan) : null
        telemetryRef.current.onUserTurn(patientTurnMeta?.protocolStage ?? `turn_${turnCountRef.current}`, Date.now())
        const companionGoodbyeRequested =
          persona === 'companion' && looksLikeUserGoodbye(transcript)
        // Screening: the per-question flow decides order/completion/reprompt (correctness); the prompt
        // TEXT is still resolved via the language-adaptive generator so a mid-session language switch is
        // honoured. A too-short answer consumes a gentle reprompt (re-asks the same question) before
        // advancing, and the flow can never wedge on a required question.
        let scriptedQuestion: string | null = null
        if (persona === 'screening') {
          const flow = checkinFlowRef.current
          let answered = false
          if (flow) {
            if (flow.recordAnswer(transcript) === 'insufficient') flow.recordRepromptOrTimeout()
            else answered = true
            const nextQuestion = flow.current()
            scriptedQuestion = nextQuestion
              ? screeningQuestionForTurn(voiceRef.current.languageKey, nextQuestion.order, dailyPlan)
              : null
          } else {
            scriptedQuestion = screeningQuestionForTurn(voiceRef.current.languageKey, turnCountRef.current, dailyPlan)
          }
          // Warm the hand-off: prepend a brief neutral acknowledgement to the next scripted question so
          // the check-in feels like a friend, not a questionnaire. Only after a satisfactory answer (not
          // a reprompt), and never before the close (the dailyFlowComplete path owns the goodbye).
          if (answered && scriptedQuestion) {
            scriptedQuestion = `${acknowledgementForLanguage(voiceRef.current.languageKey, turnCountRef.current)} ${scriptedQuestion}`
          }
        }
        const dailyFlowComplete = persona === 'screening' && !scriptedQuestion

        const signal = detectLanguageSignal(transcript)
        if (signal && signal.confidence >= 0.8 && signal.languageKey !== voiceRef.current.languageKey) {
          applyVoice(voiceProfileForLanguageKey(signal.languageKey, 'transcript_reassessment'))
        }

        // The response after the final enabled stage is the fixed positive close. A manual close
        // requested while waiting for transcription also produces one goodbye, never two turns.
        if (dailyFlowComplete || companionGoodbyeRequested || manualCloseRequestedRef.current) {
          closingRef.current = true
          wrappingUpRef.current = true
          transition({ type: 'close_requested' })
          const goodbye = persona === 'companion'
            ? companionClosingTextForLanguage(voiceRef.current.languageKey)
            : closingTextForLanguage(voiceRef.current.languageKey)
          void playDeterministicResponse(goodbye, true)
          return
        }

        if (scriptedQuestion) {
          void playDeterministicResponse(scriptedQuestion, false)
        } else {
          configureNextResponse(buildLiveSessionUpdate(patientId, voiceRef.current.languageLabel, {
            voice: voiceRef.current.voice,
            languageKey: voiceRef.current.languageKey,
            persona,
            patientName: dailyPlan.patientName,
            autoCreateResponse: false,
            memory: memoryRef.current,
          }), 'normal')
        }
        return
      }
      if (type === 'conversation.item.input_audio_transcription.failed') {
        rejectPendingInput('failed')
        return
      }
      if (type === 'response.created') {
        const createdResponseId = String(payload?.response?.id || '') || null
        if (bargeInActiveRef.current) {
          cancelledProviderResponseIdRef.current = createdResponseId
          send({ type: 'response.cancel' })
          return
        }
        hadResponseRef.current = true
        if (drainInProgressRef.current) {
          transition({ type: 'response_created' })
          failClosed('A new response started before prior playback completed.')
          return
        }
        responseRequestedRef.current = false
        responseActiveRef.current = true
        responseCompletedRef.current = false
        responseAudioReceivedRef.current = false
        currentResponseSourceRef.current = 'realtime'
        activeProviderResponseIdRef.current = createdResponseId
        currentResponseClosingRef.current = closingRef.current
        goodbyeDetectedRef.current = false
        audioRef.current?.setCaptureMuted(true)
        transition({ type: 'response_created' })
        telemetryRef.current.onAriaResponseCreated(Date.now())
        updateStatus('speaking', 'Speaking...')
        return
      }
      if (type === 'response.audio.delta') {
        if (isCancelledProviderEvent(payload, cancelledProviderResponseIdRef.current)) return
        if (!responseAudioReceivedRef.current) {
          responseAudioReceivedRef.current = true
          transition({ type: 'audio_delta' })
          telemetryRef.current.onAriaFirstAudio(Date.now())
          playbackStartedAtRef.current = Date.now()
          enqueuedPlaybackMsRef.current = 0
          if (!pushToTalk && !manualCloseRequestedRef.current && BARGE_IN_ENABLED) {
            bargeInVadRef.current.reset()
            bargeInPreRollRef.current = []
            audioRef.current?.setCaptureMuted(false)
          }
        }
        try {
          const delta = String(payload?.delta || '')
          enqueuedPlaybackMsRef.current += pcm24kBase64DurationMs(delta)
          audioRef.current?.play(delta)
        } catch (error) {
          failClosed(error instanceof Error ? error.message : 'Native audio playback failed.')
        }
        return
      }
      if (type === 'response.audio.done') {
        if (isCancelledProviderEvent(payload, cancelledProviderResponseIdRef.current)) return
        // Qwen has finished generating audio, but the native speaker may still have a backlog.
        return
      }
      if (type === 'response.audio_transcript.delta' || type === 'response.output_audio_transcript.delta') {
        if (isCancelledProviderEvent(payload, cancelledProviderResponseIdRef.current)) return
        assistantTextRef.current += String(payload?.delta || '')
        appendAssistantStreaming(assistantTextRef.current)
        return
      }
      if (type === 'response.audio_transcript.done' || type === 'response.output_audio_transcript.done') {
        if (isCancelledProviderEvent(payload, cancelledProviderResponseIdRef.current)) return
        const finalText = String(payload?.transcript ?? assistantTextRef.current)
        finalizeAssistant(finalText)
        goodbyeDetectedRef.current = looksLikeGoodbye(finalText)
        return
      }
      if (type === 'response.done') {
        if (isCancelledProviderEvent(payload, cancelledProviderResponseIdRef.current)) {
          cancelledProviderResponseIdRef.current = null
          activeProviderResponseIdRef.current = null
          currentResponseSourceRef.current = null
          return
        }
        if (responseCompletedRef.current) {
          transition({ type: 'response_done' })
          return
        }
        responseCompletedRef.current = true
        responseRequestedRef.current = false
        responseActiveRef.current = false
        audioRef.current?.setCaptureMuted(
          pushToTalk || manualCloseRequestedRef.current || !responseAudioReceivedRef.current,
        )
        transition({ type: 'response_done' })
        telemetryRef.current.onAriaResponseDone(Date.now())
        updateStatus('speaking', 'Finishing playback...')
        const closingResponse = currentResponseClosingRef.current
        waitForPlaybackDrain(() => handlePlaybackComplete(closingResponse))
        return
      }
    },
    [appendAssistantStreaming, applyVoice, clearDrain, clearTranscriptWait, configureNextResponse, dailyPlan, failClosed, finalizeAssistant, handlePlaybackComplete, patientId, persona, playDeterministicResponse, pushToTalk, rejectPendingInput, reportUnavailable, requestGoodbye, resumeListening, send, transition, updateStatus, waitForPlaybackDrain],
  )

  const cleanup = useCallback(() => {
    const lease = runtimeLeaseRef.current
    runtimeLeaseRef.current = null
    lease?.release()
    startingRef.current = false
    startAttemptRef.current += 1
    if (drainTimerRef.current) { clearTimeout(drainTimerRef.current); drainTimerRef.current = null }
    if (transcriptTimerRef.current) { clearTimeout(transcriptTimerRef.current); transcriptTimerRef.current = null }
    if (wrapupTimerRef.current) { clearTimeout(wrapupTimerRef.current); wrapupTimerRef.current = null }
    if (connectTimerRef.current) { clearTimeout(connectTimerRef.current); connectTimerRef.current = null }
    if (silenceTimerRef.current) { clearTimeout(silenceTimerRef.current); silenceTimerRef.current = null }
    silenceNudgedRef.current = false
    drainInProgressRef.current = false
    wrappingUpRef.current = false
    vadRef.current.reset()
    bargeInVadRef.current.reset()
    vadSpeakingRef.current = false
    audioPreRollRef.current = []
    bargeInPreRollRef.current = []
    bargeInActiveRef.current = false
    recordingRef.current = false
    pushToTalkGestureRef.current.reset()
    void audioRef.current?.stop()
    audioRef.current = null
    try { socketRef.current?.close() } catch {}
    socketRef.current = null
    try { phase1InjectorRef.current?.close() } catch {}
    phase1InjectorRef.current = null
    phase1InjectionEpochRef.current += 1
    phase1InjectionChainRef.current = Promise.resolve()
    phase1StopAtRef.current = null
    openingRequestedRef.current = false
    resumingRef.current = false
    responseAfterSessionUpdateRef.current = null
    responseRequestedRef.current = false
    responseActiveRef.current = false
    responseCompletedRef.current = false
    responseAudioReceivedRef.current = false
    activeProviderResponseIdRef.current = null
    cancelledProviderResponseIdRef.current = null
    currentResponseSourceRef.current = null
    currentResponseClosingRef.current = false
    streamIdRef.current = null
    assistantTextRef.current = ''
    userTranscriptsRef.current = []
    setSessionActive(false)
    setConnecting(false)
    setUserSpeaking(false)
    setBargeInActive(false)
    setRecording(false)
    resolveStop()
  }, [resolveStop])

  // Bounded auto-reconnect for a transient realtime drop on the flaky cross-border link. Pre-first-turn
  // it restarts fresh; mid-conversation it RESUMES (see startConversation's `resume` path + the
  // resumingRef branch in session.updated). Refused once ended/closing/manually-closed or the per-gap
  // budget is spent. Returns true when a reconnect was scheduled (caller must then stop/return).
  const tryReconnect = useCallback((reason: string): boolean => {
    if (endedRef.current || manualCloseRequestedRef.current || closingRef.current) return false
    if (reconnectAttemptsRef.current >= MAX_RECONNECTS_PER_GAP) return false
    reconnectAttemptsRef.current += 1
    // Mid-conversation (a turn already happened) → RESUME the session; pre-first-turn → fresh restart.
    const resume = turnCountRef.current > 0
    telemetryRef.current.onReconnect()
    console.warn(`[realtime] reconnect ${reconnectAttemptsRef.current}/${MAX_RECONNECTS_PER_GAP} (${resume ? 'resume' : 'restart'}, turns=${turnCountRef.current}) after: ${reason}`)
    updateStatus('processing', 'Reconnecting…')
    // cleanup() wipes userTranscriptsRef; capture it first so a resumed session can re-seed companion
    // continuity via session memory (screening resumes purely from client-side flow state).
    const priorTranscripts = userTranscriptsRef.current.slice()
    cleanup()
    if (resume) {
      userTranscriptsRef.current = priorTranscripts
      memoryRef.current = priorTranscripts.slice(-4)
    }
    isReconnectRef.current = true
    void startConversationRef.current({ resume })
    return true
  }, [cleanup, updateStatus])

  const startConversation = useCallback(async (opts?: { resume?: boolean }) => {
    if (Platform.OS === 'web') {
      updateStatus('error', 'Direct WS (v3) is native-only — web uses the relay (v1). Set mode=relay on web.')
      return
    }
    const resume = Boolean(opts?.resume)
    if (socketRef.current || startingRef.current) return
    const runtimeLease = acquireConversationRuntime(instanceIdRef.current, cleanup)
    runtimeLeaseRef.current = runtimeLease
    startingRef.current = true
    const startAttempt = startAttemptRef.current + 1
    startAttemptRef.current = startAttempt
    setConnecting(true)
    updateStatus('processing', 'Connecting...')
    turnTakingRef.current = createTurnTakingState()
    setTurnState('idle')
    transition({ type: 'connect_started' })
    setEnded(false)
    endedRef.current = false
    endReasonRef.current = null
    // Preserve the reconnect counter across a reconnect-driven restart; zero it only for a fresh start.
    if (!isReconnectRef.current) reconnectAttemptsRef.current = 0
    isReconnectRef.current = false
    // A mid-conversation RESUME keeps everything said so far — transcript, flow position, turn count,
    // captured audio, telemetry, echo history, and the (possibly language-switched) voice. Only a fresh
    // start wipes them. (userTranscriptsRef was wiped by cleanup() then restored in tryReconnect.)
    if (!resume) {
      setMessages([])
      voiceRef.current = voiceProfileForSession(language)
      turnCountRef.current = 0
      telemetryRef.current.reset()
      checkinFlowRef.current = persona === 'screening' ? createSessionCheckinFlow(voiceRef.current.languageKey, dailyPlan) : null
      sessionAudioFramesRef.current = []
      sessionStartedAtMsRef.current = Date.now()
      recentAriaRef.current = []
      userTranscriptsRef.current = []
    }
    // Resume skips the opening (openingRequestedRef=true), flags session.updated to pick up where we left
    // off (resumingRef), and marks the session live (hadResponseRef) so a re-drop reconnects/fails rather
    // than silently falling back to turn-based.
    openingRequestedRef.current = resume
    resumingRef.current = resume
    hadResponseRef.current = resume
    responseAfterSessionUpdateRef.current = null
    openedRef.current = false
    reportedUnavailableRef.current = false
    silenceNudgedRef.current = false
    endedAtMsRef.current = 0
    closingRef.current = false
    manualCloseRequestedRef.current = false
    responseRequestedRef.current = false
    responseActiveRef.current = false
    responseCompletedRef.current = false
    responseAudioReceivedRef.current = false
    activeProviderResponseIdRef.current = null
    cancelledProviderResponseIdRef.current = null
    currentResponseSourceRef.current = null
    currentResponseClosingRef.current = false
    goodbyeDetectedRef.current = false
    drainInProgressRef.current = false
    vadRef.current.reset()
    bargeInVadRef.current.reset()
    vadSpeakingRef.current = false
    audioPreRollRef.current = []
    bargeInPreRollRef.current = []
    bargeInActiveRef.current = false
    recordingRef.current = false
    pushToTalkGestureRef.current.reset()
    setRecording(false)
    setBargeInActive(false)

    try {
      // Short-lived server-minted token (secure) or, if explicitly enabled, the kiosk client key.
      const bearer = await getBearer()
      // Load prior-session memory for soft continuity (best-effort; empty on the first ever session). On
      // a resume, memoryRef was already seeded from the live transcript in tryReconnect — don't clobber it.
      if (!resume) {
        try { memoryRef.current = await getStoredSessionMemory(patientId) } catch { memoryRef.current = [] }
      }
      // The hook may have been cleaned up or superseded while the token request was in flight.
      if (!runtimeLease.isCurrent() || !startingRef.current || startAttemptRef.current !== startAttempt) return

      // RN WebSocket supports a 3rd `options.headers` arg (not in DOM types → cast).
      const socket = new (WebSocket as any)(realtimeWsUrl(), undefined, {
        headers: { Authorization: `Bearer ${bearer}` },
      }) as WebSocket
      socketRef.current = socket
      startingRef.current = false
      // Connection watchdog: if the socket never opens within 7s (dead region/network), fall back
      // to the turn-based stack instead of hanging in "Connecting..." forever.
      if (connectTimerRef.current) clearTimeout(connectTimerRef.current)
      connectTimerRef.current = setTimeout(() => {
        connectTimerRef.current = null
        if (!openedRef.current) {
          // A resume pre-sets hadResponseRef=true, so the old `&& !hadResponseRef` gate would never fire
          // here and a dead resume socket could hang; we also can't swap to turn-based mid-conversation.
          // Fail closed on a dead resume socket (turns>0 → finalize saves the partial); a fresh start
          // still falls back to turn-based.
          if (resume) failClosed('Realtime reconnect timed out before opening.')
          else { reportUnavailable('ws_connect_timeout'); cleanup() }
        }
      }, 7000)

      socket.onopen = () => {
        if (!runtimeLease.isCurrent()) { try { socket.close() } catch {}; return }
        openedRef.current = true
        if (connectTimerRef.current) { clearTimeout(connectTimerRef.current); connectTimerRef.current = null }
        setSessionActive(true)
        setConnecting(false)
        transition({ type: 'session_configuring' })
        updateStatus('processing', 'Starting Aria...')
        send(buildLiveSessionUpdate(patientId, voiceRef.current.languageLabel, {
          voice: voiceRef.current.voice,
          languageKey: voiceRef.current.languageKey,
          persona,
          patientName: dailyPlan.patientName,
          autoCreateResponse: false,
          memory: memoryRef.current,
        }))
        // Start native PCM capture -> stream frames upstream.
        const audio = createPcmAudioBridge()
        audioRef.current = audio
        // Capture stays muted through the opening response and its local playback drain.
        audio.setCaptureMuted(true)
        audio
          .start(handleCaptureChunk)
          .then(() => audio.setCaptureMuted(true))
          .catch((e: unknown) => {
            // Branch on a real turn, not hadResponseRef: the deterministic screening opening flips
            // hadResponseRef at session.updated with no round-trip, so a pre-first-turn capture failure
            // must still report-unavailable (fall back), not abandon.
            if (turnCountRef.current === 0) { reportUnavailable('audio_start_failed'); cleanup() }
            else failClosed(e instanceof Error ? e.message : 'Audio start failed.')
          })

        // Local MuMu acceptance can inject silent, pre-recorded PCM frames into the exact same
        // JS VAD/Qwen pipeline. This endpoint is compiled out unless explicitly configured in a
        // development bundle; it is never enabled in production or by default.
        const injectorUrl = __DEV__ ? process.env.EXPO_PUBLIC_PHASE1_AUDIO_INJECTOR_URL?.trim() : ''
        if (injectorUrl) {
          const injector = new WebSocket(injectorUrl)
          phase1InjectorRef.current = injector
          injector.onmessage = (event) => {
            const raw = String(event.data || '')
            try {
              const command = JSON.parse(raw) as { type?: string; frames?: unknown; stopAt?: unknown }
              if (command.type === 'phase1.pcm' && Array.isArray(command.frames)) {
                const frames = command.frames.filter((frame): frame is string => typeof frame === 'string')
                const allowedStopPhases: TurnTakingPhase[] = [
                  'user_speaking', 'assistant_generating', 'assistant_playing', 'playback_guard',
                ]
                phase1StopAtRef.current = allowedStopPhases.includes(command.stopAt as TurnTakingPhase)
                  ? command.stopAt as TurnTakingPhase
                  : null
                const injectionEpoch = phase1InjectionEpochRef.current
                phase1InjectionChainRef.current = phase1InjectionChainRef.current.then(async () => {
                  // Avoid interleaving the simulator's native 100 ms silence frames with injected
                  // 100 ms speech frames; real capture has only one source. The canonical state
                  // remains listening, so the injected frames still traverse the production VAD.
                  audioRef.current?.setCaptureMuted(true)
                  console.info(
                    `[phase1-injector] replaying ${frames.length} frames ` +
                      `phase=${turnTakingRef.current.phase} muted=${turnTakingRef.current.captureMuted} ` +
                      `requested=${responseRequestedRef.current} active=${responseActiveRef.current} ptt=${pushToTalk}`,
                  )
                  for (const frame of frames) {
                    if (phase1InjectionEpochRef.current !== injectionEpoch) return
                    handleCaptureChunk(frame)
                    await new Promise<void>((resolve) => setTimeout(resolve, 100))
                  }
                  console.info('[phase1-injector] replay complete')
                  if (
                    turnTakingRef.current.phase === 'listening' &&
                    !responseRequestedRef.current &&
                    !responseActiveRef.current
                  ) {
                    audioRef.current?.setCaptureMuted(false)
                  }
                })
                return
              }
            } catch {
              // Backward-compatible single-frame mode for local debugging.
            }
            handleCaptureChunk(raw)
          }
          injector.onopen = () => console.info('[phase1-injector] connected')
          injector.onerror = () => console.warn('[phase1-injector] unavailable')
        }
      }
      socket.onmessage = (event) => {
        try { handleMessage(JSON.parse(String(event.data))) } catch { /* ignore malformed */ }
      }
      socket.onerror = () => {
        // Ignore events from a socket this hook has already superseded — e.g. a reconnect (tryReconnect
        // → cleanup → new startConversation) released this lease and bumped startAttempt. Without this,
        // a dead socket's error/close would run against the SUCCESSOR session's refs and kill it.
        if (!runtimeLease.isCurrent() || startAttemptRef.current !== startAttempt) return
        // Before any response: omni is unreachable -> let the supervisor fall back (onclose cleans up).
        if (!hadResponseRef.current) { reportUnavailable('ws_error'); cleanup(); return }
        if (tryReconnect('ws_error')) return
        failClosed('Realtime connection error.')
      }
      socket.onclose = () => {
        // Superseded socket (see onerror): a reconnect already replaced this session — do not touch it.
        if (!runtimeLease.isCurrent() || startAttemptRef.current !== startAttempt) return
        // Closed before it ever opened (region block / handshake reject / connect fail): fall back.
        if (!openedRef.current && !hadResponseRef.current) {
          reportUnavailable('ws_closed_before_open')
        } else if (hadResponseRef.current && !endedRef.current) {
          // Dropped mid-opening on the flaky cross-border link: try a bounded reconnect before giving
          // up. tryReconnect runs its own cleanup + restart, so return without the teardown below.
          if (tryReconnect('ws_closed_before_turn')) return
          // A live provider close is not proof that queued audio played. Fail closed and preserve the
          // captured session instead of reopening capture or pretending the goodbye completed.
          failClosed('Realtime connection closed before the turn completed.')
        } else if (openedRef.current && !endedRef.current) {
          updateStatus('idle', 'Conversation ended')
        }
        cleanup()
      }
    } catch (e) {
      cleanup()
      if (!hadResponseRef.current) { reportUnavailable('ws_start_failed'); return }
      failClosed(`Error: ${e instanceof Error ? e.message : 'unknown'}`)
    }
  }, [cleanup, dailyPlan.patientName, failClosed, handleCaptureChunk, handleMessage, language, patientId, persona, reportUnavailable, send, transition, tryReconnect, updateStatus])
  // Let tryReconnect re-invoke the latest startConversation without a hook-ordering cycle.
  startConversationRef.current = startConversation

  const stopConversation = useCallback(async () => {
    // Auto-close already observed the complete goodbye playback; teardown is now safe.
    if (endedRef.current) { cleanup(); return }
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) { cleanup(); return }

    let stopPromise = stopPromiseRef.current
    if (!stopPromise) {
      stopPromise = new Promise<void>((resolve) => { stopResolveRef.current = resolve })
      stopPromiseRef.current = stopPromise
    }

    manualCloseRequestedRef.current = true
    wrappingUpRef.current = true
    audioRef.current?.setCaptureMuted(true)
    transition({ type: 'close_requested' })
    updateStatus('processing', 'Wrapping up. Aria will say goodbye in a moment...')

    // If nothing owns the turn, start the goodbye now. If generation/playback is active, its
    // post-playback policy will start the goodbye exactly once.
    if (
      !closingRef.current &&
      !responseRequestedRef.current &&
      !responseActiveRef.current &&
      !turnTakingRef.current.awaitingPlayback
    ) {
      clearDrain()
      requestGoodbye()
    }

    // Safety bound only. Normal completion is driven by closing response.done + native drain.
    if (wrapupTimerRef.current) clearTimeout(wrapupTimerRef.current)
    wrapupTimerRef.current = setTimeout(() => {
      wrapupTimerRef.current = null
      failClosed('Graceful close timed out; session ended with capture muted.')
    }, 30_000)

    await stopPromise
    cleanup()
  }, [cleanup, clearDrain, failClosed, requestGoodbye, transition, updateStatus])
  phase1StopConversationRef.current = () => { void stopConversation() }

  const beginPushToTalk = useCallback(() => {
    if (
      !pushToTalk || wrappingUpRef.current || !runtimeLeaseRef.current?.isCurrent() ||
      socketRef.current?.readyState !== WebSocket.OPEN || !audioRef.current ||
      responseRequestedRef.current || responseActiveRef.current || drainInProgressRef.current
    ) return
    const token = pushToTalkGestureRef.current.begin()
    if (token === null || !pushToTalkGestureRef.current.ready(token)) return

    // Each physical hold owns a fresh provider buffer. This also prevents a cancelled/empty prior
    // gesture from being committed with the next utterance.
    send({ type: 'input_audio_buffer.clear' })
    recordingRef.current = true
    setRecording(true)
    vadRef.current.reset()
    vadSpeakingRef.current = true
    audioPreRollRef.current = []
    audioRef.current.setCaptureMuted(false)
    if (turnTakingRef.current.captureMuted) transition({ type: 'mic_reopened' })
    transition({ type: 'user_speech_started' })
    telemetryRef.current.onUserSpeechStart(Date.now())
    setUserSpeaking(true)
    updateStatus('listening', 'Listening… release to send')
  }, [pushToTalk, send, transition, updateStatus])

  const endPushToTalk = useCallback(() => {
    if (!pushToTalk) return
    const release = pushToTalkGestureRef.current.release()
    if (release !== 'send' || !recordingRef.current) return
    recordingRef.current = false
    finishUserTurn()
  }, [finishUserTurn, pushToTalk])

  // Retained for automated diagnostics and callers that cannot expose press-in/press-out events.
  const toggleRecording = useCallback(() => {
    if (recordingRef.current) endPushToTalk()
    else beginPushToTalk()
  }, [beginPushToTalk, endPushToTalk])

  useEffect(() => cleanup, [cleanup])

  return {
    mode: 'ws',
    statusKind,
    statusText,
    messages,
    startConversation,
    stopConversation,
    connecting,
    sessionActive,
    userSpeaking,
    bargeInActive,
    turnState,
    ended,
    endReason: endReasonRef.current ?? undefined,
    recording: pushToTalk ? recording : undefined,
    beginPushToTalk: pushToTalk ? beginPushToTalk : undefined,
    endPushToTalk: pushToTalk ? endPushToTalk : undefined,
    toggleRecording: pushToTalk ? toggleRecording : undefined,
    getSessionTelemetry: () => (sessionStartedAtMsRef.current
      ? telemetryRef.current.snapshot(sessionStartedAtMsRef.current, endedAtMsRef.current || Date.now())
      : null),
    getSessionAudio: () => (sessionAudioFramesRef.current.length
      ? { base64Frames: sessionAudioFramesRef.current, sampleRate: CAPTURE_SAMPLE_RATE }
      : null),
  }
}
