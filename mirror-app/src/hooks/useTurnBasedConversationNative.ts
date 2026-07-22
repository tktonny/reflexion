import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AudioModule,
  RecordingPresets,
  createAudioPlayer,
  setAudioModeAsync,
  useAudioRecorder,
} from 'expo-audio'
import * as FileSystem from 'expo-file-system/legacy'

import {
  RECALL_DEADLINE_TURN,
  RECALL_DIRECTIVE,
  buildLiveInstructions,
  looksLikeRecallProbe,
  looksLikeUserGoodbye,
  openingMessageForLanguage,
  recallBudgetStep,
} from '../orchestration/orchestrator'
import {
  closingTextForLanguage,
  companionClosingTextForLanguage,
  screeningQuestionForTurn,
} from '../orchestration/deterministicSpeech'
import {
  acquireConversationRuntime,
  type ConversationRuntimeLease,
} from '../orchestration/conversationRuntime'
import {
  playAndWaitForCompletion,
  type PlaybackCompletionPlayer,
} from '../orchestration/playbackCompletion'
import { createPushToTalkGesture } from '../orchestration/pushToTalkGesture'
import { detectLanguageSignal, voiceProfileForLanguageKey, voiceProfileForSession, type VoiceProfile } from '../orchestration/voice'
import { qwenASR, qwenChat, qwenTTS, type QwenChatMessage } from '../api/qwenClient'
import { randomId } from '../utils/id'
import type { ChatMessage, ConversationApi, StatusKind } from './conversationTypes'

type Options = { patientId?: string; language?: string; persona?: 'screening' | 'companion' }

/**
 * Version 2 (Flavor B) — NATIVE build. Turn-based voice loop using expo-audio:
 * record a clip to a file -> base64 -> Qwen ASR -> chat (on-device orchestration) -> TTS -> play.
 * Selector routes 'http' + native here; web keeps useTurnBasedConversation (Web Audio).
 *
 * NOTE: needs on-device verification (built via EAS dev/preview build; Expo Go cannot record).
 * If ASR rejects the m4a clip, adjust the format hint / recording preset.
 */
export function useTurnBasedConversationNative(options: Options = {}): ConversationApi {
  const patientId = options.patientId ?? 'demo-patient'
  const language = options.language ?? 'en'
  const persona = options.persona ?? 'screening'

  const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY)

  const [statusKind, setStatusKind] = useState<StatusKind>('idle')
  const [statusText, setStatusText] = useState('Ready to start')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [connecting, setConnecting] = useState(false)
  const [sessionActive, setSessionActive] = useState(false)
  const [recording, setRecording] = useState(false)
  const [ended, setEnded] = useState(false)

  const llmRef = useRef<QwenChatMessage[]>([])
  const recordingRef = useRef(false) // mirrors `recording` so cleanup needn't depend on the state
  const recorderRef = useRef(recorder)
  recorderRef.current = recorder
  const voiceRef = useRef<VoiceProfile>(voiceProfileForSession(language))
  const playerRef = useRef<ReturnType<typeof createAudioPlayer> | null>(null)
  const playbackAbortRef = useRef<AbortController | null>(null)
  const runtimeLeaseRef = useRef<ConversationRuntimeLease | null>(null)
  const lifecycleEpochRef = useRef(0)
  const cleanupRef = useRef<() => void>(() => {})
  const instanceIdRef = useRef(randomId('turnbased'))
  const startingRef = useRef(false)
  const sessionActiveRef = useRef(false)
  const turnCountRef = useRef(0)
  const recallProbeIssuedRef = useRef(false)
  const recallAnsweredRef = useRef(false)
  const closingRef = useRef(false)
  const endedRef = useRef(false)
  const manualCloseRequestedRef = useRef(false)
  const speechPromiseRef = useRef<Promise<boolean> | null>(null)
  const turnPromiseRef = useRef<Promise<void> | null>(null)
  const stopPromiseRef = useRef<Promise<void> | null>(null)
  // Ignore accidental taps and press/release noise; a real spoken response comfortably exceeds
  // this duration, including for short words such as "yes" or "no".
  const pushToTalkGestureRef = useRef(createPushToTalkGesture({ minimumRecordingMs: 250 }))
  const recorderPreparePromiseRef = useRef<Promise<void> | null>(null)
  const recorderPreparedRef = useRef(false)

  const updateStatus = useCallback((kind: StatusKind, text: string) => {
    setStatusKind(kind)
    setStatusText(text)
  }, [])

  const addMessage = useCallback((role: ChatMessage['role'], text: string) => {
    setMessages((prev) => [...prev, { id: randomId(role), role, text }])
  }, [])

  const isCurrentSession = useCallback((epoch: number) => (
    lifecycleEpochRef.current === epoch && runtimeLeaseRef.current?.isCurrent() === true
  ), [])

  const play = useCallback(async (
    tts: { audioBase64: string | null; url: string | null },
    epoch: number,
  ): Promise<boolean> => {
    if (!isCurrentSession(epoch)) return false
    const source = tts.url || (tts.audioBase64 ? `data:audio/wav;base64,${tts.audioBase64}` : null)
    if (!source) throw new Error('TTS returned no playable audio.')
    playerRef.current?.remove()
    playbackAbortRef.current?.abort()
    const abortController = new AbortController()
    playbackAbortRef.current = abortController
    const player = createAudioPlayer(source, { updateInterval: 100 })
    playerRef.current = player
    try {
      // expo-audio's SharedObject exposes addListener at runtime, although AudioPlayer's generated
      // declaration currently omits the inherited method.
      await playAndWaitForCompletion(
        player as unknown as PlaybackCompletionPlayer,
        45_000,
        abortController.signal,
      )
      return isCurrentSession(epoch)
    } finally {
      if (playerRef.current === player) playerRef.current = null
      if (playbackAbortRef.current === abortController) playbackAbortRef.current = null
      try { player.remove() } catch {}
    }
  }, [isCurrentSession])

  const speak = useCallback((
    text: string,
    status = 'Speaking...',
    epoch = lifecycleEpochRef.current,
  ): Promise<boolean> => {
    const task = (async () => {
      if (!isCurrentSession(epoch)) return false
      updateStatus('speaking', status)
      try {
        const tts = await qwenTTS(text, { voice: voiceRef.current.voice })
        if (!isCurrentSession(epoch)) return false
        return await play(tts, epoch)
      } catch (error) {
        if (!isCurrentSession(epoch)) return false
        updateStatus('error', `TTS failed: ${error instanceof Error ? error.message : 'unknown'}`)
        throw error
      }
    })()
    speechPromiseRef.current = task
    void task.finally(() => {
      if (speechPromiseRef.current === task) speechPromiseRef.current = null
    }).catch(() => {})
    return task
  }, [isCurrentSession, play, updateStatus])

  const startConversation = useCallback(async () => {
    if (startingRef.current || sessionActiveRef.current || stopPromiseRef.current) return
    const epoch = lifecycleEpochRef.current + 1
    lifecycleEpochRef.current = epoch
    const runtimeLease = acquireConversationRuntime(instanceIdRef.current, () => cleanupRef.current())
    runtimeLeaseRef.current = runtimeLease
    startingRef.current = true
    setConnecting(true)
    updateStatus('processing', 'Starting...')
    setMessages([])
    voiceRef.current = voiceProfileForSession(language)
    llmRef.current = [{ role: 'system', content: buildLiveInstructions(patientId, language, { persona }) }]
    turnCountRef.current = 0
    recallProbeIssuedRef.current = false
    recallAnsweredRef.current = false
    closingRef.current = false
    endedRef.current = false
    manualCloseRequestedRef.current = false
    stopPromiseRef.current = null
    setEnded(false)
    try {
      const { granted } = await AudioModule.requestRecordingPermissionsAsync()
      if (!isCurrentSession(epoch)) return
      if (!granted) throw new Error('microphone permission denied')
      await setAudioModeAsync({ playsInSilentMode: true, allowsRecording: true })
      if (!isCurrentSession(epoch)) return
      sessionActiveRef.current = true
      setSessionActive(true)
      setConnecting(false)
      const opening = openingMessageForLanguage(language)
      llmRef.current.push({ role: 'assistant', content: opening })
      addMessage('assistant', opening)
      const spoken = await speak(opening, 'Speaking...', epoch)
      if (spoken && isCurrentSession(epoch) && !manualCloseRequestedRef.current) {
        updateStatus('listening', '点麦克风开始回答')
      }
    } catch (e) {
      if (!isCurrentSession(epoch)) return
      cleanup()
      updateStatus('error', `Error: ${e instanceof Error ? e.message : 'unknown'}`)
    } finally {
      if (lifecycleEpochRef.current === epoch) startingRef.current = false
    }
  }, [addMessage, isCurrentSession, language, patientId, speak, updateStatus])

  const runTurn = useCallback(async () => {
    const epoch = lifecycleEpochRef.current
    if (!isCurrentSession(epoch) || manualCloseRequestedRef.current) return
    updateStatus('processing', 'Transcribing...')
    try {
      recorderPreparedRef.current = false
      await recorder.stop()
      if (!isCurrentSession(epoch)) return
      const uri = recorder.uri
      if (!uri) { updateStatus('listening', '点麦克风开始回答'); return }
      const base64 = await FileSystem.readAsStringAsync(uri, { encoding: 'base64' as FileSystem.EncodingType })
      if (!isCurrentSession(epoch)) return
      const transcript = await qwenASR(base64, { format: 'm4a' })
      if (!isCurrentSession(epoch)) return
      if (!transcript) { updateStatus('listening', '没听清,再试一次'); return }
      addMessage('user', transcript)
      llmRef.current.push({ role: 'user', content: transcript })
      turnCountRef.current += 1

      const signal = detectLanguageSignal(transcript)
      if (signal && signal.confidence >= 0.8 && signal.languageKey !== voiceRef.current.languageKey) {
        voiceRef.current = voiceProfileForLanguageKey(signal.languageKey, 'transcript_reassessment')
      }

      // Recall floor + forced wrap-up is SCREENING-only. Companion is open chat with no agenda.
      let directive: string | null = null
      let scriptedQuestion: string | null = null
      const companionGoodbyeRequested = persona === 'companion' && looksLikeUserGoodbye(transcript)
      if (persona === 'screening') {
        const step = recallBudgetStep({
          turnCount: turnCountRef.current,
          recallProbeIssued: recallProbeIssuedRef.current,
          recallAnswered: recallAnsweredRef.current,
        })
        if (step.action === 'force_recall') { recallProbeIssuedRef.current = true; directive = RECALL_DIRECTIVE }
        else if (step.action === 'wrap_up') { recallAnsweredRef.current = true; closingRef.current = true }
        else scriptedQuestion = screeningQuestionForTurn(voiceRef.current.languageKey, turnCountRef.current)
      }
      if (companionGoodbyeRequested) closingRef.current = true

      updateStatus('processing', 'Thinking...')
      const msgs = directive ? [...llmRef.current, { role: 'system' as const, content: directive }] : llmRef.current
      // The screening close is deterministic. Relying on a one-turn LLM directive occasionally
      // produced another question or no goodbye at all on the fallback transport.
      const reply = closingRef.current
        ? persona === 'companion'
          ? companionClosingTextForLanguage(voiceRef.current.languageKey)
          : closingTextForLanguage(voiceRef.current.languageKey)
        : scriptedQuestion ?? await qwenChat(msgs, { maxTokens: 120, temperature: 0.4 })
      if (!isCurrentSession(epoch)) return
      if (reply) {
        llmRef.current.push({ role: 'assistant', content: reply })
        addMessage('assistant', reply)
        if (
          !recallProbeIssuedRef.current &&
          turnCountRef.current >= RECALL_DEADLINE_TURN &&
          looksLikeRecallProbe(reply)
        ) recallProbeIssuedRef.current = true
        const spoken = await speak(reply, 'Speaking...', epoch)
        if (!spoken || !isCurrentSession(epoch)) return
      }
      // Companion ends only from explicit patient intent or manual End Chat. Polite phrases in an
      // otherwise normal assistant answer are never allowed to close it.
      if (closingRef.current) {
        endedRef.current = true
        sessionActiveRef.current = false
        setSessionActive(false)
        updateStatus('idle', 'Goodbye complete. Saving the conversation...')
        setEnded(true) // hands-free auto-finalize (parity with v3; needed for the v3->v2 fallback path)
      } else if (!manualCloseRequestedRef.current) {
        updateStatus('listening', '点麦克风开始回答')
      }
    } catch (e) {
      if (!isCurrentSession(epoch)) return
      updateStatus('error', `Error: ${e instanceof Error ? e.message : 'unknown'}`)
    }
  }, [addMessage, isCurrentSession, persona, recorder, speak, updateStatus])

  const beginPushToTalk = useCallback(() => {
    if (
      !sessionActiveRef.current || manualCloseRequestedRef.current ||
      turnPromiseRef.current || speechPromiseRef.current
    ) return
    const token = pushToTalkGestureRef.current.begin()
    if (token === null) return
    const epoch = lifecycleEpochRef.current

    void (async () => {
      try {
        if (!recorderPreparedRef.current) {
          let preparing = recorderPreparePromiseRef.current
          if (!preparing) {
            preparing = recorder.prepareToRecordAsync()
            recorderPreparePromiseRef.current = preparing
          }
          try {
            await preparing
            if (!isCurrentSession(epoch)) return
            recorderPreparedRef.current = true
          } finally {
            if (recorderPreparePromiseRef.current === preparing) recorderPreparePromiseRef.current = null
          }
        }
        if (
          !isCurrentSession(epoch) || manualCloseRequestedRef.current || !sessionActiveRef.current ||
          !pushToTalkGestureRef.current.ready(token)
        ) return
        recorder.record()
        recordingRef.current = true
        setRecording(true)
        updateStatus('listening', '正在听… 松开发送')
      } catch (error) {
        pushToTalkGestureRef.current.cancel(token)
        if (!isCurrentSession(epoch)) return
        updateStatus('error', `无法开始录音: ${error instanceof Error ? error.message : 'unknown'}`)
      }
    })()
  }, [isCurrentSession, recorder, updateStatus])

  const endPushToTalk = useCallback(() => {
    const release = pushToTalkGestureRef.current.release()
    if (release === 'cancelled') {
      if (sessionActiveRef.current && !turnPromiseRef.current && !speechPromiseRef.current) {
        updateStatus('listening', '按住说话，松开发送')
      }
      return
    }
    if (release === 'discard' && recordingRef.current) {
      recordingRef.current = false
      setRecording(false)
      recorderPreparedRef.current = false
      const epoch = lifecycleEpochRef.current
      const task = (async () => {
        try { await recorder.stop() } catch {}
        if (isCurrentSession(epoch) && !manualCloseRequestedRef.current) {
          updateStatus('listening', '按住说话，松开发送')
        }
      })()
      turnPromiseRef.current = task
      void task.finally(() => {
        if (turnPromiseRef.current === task) turnPromiseRef.current = null
      })
      return
    }
    if (release !== 'send' || !recordingRef.current) return
    recordingRef.current = false
    setRecording(false)
    const task = runTurn()
    turnPromiseRef.current = task
    void task.finally(() => {
      if (turnPromiseRef.current === task) turnPromiseRef.current = null
    })
  }, [isCurrentSession, recorder, runTurn, updateStatus])

  const toggleRecording = useCallback(() => {
    if (pushToTalkGestureRef.current.getState() === 'idle') beginPushToTalk()
    else endPushToTalk()
  }, [beginPushToTalk, endPushToTalk])

  // Stable teardown ([] deps + refs): the unmount effect is `useEffect(() => cleanup, [cleanup])`,
  // so cleanup MUST NOT change identity mid-session — otherwise React fires the previous cleanup on
  // every `recording` toggle and the session self-destructs on the first mic tap.
  const cleanup = useCallback(() => {
    lifecycleEpochRef.current += 1
    startingRef.current = false
    const lease = runtimeLeaseRef.current
    runtimeLeaseRef.current = null
    lease?.release()
    playbackAbortRef.current?.abort()
    playbackAbortRef.current = null
    sessionActiveRef.current = false
    try { if (recordingRef.current) void recorderRef.current.stop() } catch {}
    try { playerRef.current?.remove() } catch {}
    playerRef.current = null
    speechPromiseRef.current = null
    turnPromiseRef.current = null
    pushToTalkGestureRef.current.reset()
    recorderPreparedRef.current = false
    recordingRef.current = false
    setRecording(false)
    setSessionActive(false)
    setConnecting(false)
  }, [])
  cleanupRef.current = cleanup

  const stopConversation = useCallback(async () => {
    if (endedRef.current) { cleanup(); return }
    if (stopPromiseRef.current) return stopPromiseRef.current
    const epoch = lifecycleEpochRef.current
    if (!isCurrentSession(epoch)) { cleanup(); return }

    manualCloseRequestedRef.current = true
    sessionActiveRef.current = false
    setSessionActive(false)
    updateStatus('processing', 'Wrapping up. Aria will say goodbye in a moment...')

    const task = (async () => {
      // A tap during recording means the unfinished utterance is discarded. A tap during an
      // assistant turn waits for that complete turn instead of removing its player midway.
      if (recordingRef.current) {
        pushToTalkGestureRef.current.reset()
        recordingRef.current = false
        setRecording(false)
        recorderPreparedRef.current = false
        try { await recorderRef.current.stop() } catch {}
      }
      const activeTurn = turnPromiseRef.current
      if (activeTurn) await activeTurn.catch(() => {})
      if (!isCurrentSession(epoch)) return
      const activeSpeech = speechPromiseRef.current
      if (activeSpeech) await activeSpeech.catch(() => {})
      if (!isCurrentSession(epoch)) return
      if (endedRef.current) { cleanup(); return }

      closingRef.current = true
      const goodbye = persona === 'companion'
        ? companionClosingTextForLanguage(voiceRef.current.languageKey)
        : closingTextForLanguage(voiceRef.current.languageKey)
      llmRef.current.push({ role: 'assistant', content: goodbye })
      addMessage('assistant', goodbye)
      const spoken = await speak(goodbye, 'Aria is saying goodbye...', epoch)
      if (!spoken || !isCurrentSession(epoch)) return

      endedRef.current = true
      updateStatus('idle', 'Goodbye complete. Saving the conversation...')
      setEnded(true)
      cleanup()
    })()
    stopPromiseRef.current = task
    try {
      await task
    } finally {
      if (stopPromiseRef.current === task) stopPromiseRef.current = null
    }
  }, [addMessage, cleanup, isCurrentSession, persona, speak, updateStatus])

  useEffect(() => cleanup, [cleanup])

  return {
    mode: 'http',
    statusKind,
    statusText,
    messages,
    startConversation,
    stopConversation,
    connecting,
    sessionActive,
    userSpeaking: recording,
    ended,
    recording,
    beginPushToTalk,
    endPushToTalk,
    toggleRecording,
  }
}
