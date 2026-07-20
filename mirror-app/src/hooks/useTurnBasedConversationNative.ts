import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AudioModule,
  RecordingPresets,
  createAudioPlayer,
  setAudioModeAsync,
  useAudioRecorder,
} from 'expo-audio'
import * as FileSystem from 'expo-file-system'

import {
  RECALL_DIRECTIVE,
  WRAPUP_DIRECTIVE,
  buildLiveInstructions,
  looksLikeRecallProbe,
  openingMessageForLanguage,
  recallBudgetStep,
} from '../orchestration/orchestrator'
import { detectLanguageSignal, voiceProfileForLanguageKey, voiceProfileForSession, type VoiceProfile } from '../orchestration/voice'
import { qwenASR, qwenChat, qwenTTS, type QwenChatMessage } from '../api/qwenClient'
import { randomId } from '../utils/id'
import type { ChatMessage, ConversationApi, StatusKind } from './conversationTypes'

type Options = { patientId?: string; language?: string }

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

  const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY)

  const [statusKind, setStatusKind] = useState<StatusKind>('idle')
  const [statusText, setStatusText] = useState('Ready to start')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [connecting, setConnecting] = useState(false)
  const [sessionActive, setSessionActive] = useState(false)
  const [recording, setRecording] = useState(false)

  const llmRef = useRef<QwenChatMessage[]>([])
  const voiceRef = useRef<VoiceProfile>(voiceProfileForSession(language))
  const playerRef = useRef<ReturnType<typeof createAudioPlayer> | null>(null)
  const sessionActiveRef = useRef(false)
  const turnCountRef = useRef(0)
  const recallProbeIssuedRef = useRef(false)
  const recallAnsweredRef = useRef(false)
  const closingRef = useRef(false)

  const updateStatus = useCallback((kind: StatusKind, text: string) => {
    setStatusKind(kind)
    setStatusText(text)
  }, [])

  const addMessage = useCallback((role: ChatMessage['role'], text: string) => {
    setMessages((prev) => [...prev, { id: randomId(role), role, text }])
  }, [])

  const play = useCallback(async (tts: { audioBase64: string | null; url: string | null }) => {
    const source = tts.url || (tts.audioBase64 ? `data:audio/wav;base64,${tts.audioBase64}` : null)
    if (!source) return
    try {
      playerRef.current?.remove()
      const player = createAudioPlayer(source)
      playerRef.current = player
      player.play()
    } catch {
      /* playback failure is non-fatal */
    }
  }, [])

  const speak = useCallback(async (text: string) => {
    updateStatus('speaking', 'Speaking...')
    try {
      const tts = await qwenTTS(text, { voice: voiceRef.current.voice })
      await play(tts)
    } catch (e) {
      updateStatus('error', `TTS failed: ${e instanceof Error ? e.message : 'unknown'}`)
    }
  }, [play, updateStatus])

  const startConversation = useCallback(async () => {
    setConnecting(true)
    updateStatus('processing', 'Starting...')
    setMessages([])
    voiceRef.current = voiceProfileForSession(language)
    llmRef.current = [{ role: 'system', content: buildLiveInstructions(patientId, language, {}) }]
    turnCountRef.current = 0
    recallProbeIssuedRef.current = false
    recallAnsweredRef.current = false
    closingRef.current = false
    try {
      const { granted } = await AudioModule.requestRecordingPermissionsAsync()
      if (!granted) throw new Error('microphone permission denied')
      await setAudioModeAsync({ playsInSilentMode: true, allowsRecording: true })
      sessionActiveRef.current = true
      setSessionActive(true)
      setConnecting(false)
      const opening = openingMessageForLanguage(language)
      llmRef.current.push({ role: 'assistant', content: opening })
      addMessage('assistant', opening)
      await speak(opening)
      updateStatus('listening', '点麦克风开始回答')
    } catch (e) {
      cleanup()
      updateStatus('error', `Error: ${e instanceof Error ? e.message : 'unknown'}`)
    }
  }, [addMessage, language, patientId, speak, updateStatus])

  const runTurn = useCallback(async () => {
    updateStatus('processing', 'Transcribing...')
    try {
      await recorder.stop()
      const uri = recorder.uri
      if (!uri) { updateStatus('listening', '点麦克风开始回答'); return }
      const base64 = await FileSystem.readAsStringAsync(uri, { encoding: 'base64' as FileSystem.EncodingType })
      const transcript = await qwenASR(base64, { format: 'm4a' })
      if (!transcript) { updateStatus('listening', '没听清,再试一次'); return }
      addMessage('user', transcript)
      llmRef.current.push({ role: 'user', content: transcript })
      turnCountRef.current += 1

      const signal = detectLanguageSignal(transcript)
      if (signal && signal.confidence >= 0.8 && signal.languageKey !== voiceRef.current.languageKey) {
        voiceRef.current = voiceProfileForLanguageKey(signal.languageKey, 'transcript_reassessment')
      }

      const step = recallBudgetStep({
        turnCount: turnCountRef.current,
        recallProbeIssued: recallProbeIssuedRef.current,
        recallAnswered: recallAnsweredRef.current,
      })
      let directive: string | null = null
      if (step.action === 'force_recall') { recallProbeIssuedRef.current = true; directive = RECALL_DIRECTIVE }
      else if (step.action === 'wrap_up') { recallAnsweredRef.current = true; directive = WRAPUP_DIRECTIVE; closingRef.current = true }

      updateStatus('processing', 'Thinking...')
      const msgs = directive ? [...llmRef.current, { role: 'system' as const, content: directive }] : llmRef.current
      const reply = await qwenChat(msgs, { maxTokens: 120, temperature: 0.4 })
      if (reply) {
        llmRef.current.push({ role: 'assistant', content: reply })
        addMessage('assistant', reply)
        if (!recallProbeIssuedRef.current && looksLikeRecallProbe(reply)) recallProbeIssuedRef.current = true
        await speak(reply)
      }
      if (closingRef.current) {
        sessionActiveRef.current = false
        setSessionActive(false)
        updateStatus('idle', '结束了,可点"结束并评估"看判断')
      } else {
        updateStatus('listening', '点麦克风开始回答')
      }
    } catch (e) {
      updateStatus('error', `Error: ${e instanceof Error ? e.message : 'unknown'}`)
    }
  }, [addMessage, recorder, speak, updateStatus])

  const toggleRecording = useCallback(() => {
    if (!sessionActiveRef.current) return
    if (recording) {
      setRecording(false)
      void runTurn()
    } else {
      void (async () => {
        await recorder.prepareToRecordAsync()
        recorder.record()
        setRecording(true)
        updateStatus('listening', '录音中… 再点一次发送')
      })()
    }
  }, [recorder, recording, runTurn, updateStatus])

  const cleanup = useCallback(() => {
    sessionActiveRef.current = false
    try { if (recording) void recorder.stop() } catch {}
    try { playerRef.current?.remove() } catch {}
    playerRef.current = null
    setRecording(false)
    setSessionActive(false)
    setConnecting(false)
  }, [recorder, recording])

  const stopConversation = useCallback(async () => {
    cleanup()
    updateStatus('idle', 'Conversation ended')
  }, [cleanup, updateStatus])

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
    recording,
    toggleRecording,
  }
}
