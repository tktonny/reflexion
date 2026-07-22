import { useCallback, useEffect, useRef, useState } from 'react'
import { Platform } from 'react-native'

import { QWEN } from '../config/conversationMode'
import {
  RECALL_DEADLINE_TURN,
  RECALL_DIRECTIVE,
  WRAPUP_DIRECTIVE,
  buildLiveInstructions,
  looksLikeRecallProbe,
  looksLikeUserGoodbye,
  openingMessageForLanguage,
  recallBudgetStep,
} from '../orchestration/orchestrator'
import {
  companionClosingTextForLanguage,
  screeningQuestionForTurn,
} from '../orchestration/deterministicSpeech'
import { detectLanguageSignal, voiceProfileForLanguageKey, voiceProfileForSession, type VoiceProfile } from '../orchestration/voice'
import { qwenASR, qwenChat, qwenTTS, type QwenChatMessage } from '../api/qwenClient'
import { randomId } from '../utils/id'
import type { ChatMessage, ConversationApi, StatusKind } from './conversationTypes'

const CAPTURE_SAMPLE_RATE = 16000

type Options = { patientId?: string; language?: string; persona?: 'screening' | 'companion' }

/**
 * Version 2 (Flavor B): fully on-device turn-based voice loop over plain HTTPS —
 * record a turn -> ASR -> Qwen chat (on-device orchestration prompt) -> TTS -> play.
 * No relay. Web path uses the browser Web Audio stack (capture PCM -> WAV).
 * Native path needs a recorder module (expo-audio) — TODO; throws a clear message for now.
 */
export function useTurnBasedConversation(options: Options = {}): ConversationApi {
  const patientId = options.patientId ?? 'demo-patient'
  const language = options.language ?? 'en'
  const persona = options.persona ?? 'screening'

  const [statusKind, setStatusKind] = useState<StatusKind>('idle')
  const [statusText, setStatusText] = useState('Ready to start')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [connecting, setConnecting] = useState(false)
  const [sessionActive, setSessionActive] = useState(false)
  const [recording, setRecording] = useState(false)

  const audioContextRef = useRef<any>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<any>(null)
  const sourceRef = useRef<any>(null)
  const chunksRef = useRef<Float32Array[]>([])
  const recordingRef = useRef(false)
  const llmRef = useRef<QwenChatMessage[]>([])
  const voiceRef = useRef<VoiceProfile>(voiceProfileForSession(language))
  const playbackCursorRef = useRef(0)
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

  const playWav = useCallback(async (wavBase64: string | null) => {
    const ctx = audioContextRef.current
    if (!ctx || !wavBase64) return
    const bytes = base64ToBytes(wavBase64)
    const buffer = await ctx.decodeAudioData(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))
    const source = ctx.createBufferSource()
    source.buffer = buffer
    source.connect(ctx.destination)
    const startAt = Math.max(ctx.currentTime, playbackCursorRef.current || 0)
    source.start(startAt)
    playbackCursorRef.current = startAt + buffer.duration
  }, [])

  const speakAssistant = useCallback(async (text: string) => {
    updateStatus('speaking', 'Speaking...')
    try {
      const tts = await qwenTTS(text, { voice: voiceRef.current.voice })
      await playWav(tts.audioBase64)
    } catch (e) {
      // TTS failure is non-fatal for the transcript; surface but continue.
      updateStatus('error', `TTS failed: ${e instanceof Error ? e.message : 'unknown'}`)
    }
  }, [playWav, updateStatus])

  const ensureAudio = useCallback(async () => {
    if (audioContextRef.current) return
    const w = window as any
    const ctx = new (w.AudioContext || w.webkitAudioContext)()
    audioContextRef.current = ctx
    await ctx.resume?.()
    const stream = await (navigator as any).mediaDevices.getUserMedia({ audio: true, video: false })
    mediaStreamRef.current = stream
    const source = ctx.createMediaStreamSource(stream)
    const processor = ctx.createScriptProcessor(4096, 1, 1)
    const mute = ctx.createGain()
    mute.gain.value = 0
    processor.onaudioprocess = (event: any) => {
      if (!recordingRef.current) return
      chunksRef.current.push(new Float32Array(event.inputBuffer.getChannelData(0)))
    }
    source.connect(processor)
    processor.connect(mute)
    mute.connect(ctx.destination)
    sourceRef.current = source
    processorRef.current = processor
  }, [])

  const startConversation = useCallback(async () => {
    if (Platform.OS !== 'web') {
      updateStatus('error', 'Turn-based audio currently runs on web only (native needs expo-audio recorder — TODO).')
      return
    }
    setConnecting(true)
    updateStatus('processing', 'Starting...')
    setMessages([])
    voiceRef.current = voiceProfileForSession(language)
    llmRef.current = [{ role: 'system', content: buildLiveInstructions(patientId, language, { persona }) }]
    turnCountRef.current = 0
    recallProbeIssuedRef.current = false
    recallAnsweredRef.current = false
    closingRef.current = false

    try {
      await ensureAudio()
      sessionActiveRef.current = true
      setSessionActive(true)
      setConnecting(false)
      // Assistant speaks the scripted opening first.
      const opening = openingMessageForLanguage(language)
      llmRef.current.push({ role: 'assistant', content: opening })
      addMessage('assistant', opening)
      await speakAssistant(opening)
      updateStatus('listening', 'Tap the mic to answer')
    } catch (e) {
      cleanup()
      updateStatus('error', `Error: ${e instanceof Error ? e.message : 'unknown'}`)
    }
  }, [addMessage, ensureAudio, language, patientId, persona, speakAssistant, updateStatus])

  const runTurn = useCallback(async () => {
    updateStatus('processing', 'Transcribing...')
    const wavBase64 = encodeWavBase64(chunksRef.current, audioContextRef.current?.sampleRate || 48000)
    chunksRef.current = []
    if (!wavBase64) { updateStatus('listening', 'Tap the mic to answer'); return }
    try {
      const transcript = await qwenASR(wavBase64)
      if (!transcript) { updateStatus('listening', "Didn't catch that — tap to try again"); return }
      addMessage('user', transcript)
      llmRef.current.push({ role: 'user', content: transcript })
      turnCountRef.current += 1

      // Dynamic voice: switch TTS voice if the patient's language clearly changed.
      const signal = detectLanguageSignal(transcript)
      if (signal && signal.confidence >= 0.8 && signal.languageKey !== voiceRef.current.languageKey) {
        voiceRef.current = voiceProfileForLanguageKey(signal.languageKey, 'transcript_reassessment')
      }

      // Layer 2: deterministic recall floor — inject a one-turn steering directive if needed.
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
        else if (step.action === 'wrap_up') { recallAnsweredRef.current = true; directive = WRAPUP_DIRECTIVE; closingRef.current = true }
        else scriptedQuestion = screeningQuestionForTurn(voiceRef.current.languageKey, turnCountRef.current)
      }
      if (companionGoodbyeRequested) closingRef.current = true

      updateStatus('processing', 'Thinking...')
      // Ephemeral directive: passed for this one turn only, NOT pushed into history.
      const msgs = directive ? [...llmRef.current, { role: 'system' as const, content: directive }] : llmRef.current
      const reply = companionGoodbyeRequested
        ? companionClosingTextForLanguage(voiceRef.current.languageKey)
        : scriptedQuestion ?? await qwenChat(msgs, { maxTokens: 120, temperature: 0.4 })
      if (reply) {
        llmRef.current.push({ role: 'assistant', content: reply })
        addMessage('assistant', reply)
        if (
          persona === 'screening' &&
          !recallProbeIssuedRef.current &&
          turnCountRef.current >= RECALL_DEADLINE_TURN &&
          looksLikeRecallProbe(reply)
        ) recallProbeIssuedRef.current = true
        await speakAssistant(reply)
      }
      if (closingRef.current) {
        sessionActiveRef.current = false
        setSessionActive(false)
        updateStatus('idle', '结束了,可点"结束并评估"看判断')
      } else {
        updateStatus('listening', 'Tap the mic to answer')
      }
    } catch (e) {
      updateStatus('error', `Error: ${e instanceof Error ? e.message : 'unknown'}`)
    }
  }, [addMessage, persona, speakAssistant, updateStatus])

  const beginPushToTalk = useCallback(() => {
    if (!sessionActiveRef.current) return
    if (recordingRef.current) return
    chunksRef.current = []
    recordingRef.current = true
    setRecording(true)
    updateStatus('listening', 'Listening… release to send')
  }, [updateStatus])

  const endPushToTalk = useCallback(() => {
    if (!recordingRef.current) return
    recordingRef.current = false
    setRecording(false)
    void runTurn()
  }, [runTurn])

  const toggleRecording = useCallback(() => {
    if (recordingRef.current) endPushToTalk()
    else beginPushToTalk()
  }, [beginPushToTalk, endPushToTalk])

  const cleanup = useCallback(() => {
    recordingRef.current = false
    sessionActiveRef.current = false
    try { processorRef.current?.disconnect() } catch {}
    try { sourceRef.current?.disconnect() } catch {}
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop())
    try { audioContextRef.current?.close() } catch {}
    audioContextRef.current = null
    mediaStreamRef.current = null
    processorRef.current = null
    sourceRef.current = null
    chunksRef.current = []
    playbackCursorRef.current = 0
    setRecording(false)
    setSessionActive(false)
    setConnecting(false)
  }, [])

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
    beginPushToTalk,
    endPushToTalk,
    toggleRecording,
  }
}

// ---- audio helpers (web) ----

function base64ToBytes(base64: string): Uint8Array {
  const binary = (globalThis as any).atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)
  return bytes
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = ''
  const chunk = 0x8000
  for (let i = 0; i < bytes.length; i += chunk) binary += String.fromCharCode(...bytes.subarray(i, i + chunk))
  return (globalThis as any).btoa(binary)
}

/** Concatenate recorded Float32 chunks, downsample to 16k mono, encode a 16-bit PCM WAV, return base64. */
function encodeWavBase64(chunks: Float32Array[], inputSampleRate: number): string | null {
  const total = chunks.reduce((n, c) => n + c.length, 0)
  if (total === 0) return null
  const merged = new Float32Array(total)
  let off = 0
  for (const c of chunks) { merged.set(c, off); off += c.length }

  const ratio = inputSampleRate / CAPTURE_SAMPLE_RATE
  const outLen = Math.max(1, Math.floor(merged.length / ratio))
  const pcm = new Int16Array(outLen)
  let oi = 0
  let bi = 0
  while (oi < outLen) {
    const next = Math.min(merged.length, Math.round((oi + 1) * ratio))
    let acc = 0
    let cnt = 0
    for (let i = bi; i < next; i += 1) { acc += merged[i]; cnt += 1 }
    const s = cnt > 0 ? acc / cnt : merged[bi] || 0
    const clamped = Math.max(-1, Math.min(1, s))
    pcm[oi] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff
    oi += 1
    bi = next
  }

  // WAV (RIFF) 16-bit PCM mono @ 16k
  const dataBytes = pcm.length * 2
  const buffer = new ArrayBuffer(44 + dataBytes)
  const view = new DataView(buffer)
  const writeStr = (o: number, s: string) => { for (let i = 0; i < s.length; i += 1) view.setUint8(o + i, s.charCodeAt(i)) }
  writeStr(0, 'RIFF')
  view.setUint32(4, 36 + dataBytes, true)
  writeStr(8, 'WAVE')
  writeStr(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, CAPTURE_SAMPLE_RATE, true)
  view.setUint32(28, CAPTURE_SAMPLE_RATE * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeStr(36, 'data')
  view.setUint32(40, dataBytes, true)
  let p = 44
  for (let i = 0; i < pcm.length; i += 1) { view.setInt16(p, pcm[i], true); p += 2 }
  return bytesToBase64(new Uint8Array(buffer))
}
