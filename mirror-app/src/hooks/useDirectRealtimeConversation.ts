import { useCallback, useEffect, useRef, useState } from 'react'
import { Platform } from 'react-native'

import { getBearer } from '../api/qwenToken'
import { buildLiveSessionUpdate, realtimeWsUrl } from '../orchestration/realtime'
import {
  detectLanguageSignal,
  voiceProfileForLanguageKey,
  voiceProfileForSession,
  type VoiceProfile,
} from '../orchestration/voice'
import { createPcmAudioBridge, type PcmAudioBridge } from '../native/pcmAudio'
import { randomId } from '../utils/id'
import type { ChatMessage, ConversationApi, StatusKind } from './conversationTypes'

type Options = { patientId?: string; language?: string }

/**
 * Version 3 (Flavor A): NATIVE device opens a direct realtime WebSocket to Qwen (header auth
 * with a short-lived token minted by /api/qwen-token), running the on-device orchestration
 * (session.update / server-VAD / dynamic voice / wrap-up). No relay.
 *
 * Verified headlessly by server/smoke-direct-ws.mjs (token → direct WS → orchestration →
 * audio deltas). The only device-bound gap is native PCM capture/playback (src/native/pcmAudio.ts).
 * Web cannot set WS headers, so the selector routes web to the relay instead of this hook.
 */
export function useDirectRealtimeConversation(options: Options = {}): ConversationApi {
  const patientId = options.patientId ?? 'demo-patient'
  const language = options.language ?? 'en'

  const [statusKind, setStatusKind] = useState<StatusKind>('idle')
  const [statusText, setStatusText] = useState('Ready to start')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [connecting, setConnecting] = useState(false)
  const [sessionActive, setSessionActive] = useState(false)
  const [userSpeaking, setUserSpeaking] = useState(false)

  const socketRef = useRef<WebSocket | null>(null)
  const audioRef = useRef<PcmAudioBridge | null>(null)
  const voiceRef = useRef<VoiceProfile>(voiceProfileForSession(language))
  const openingRequestedRef = useRef(false)
  const streamIdRef = useRef<string | null>(null)
  const assistantTextRef = useRef('')
  const drainTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const wrapupTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const wrappingUpRef = useRef(false)

  const updateStatus = useCallback((kind: StatusKind, text: string) => {
    setStatusKind(kind)
    setStatusText(text)
  }, [])

  const clearDrain = useCallback(() => {
    if (drainTimerRef.current) {
      clearTimeout(drainTimerRef.current)
      drainTimerRef.current = null
    }
  }, [])

  const send = useCallback((event: Record<string, unknown>) => {
    const s = socketRef.current
    if (s && s.readyState === WebSocket.OPEN) s.send(JSON.stringify(event))
  }, [])

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
    if (id) setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, text: clean, streaming: false } : m)))
    else setMessages((prev) => [...prev, { id: randomId('assistant'), role: 'assistant', text: clean }])
  }, [])

  const applyVoice = useCallback((profile: VoiceProfile) => {
    voiceRef.current = profile
    send(buildLiveSessionUpdate(patientId, profile.languageLabel, { voice: profile.voice }))
  }, [patientId, send])

  const handleMessage = useCallback(
    (payload: any) => {
      const type = String(payload?.type || '')

      if (type === 'error') {
        // Qwen delivers app-level errors as in-band frames over the OPEN socket (not socket.onerror).
        // Surface them so an unattended kiosk doesn't hang silently, and free the mic.
        clearDrain()
        if (!wrappingUpRef.current) audioRef.current?.setCaptureMuted(false)
        updateStatus('error', String(payload?.error?.message || 'realtime error'))
        return
      }
      if (type === 'session.created' || type === 'session.updated') {
        if (!openingRequestedRef.current) {
          openingRequestedRef.current = true
          send({ type: 'response.create' })
        }
        return
      }
      if (type === 'conversation.item.input_audio_transcription.completed') {
        const transcript = String(payload?.transcript || '').trim()
        if (transcript) {
          setMessages((prev) => [...prev, { id: randomId('user'), role: 'user', text: transcript }])
          const signal = detectLanguageSignal(transcript)
          if (signal && signal.confidence >= 0.8 && signal.languageKey !== voiceRef.current.languageKey) {
            applyVoice(voiceProfileForLanguageKey(signal.languageKey, 'transcript_reassessment'))
          }
        }
        return
      }
      if (type === 'input_audio_buffer.speech_started') { setUserSpeaking(true); updateStatus('listening', 'Listening...'); return }
      if (type === 'input_audio_buffer.speech_stopped') { setUserSpeaking(false); updateStatus('processing', 'Thinking...'); return }
      if (type === 'response.created') { clearDrain(); audioRef.current?.setCaptureMuted(true); updateStatus('speaking', 'Speaking...'); return }
      if (type === 'response.audio.delta') { audioRef.current?.play(String(payload?.delta || '')); return }
      if (type === 'response.audio_transcript.delta' || type === 'response.output_audio_transcript.delta') {
        assistantTextRef.current += String(payload?.delta || '')
        appendAssistantStreaming(assistantTextRef.current)
        return
      }
      if (type === 'response.audio_transcript.done' || type === 'response.output_audio_transcript.done') {
        finalizeAssistant(String(payload?.transcript ?? assistantTextRef.current))
        return
      }
      if (type === 'response.done') {
        updateStatus('listening', 'Listening...')
        // During wrap-up we deliberately stay muted until cleanup (stopConversation drives teardown).
        if (wrappingUpRef.current) return
        // `response.done` fires when the last audio delta is ENQUEUED, not when it finishes playing.
        // Poll the native playback backlog and un-mute only once the assistant is actually silent,
        // otherwise the mirror captures its own voice tail and server-VAD self-triggers a turn.
        const bridge = audioRef.current
        clearDrain()
        const drain = () => {
          const backlogMs = bridge?.getPlaybackBacklogMs?.() ?? 0
          if (backlogMs <= 20 || bridge !== audioRef.current) {
            bridge?.setCaptureMuted(false)
            drainTimerRef.current = null
            return
          }
          drainTimerRef.current = setTimeout(drain, Math.min(backlogMs, 250))
        }
        drain()
        return
      }
    },
    [appendAssistantStreaming, applyVoice, clearDrain, finalizeAssistant, send, updateStatus],
  )

  const cleanup = useCallback(() => {
    if (drainTimerRef.current) { clearTimeout(drainTimerRef.current); drainTimerRef.current = null }
    if (wrapupTimerRef.current) { clearTimeout(wrapupTimerRef.current); wrapupTimerRef.current = null }
    wrappingUpRef.current = false
    void audioRef.current?.stop()
    audioRef.current = null
    try { socketRef.current?.close() } catch {}
    socketRef.current = null
    openingRequestedRef.current = false
    streamIdRef.current = null
    assistantTextRef.current = ''
    setSessionActive(false)
    setConnecting(false)
    setUserSpeaking(false)
  }, [])

  const startConversation = useCallback(async () => {
    if (Platform.OS === 'web') {
      updateStatus('error', 'Direct WS (v3) is native-only — web uses the relay (v1). Set mode=relay on web.')
      return
    }
    if (socketRef.current) return
    setConnecting(true)
    updateStatus('processing', 'Connecting...')
    setMessages([])
    voiceRef.current = voiceProfileForSession(language)
    openingRequestedRef.current = false

    try {
      // Short-lived server-minted token (secure) or, if explicitly enabled, the kiosk client key.
      const bearer = await getBearer()

      // RN WebSocket supports a 3rd `options.headers` arg (not in DOM types → cast).
      const socket = new (WebSocket as any)(realtimeWsUrl(), undefined, {
        headers: { Authorization: `Bearer ${bearer}` },
      }) as WebSocket
      socketRef.current = socket

      socket.onopen = () => {
        setSessionActive(true)
        setConnecting(false)
        updateStatus('listening', 'Listening...')
        send(buildLiveSessionUpdate(patientId, voiceRef.current.languageLabel, { voice: voiceRef.current.voice }))
        // Start native PCM capture -> stream frames upstream.
        const audio = createPcmAudioBridge()
        audioRef.current = audio
        audio
          .start((base64Pcm16) => send({ type: 'input_audio_buffer.append', audio: base64Pcm16 }))
          .catch((e: unknown) => updateStatus('error', e instanceof Error ? e.message : 'audio start failed'))
      }
      socket.onmessage = (event) => {
        try { handleMessage(JSON.parse(String(event.data))) } catch { /* ignore malformed */ }
      }
      socket.onerror = () => updateStatus('error', 'Realtime WS error (check token / region).')
      socket.onclose = () => { if (sessionActive) updateStatus('idle', 'Conversation ended'); cleanup() }
    } catch (e) {
      cleanup()
      updateStatus('error', `Error: ${e instanceof Error ? e.message : 'unknown'}`)
    }
  }, [cleanup, handleMessage, language, patientId, send, sessionActive, updateStatus])

  const stopConversation = useCallback(async () => {
    // Latch wrap-up and mute immediately: the goodbye's response.done must NOT re-open the mic, or
    // server-VAD spawns a spurious turn that teardown would truncate. Cancel any pending
    // normal-turn drain so it can't un-mute us mid-wrap-up.
    wrappingUpRef.current = true
    clearDrain()
    audioRef.current?.setCaptureMuted(true)
    // Ask for a graceful wrap-up (goodbye) before closing.
    send(buildLiveSessionUpdate(patientId, voiceRef.current.languageLabel, { voice: voiceRef.current.voice, wrapUp: true }))
    send({ type: 'response.create' })
    updateStatus('idle', 'Wrapping up...')
    // Wait for the queued goodbye to finish PLAYING before teardown (cap ~10s), instead of a fixed
    // 4s that could clip a longer goodbye. Give it a moment to start producing audio, then drain.
    const startedAt = Date.now()
    if (wrapupTimerRef.current) clearTimeout(wrapupTimerRef.current)
    const waitDrain = () => {
      const backlogMs = audioRef.current?.getPlaybackBacklogMs?.() ?? 0
      if (backlogMs <= 20 || Date.now() - startedAt > 10000) { cleanup(); return }
      wrapupTimerRef.current = setTimeout(waitDrain, Math.min(Math.max(backlogMs, 60), 300))
    }
    wrapupTimerRef.current = setTimeout(waitDrain, 1500)
  }, [cleanup, clearDrain, patientId, send, updateStatus])

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
  }
}

