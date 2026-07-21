import { useCallback, useEffect, useRef, useState } from 'react'
import { Platform } from 'react-native'

import { getBearer } from '../api/qwenToken'
import {
  HARD_MAX_TURN,
  RECALL_DEADLINE_TURN,
  RECALL_DIRECTIVE,
  looksLikeGoodbye,
  looksLikeRecallProbe,
} from '../orchestration/orchestrator'
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

type Options = {
  patientId?: string
  language?: string
  persona?: 'screening' | 'companion'
  // Manual push-to-talk: mic stays muted except while the user holds "speak". Kills the echo loop on
  // devices with no hardware AEC (e.g. an emulator sharing the host mic + speaker).
  pushToTalk?: boolean
  onUnavailable?: (reason: string) => void
}

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
  const persona = options.persona ?? 'screening'
  const pushToTalk = options.pushToTalk ?? false

  const [statusKind, setStatusKind] = useState<StatusKind>('idle')
  const [statusText, setStatusText] = useState('Ready to start')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [connecting, setConnecting] = useState(false)
  const [sessionActive, setSessionActive] = useState(false)
  const [userSpeaking, setUserSpeaking] = useState(false)
  const [ended, setEnded] = useState(false)
  const [recording, setRecording] = useState(false)

  const endedRef = useRef(false)
  const endTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const socketRef = useRef<WebSocket | null>(null)
  const audioRef = useRef<PcmAudioBridge | null>(null)
  const voiceRef = useRef<VoiceProfile>(voiceProfileForSession(language))
  const openingRequestedRef = useRef(false)
  const streamIdRef = useRef<string | null>(null)
  const assistantTextRef = useRef('')
  const drainTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const wrapupTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const wrappingUpRef = useRef(false)
  // Fallback plumbing: tell the supervisor (useConversation) when omni is unavailable so it can drop
  // to the turn-based (v2) stack. Only fires during STARTUP — once real audio/response has flowed
  // (hadResponseRef) a mid-session blip is surfaced as an error, not restarted in a new transport.
  const onUnavailableRef = useRef(options.onUnavailable)
  onUnavailableRef.current = options.onUnavailable
  const hadResponseRef = useRef(false)
  const openedRef = useRef(false)
  const connectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reportedUnavailableRef = useRef(false)
  // Deterministic recall floor + natural ending (parity with v2). The realtime model auto-responds
  // via server-VAD, so we drive the mandatory recall + goodbye from the turn counter instead of
  // relying on the model to conclude on its own (which it doesn't — it keeps asking follow-ups).
  const turnCountRef = useRef(0)
  const recallProbeIssuedRef = useRef(false)
  const recallAnsweredRef = useRef(false)
  const recallForcedRef = useRef(false)
  const closingRef = useRef(false)

  const updateStatus = useCallback((kind: StatusKind, text: string) => {
    setStatusKind(kind)
    setStatusText(text)
  }, [])

  const reportUnavailable = useCallback((reason: string) => {
    if (hadResponseRef.current || reportedUnavailableRef.current) return
    reportedUnavailableRef.current = true
    if (connectTimerRef.current) { clearTimeout(connectTimerRef.current); connectTimerRef.current = null }
    onUnavailableRef.current?.(reason)
  }, [])

  // Aria delivered her closing goodbye — auto-finalize. Keep the mic muted through the goodbye
  // tail, then flip `ended` so the screen runs the screening + save on its own.
  const scheduleEnd = useCallback(() => {
    if (endedRef.current) return
    endedRef.current = true
    wrappingUpRef.current = true
    audioRef.current?.setCaptureMuted(true)
    updateStatus('idle', '检查完成,正在生成判断…')
    if (endTimerRef.current) clearTimeout(endTimerRef.current)
    endTimerRef.current = setTimeout(() => setEnded(true), 3000)
  }, [updateStatus])

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
    send(buildLiveSessionUpdate(patientId, profile.languageLabel, { voice: profile.voice, languageKey: profile.languageKey, persona }))
  }, [patientId, persona, send])

  // Ask the model for its closing goodbye (once). The goodbye's transcript triggers scheduleEnd,
  // which flips `ended` so the screen finalizes (screening + save). Shared by the auto wrap-up and
  // the manual stopConversation.
  const requestGoodbye = useCallback(() => {
    if (closingRef.current) return
    closingRef.current = true
    wrappingUpRef.current = true
    clearDrain()
    audioRef.current?.setCaptureMuted(true)
    send(buildLiveSessionUpdate(patientId, voiceRef.current.languageLabel, { voice: voiceRef.current.voice, languageKey: voiceRef.current.languageKey, persona, wrapUp: true }))
    send({ type: 'response.create' })
  }, [clearDrain, patientId, persona, send])

  // The model reached the recall deadline without asking it — steer it to do the recall step now.
  const steerRecall = useCallback((): boolean => {
    if (recallForcedRef.current || closingRef.current) return false
    recallForcedRef.current = true
    send(buildLiveSessionUpdate(patientId, voiceRef.current.languageLabel, { voice: voiceRef.current.voice, languageKey: voiceRef.current.languageKey, persona, steer: RECALL_DIRECTIVE }))
    send({ type: 'response.create' })
    return true
  }, [patientId, persona, send])

  const handleMessage = useCallback(
    (payload: any) => {
      const type = String(payload?.type || '')

      if (type === 'error') {
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
          turnCountRef.current += 1
          // A patient turn that follows a recall probe = the recall was answered -> we may wrap up.
          if (recallProbeIssuedRef.current && !recallAnsweredRef.current) recallAnsweredRef.current = true
          const signal = detectLanguageSignal(transcript)
          if (signal && signal.confidence >= 0.8 && signal.languageKey !== voiceRef.current.languageKey) {
            applyVoice(voiceProfileForLanguageKey(signal.languageKey, 'transcript_reassessment'))
          }
        }
        return
      }
      if (type === 'input_audio_buffer.speech_started') { setUserSpeaking(true); updateStatus('listening', 'Listening...'); return }
      if (type === 'input_audio_buffer.speech_stopped') { setUserSpeaking(false); updateStatus('processing', 'Thinking...'); return }
      if (type === 'response.created') { hadResponseRef.current = true; clearDrain(); audioRef.current?.setCaptureMuted(true); updateStatus('speaking', 'Speaking...'); return }
      if (type === 'response.audio.delta') { audioRef.current?.play(String(payload?.delta || '')); return }
      if (type === 'response.audio_transcript.delta' || type === 'response.output_audio_transcript.delta') {
        assistantTextRef.current += String(payload?.delta || '')
        appendAssistantStreaming(assistantTextRef.current)
        return
      }
      if (type === 'response.audio_transcript.done' || type === 'response.output_audio_transcript.done') {
        const finalText = String(payload?.transcript ?? assistantTextRef.current)
        finalizeAssistant(finalText)
        // Only latch recall once we're actually in the recall window — otherwise an ordinary early
        // acknowledgement ("you mentioned…") trips the auto-goodbye and ends the check-in after ~2 topics.
        if (looksLikeRecallProbe(finalText) && (recallForcedRef.current || turnCountRef.current >= RECALL_DEADLINE_TURN)) {
          recallProbeIssuedRef.current = true
        }
        if (looksLikeGoodbye(finalText)) scheduleEnd()
        return
      }
      if (type === 'response.done') {
        updateStatus('listening', pushToTalk ? '按住说话' : 'Listening...')
        // During wrap-up we deliberately stay muted until cleanup (stopConversation drives teardown).
        if (wrappingUpRef.current) return
        // Natural ending: the realtime model won't conclude on its own, so drive it from the turn
        // counter. Once recall has been asked AND answered (or we hit the hard cap), request the
        // goodbye -> scheduleEnd -> finalize. If recall hasn't happened by the deadline, steer it now.
        // Recall floor + forced wrap-up is a SCREENING-only mechanism. The companion persona is
        // open chat with no agenda: it ends only on the user's goodbye (looksLikeGoodbye) or a manual
        // stop, so we skip this entirely and just drain/un-mute normally.
        if (persona === 'screening' && !closingRef.current) {
          if (recallAnsweredRef.current || turnCountRef.current >= HARD_MAX_TURN) { requestGoodbye(); return }
          // Only skip the drain (stay muted) if steerRecall ACTUALLY issued a new response; a no-op
          // (already forced) must fall through to the drain so the mic re-opens — else it wedges.
          if (!recallProbeIssuedRef.current && turnCountRef.current >= RECALL_DEADLINE_TURN && steerRecall()) return
        }
        // Push-to-talk: never auto-un-mute — the mic stays muted until the user holds "speak" again.
        if (pushToTalk) return
        // `response.done` fires when the last audio delta is ENQUEUED, not when it finishes playing.
        // Poll the native playback backlog (now includes the not-yet-written queue) and un-mute only
        // once the assistant is actually silent, otherwise the mirror captures its own voice and
        // server-VAD self-triggers a turn (Aria talking to her own echo).
        const bridge = audioRef.current
        clearDrain()
        const startedAt = Date.now()
        const THRESHOLD_MS = 120 // treat as drained once backlog is this small
        const GUARD_MS = 550 // acoustic + capture-pipeline tail to let the room go quiet (echo guard)
        const MAX_WAIT_MS = 20000 // safety: never leave the mic muted forever if playback wedges
        const drain = () => {
          if (bridge !== audioRef.current) { drainTimerRef.current = null; return }
          const backlogMs = bridge?.getPlaybackBacklogMs?.() ?? 0
          if (backlogMs <= THRESHOLD_MS || Date.now() - startedAt > MAX_WAIT_MS) {
            // Playback has drained — wait a short guard tail, then re-open the mic (unless we've
            // since started wrapping up or the session was torn down).
            drainTimerRef.current = setTimeout(() => {
              drainTimerRef.current = null
              if (bridge === audioRef.current && !wrappingUpRef.current) bridge?.setCaptureMuted(false)
            }, GUARD_MS)
            return
          }
          drainTimerRef.current = setTimeout(drain, Math.min(backlogMs, 250))
        }
        drain()
        return
      }
    },
    [appendAssistantStreaming, applyVoice, clearDrain, finalizeAssistant, persona, reportUnavailable, requestGoodbye, scheduleEnd, steerRecall, send, updateStatus],
  )

  const cleanup = useCallback(() => {
    if (drainTimerRef.current) { clearTimeout(drainTimerRef.current); drainTimerRef.current = null }
    if (wrapupTimerRef.current) { clearTimeout(wrapupTimerRef.current); wrapupTimerRef.current = null }
    if (endTimerRef.current) { clearTimeout(endTimerRef.current); endTimerRef.current = null }
    if (connectTimerRef.current) { clearTimeout(connectTimerRef.current); connectTimerRef.current = null }
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
    setEnded(false)
    endedRef.current = false
    voiceRef.current = voiceProfileForSession(language)
    openingRequestedRef.current = false
    hadResponseRef.current = false
    openedRef.current = false
    reportedUnavailableRef.current = false
    turnCountRef.current = 0
    recallProbeIssuedRef.current = false
    recallAnsweredRef.current = false
    recallForcedRef.current = false
    closingRef.current = false
    setRecording(false)

    try {
      // Short-lived server-minted token (secure) or, if explicitly enabled, the kiosk client key.
      const bearer = await getBearer()

      // RN WebSocket supports a 3rd `options.headers` arg (not in DOM types → cast).
      const socket = new (WebSocket as any)(realtimeWsUrl(), undefined, {
        headers: { Authorization: `Bearer ${bearer}` },
      }) as WebSocket
      socketRef.current = socket
      // Connection watchdog: if the socket never opens within 7s (dead region/network), fall back
      // to the turn-based stack instead of hanging in "Connecting..." forever.
      if (connectTimerRef.current) clearTimeout(connectTimerRef.current)
      connectTimerRef.current = setTimeout(() => {
        connectTimerRef.current = null
        if (!openedRef.current && !hadResponseRef.current) { reportUnavailable('ws_connect_timeout'); cleanup() }
      }, 7000)

      socket.onopen = () => {
        openedRef.current = true
        if (connectTimerRef.current) { clearTimeout(connectTimerRef.current); connectTimerRef.current = null }
        setSessionActive(true)
        setConnecting(false)
        updateStatus('listening', 'Listening...')
        send(buildLiveSessionUpdate(patientId, voiceRef.current.languageLabel, { voice: voiceRef.current.voice, languageKey: voiceRef.current.languageKey, persona }))
        // Start native PCM capture -> stream frames upstream.
        const audio = createPcmAudioBridge()
        audioRef.current = audio
        // Push-to-talk: capture starts muted; the mic only opens while the user holds "speak".
        if (pushToTalk) { audio.setCaptureMuted(true); updateStatus('listening', '按住说话') }
        audio
          .start((base64Pcm16) => send({ type: 'input_audio_buffer.append', audio: base64Pcm16 }))
          .catch((e: unknown) => updateStatus('error', e instanceof Error ? e.message : 'audio start failed'))
      }
      socket.onmessage = (event) => {
        try { handleMessage(JSON.parse(String(event.data))) } catch { /* ignore malformed */ }
      }
      socket.onerror = () => {
        // Before any response: omni is unreachable -> let the supervisor fall back (onclose cleans up).
        if (!hadResponseRef.current) { reportUnavailable('ws_error'); return }
        updateStatus('error', 'Realtime WS error (check token / region).')
      }
      socket.onclose = () => {
        // Closed before it ever opened (region block / handshake reject / connect fail): fall back.
        if (!openedRef.current && !hadResponseRef.current) {
          reportUnavailable('ws_closed_before_open')
        } else if (hadResponseRef.current && !closingRef.current) {
          // Provider closed a LIVE session (e.g. the 120-min / turn cap) after real turns — don't
          // silently drop the check-in: drive finalize so assessCheckin + saveCheckin still run.
          updateStatus('idle', 'Conversation ended')
          endedRef.current = true
          setEnded(true)
        } else if (sessionActive) {
          updateStatus('idle', 'Conversation ended')
        }
        cleanup()
      }
    } catch (e) {
      cleanup()
      if (!hadResponseRef.current) { reportUnavailable('ws_start_failed'); return }
      updateStatus('error', `Error: ${e instanceof Error ? e.message : 'unknown'}`)
    }
  }, [cleanup, handleMessage, language, patientId, persona, reportUnavailable, send, sessionActive, updateStatus])

  const stopConversation = useCallback(async () => {
    // If an auto wrap-up (goodbye) is already in flight (the natural-ending path), don't request a
    // SECOND goodbye — just latch mute + teardown. Otherwise ask for the graceful goodbye now.
    // (requestGoodbye is guarded by closingRef, so it also latches wrap-up state.)
    if (!closingRef.current) {
      requestGoodbye()
    } else {
      wrappingUpRef.current = true
      clearDrain()
      audioRef.current?.setCaptureMuted(true)
    }
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
  }, [cleanup, clearDrain, requestGoodbye, updateStatus])

  // Push-to-talk control: tap to open the mic ("speaking"), tap again to send — muting ends the turn
  // so server-VAD generates Aria's reply. No-op unless pushToTalk is enabled.
  const toggleRecording = useCallback(() => {
    if (!pushToTalk || wrappingUpRef.current) return
    setRecording((r) => {
      const next = !r
      audioRef.current?.setCaptureMuted(!next)
      setUserSpeaking(next)
      return next
    })
  }, [pushToTalk])

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
    ended,
    recording: pushToTalk ? recording : undefined,
    toggleRecording: pushToTalk ? toggleRecording : undefined,
  }
}

