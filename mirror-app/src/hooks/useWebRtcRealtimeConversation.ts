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
import { buildLiveSessionUpdate, hasWebrtcHost, realtimeWebrtcUrl } from '../orchestration/realtime'
import {
  detectLanguageSignal,
  voiceProfileForLanguageKey,
  voiceProfileForSession,
  type VoiceProfile,
} from '../orchestration/voice'
import { randomId } from '../utils/id'
import type { ChatMessage, ConversationApi, StatusKind } from './conversationTypes'

/**
 * webrtc-v0.0.0 — native WebRTC realtime to Qwen-Omni.
 *
 * Unlike websocket-v0.0.0 (useDirectRealtimeConversation), the audio rides an RTP media track that
 * libwebrtc processes with BUILT-IN acoustic echo cancellation + noise reduction. That removes the
 * root cause of the mirror's self-conversation, so this hook needs NO half-duplex mic-muting, NO
 * playback-drain guard, and NO transcript-level echo suppressor. Only the non-audio protocol events
 * (session.update, transcripts, response lifecycle) travel over the `oai-events` data channel; the
 * shared orchestration (opening, deterministic recall floor, natural goodbye) is reused verbatim.
 *
 * Connection: create RTCPeerConnection + local mic track + data channel, createOffer, POST the offer
 * SDP to the workspace-scoped Qwen WebRTC endpoint (Bearer auth, Content-Type application/sdp), apply
 * the answer SDP. See realtimeWebrtcUrl() — needs EXPO_PUBLIC_QWEN_WORKSPACE_ID (the generic dashscope
 * host is WebSocket-only).
 *
 * react-native-webrtc is loaded lazily so the web bundle (which never selects this hook) doesn't need it.
 */

type Options = {
  patientId?: string
  language?: string
  persona?: 'screening' | 'companion'
  onUnavailable?: (reason: string) => void
}

type WebrtcModule = {
  RTCPeerConnection: new (config?: unknown) => any
  RTCSessionDescription: new (init: { type: string; sdp: string }) => any
  mediaDevices: { getUserMedia: (constraints: unknown) => Promise<any> }
}

function loadWebrtc(): WebrtcModule | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    return require('react-native-webrtc') as WebrtcModule
  } catch {
    return null
  }
}

const CONNECT_TIMEOUT_MS = 8000

export function useWebRtcRealtimeConversation(options: Options = {}): ConversationApi {
  const patientId = options.patientId ?? 'demo-patient'
  const language = options.language ?? 'en'
  const persona = options.persona ?? 'screening'

  const [statusKind, setStatusKind] = useState<StatusKind>('idle')
  const [statusText, setStatusText] = useState('Ready to start')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [connecting, setConnecting] = useState(false)
  const [sessionActive, setSessionActive] = useState(false)
  const [userSpeaking, setUserSpeaking] = useState(false)
  const [ended, setEnded] = useState(false)

  const pcRef = useRef<any>(null)
  const dcRef = useRef<any>(null)
  const localStreamRef = useRef<any>(null)
  const remoteStreamRef = useRef<any>(null)
  const streamIdRef = useRef<string | null>(null)
  const assistantTextRef = useRef('')
  const voiceRef = useRef<VoiceProfile>(voiceProfileForSession(language))
  const openingRequestedRef = useRef(false)

  const endedRef = useRef(false)
  const endTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const connectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const hadResponseRef = useRef(false)
  const openedRef = useRef(false)
  const reportedUnavailableRef = useRef(false)
  const wrappingUpRef = useRef(false)

  // Deterministic recall floor + natural ending (parity with websocket-v0.0.0).
  const turnCountRef = useRef(0)
  const recallProbeIssuedRef = useRef(false)
  const recallAnsweredRef = useRef(false)
  const recallForcedRef = useRef(false)
  const closingRef = useRef(false)

  const onUnavailableRef = useRef(options.onUnavailable)
  onUnavailableRef.current = options.onUnavailable

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

  // Send a client event over the data channel (JSON). No-op until the channel is open.
  const send = useCallback((event: Record<string, unknown>) => {
    const dc = dcRef.current
    if (dc && dc.readyState === 'open') dc.send(JSON.stringify(event))
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

  // Aria delivered her closing goodbye — auto-finalize. WebRTC keeps playing her goodbye over the RTP
  // track; give it a moment, then flip `ended` so the screen runs the screening + save.
  const scheduleEnd = useCallback(() => {
    if (endedRef.current) return
    endedRef.current = true
    wrappingUpRef.current = true
    updateStatus('idle', '检查完成,正在生成判断…')
    if (endTimerRef.current) clearTimeout(endTimerRef.current)
    endTimerRef.current = setTimeout(() => setEnded(true), 3000)
  }, [updateStatus])

  const requestGoodbye = useCallback(() => {
    if (closingRef.current) return
    closingRef.current = true
    wrappingUpRef.current = true
    send(buildLiveSessionUpdate(patientId, voiceRef.current.languageLabel, { voice: voiceRef.current.voice, languageKey: voiceRef.current.languageKey, persona, wrapUp: true }))
    send({ type: 'response.create' })
  }, [patientId, persona, send])

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
        if (!hadResponseRef.current) {
          reportUnavailable('webrtc_error_frame')
          return
        }
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
      if (type === 'response.created') { hadResponseRef.current = true; updateStatus('speaking', 'Speaking...'); return }
      // Assistant AUDIO is delivered on the RTP track (libwebrtc plays it, with AEC); only the TEXT
      // transcript comes over the data channel.
      if (type === 'response.audio_transcript.delta' || type === 'response.output_audio_transcript.delta') {
        assistantTextRef.current += String(payload?.delta || '')
        appendAssistantStreaming(assistantTextRef.current)
        return
      }
      if (type === 'response.audio_transcript.done' || type === 'response.output_audio_transcript.done') {
        const finalText = String(payload?.transcript ?? assistantTextRef.current)
        finalizeAssistant(finalText)
        if (looksLikeRecallProbe(finalText) && (recallForcedRef.current || turnCountRef.current >= RECALL_DEADLINE_TURN)) {
          recallProbeIssuedRef.current = true
        }
        if (persona === 'screening' && looksLikeGoodbye(finalText)) scheduleEnd()
        return
      }
      if (type === 'response.done') {
        updateStatus('listening', 'Listening...')
        if (wrappingUpRef.current) return
        // Screening-only deterministic recall floor + wrap-up (companion ends naturally on goodbye).
        if (persona === 'screening' && !closingRef.current) {
          if (recallAnsweredRef.current || turnCountRef.current >= HARD_MAX_TURN) { requestGoodbye(); return }
          if (!recallProbeIssuedRef.current && turnCountRef.current >= RECALL_DEADLINE_TURN) steerRecall()
        }
        return
      }
    },
    [appendAssistantStreaming, applyVoice, finalizeAssistant, persona, reportUnavailable, requestGoodbye, scheduleEnd, send, steerRecall, updateStatus],
  )

  const cleanup = useCallback(() => {
    if (connectTimerRef.current) { clearTimeout(connectTimerRef.current); connectTimerRef.current = null }
    if (endTimerRef.current) { clearTimeout(endTimerRef.current); endTimerRef.current = null }
    try { dcRef.current?.close?.() } catch {}
    try {
      localStreamRef.current?.getTracks?.().forEach((t: any) => { try { t.stop() } catch {} })
    } catch {}
    try { pcRef.current?.close?.() } catch {}
    dcRef.current = null
    pcRef.current = null
    localStreamRef.current = null
    remoteStreamRef.current = null
    setSessionActive(false)
    setConnecting(false)
    setUserSpeaking(false)
  }, [])

  // Wait for ICE gathering to finish so the offer SDP carries candidates (Qwen does a single POST
  // exchange, not trickle ICE). Resolves early on the `complete` state, with a hard cap.
  const waitForIceGathering = useCallback((pc: any): Promise<void> => {
    return new Promise((resolve) => {
      if (pc.iceGatheringState === 'complete') { resolve(); return }
      let done = false
      const finish = () => { if (done) return; done = true; resolve() }
      const check = () => { if (pc.iceGatheringState === 'complete') finish() }
      pc.addEventListener('icegatheringstatechange', check)
      pc.addEventListener('icecandidate', (e: any) => { if (!e.candidate) finish() })
      setTimeout(finish, 2000)
    })
  }, [])

  const startConversation = useCallback(async () => {
    if (Platform.OS === 'web') {
      updateStatus('error', 'WebRTC (webrtc-v0.0.0) is native-only — web uses the relay.')
      reportUnavailable('webrtc_web_unsupported')
      return
    }
    const webrtc = loadWebrtc()
    if (!webrtc) {
      updateStatus('error', 'react-native-webrtc not linked in this build.')
      reportUnavailable('webrtc_module_missing')
      return
    }
    if (!hasWebrtcHost()) {
      // No workspace host → the POST would 404. Surface it and let the supervisor fall back to v2.
      updateStatus('error', '未配置 WebRTC workspace 域名(EXPO_PUBLIC_QWEN_WORKSPACE_ID)')
      reportUnavailable('webrtc_no_host')
      return
    }

    setMessages([])
    setEnded(false)
    endedRef.current = false
    openingRequestedRef.current = false
    hadResponseRef.current = false
    openedRef.current = false
    reportedUnavailableRef.current = false
    wrappingUpRef.current = false
    turnCountRef.current = 0
    recallProbeIssuedRef.current = false
    recallAnsweredRef.current = false
    recallForcedRef.current = false
    closingRef.current = false
    voiceRef.current = voiceProfileForSession(language)

    setConnecting(true)
    updateStatus('processing', 'Connecting…')
    // Startup watchdog: if the SDP handshake / connection never completes, fall back to v2.
    connectTimerRef.current = setTimeout(() => {
      if (!openedRef.current && !hadResponseRef.current) { reportUnavailable('webrtc_connect_timeout'); cleanup() }
    }, CONNECT_TIMEOUT_MS)

    try {
      const token = await getBearer()
      const pc = new webrtc.RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
      })
      pcRef.current = pc

      pc.addEventListener('track', (e: any) => {
        // Remote assistant audio — libwebrtc plays it out automatically (with AEC on the mic path).
        if (e.streams && e.streams[0]) remoteStreamRef.current = e.streams[0]
      })
      pc.addEventListener('connectionstatechange', () => {
        const st = pc.connectionState
        if (st === 'failed' || st === 'disconnected') {
          if (!hadResponseRef.current) { reportUnavailable('webrtc_conn_failed'); cleanup() }
          else if (!wrappingUpRef.current) { setEnded(true) }
        }
      })

      const stream = await webrtc.mediaDevices.getUserMedia({ audio: true })
      localStreamRef.current = stream
      stream.getTracks().forEach((t: any) => pc.addTrack(t, stream))

      // Events channel (mirrors the OpenAI-compatible realtime WebRTC contract).
      const dc = pc.createDataChannel('oai-events')
      dcRef.current = dc
      dc.addEventListener('open', () => {
        openedRef.current = true
        setConnecting(false)
        setSessionActive(true)
        updateStatus('listening', 'Listening...')
        send(buildLiveSessionUpdate(patientId, voiceRef.current.languageLabel, { voice: voiceRef.current.voice, languageKey: voiceRef.current.languageKey, persona }))
        // Kick the opening turn even if session.created doesn't arrive first.
        if (!openingRequestedRef.current) { openingRequestedRef.current = true; send({ type: 'response.create' }) }
      })
      dc.addEventListener('message', (e: any) => {
        try { handleMessage(JSON.parse(String(e.data))) } catch {}
      })
      dc.addEventListener('close', () => {
        if (!endedRef.current && hadResponseRef.current && !reportedUnavailableRef.current) setEnded(true)
      })

      const offer = await pc.createOffer({})
      await pc.setLocalDescription(offer)
      await waitForIceGathering(pc)

      const res = await fetch(realtimeWebrtcUrl(), {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/sdp' },
        body: pc.localDescription?.sdp ?? offer.sdp,
      })
      if (!res.ok) {
        const detail = await res.text().catch(() => '')
        updateStatus('error', `WebRTC 握手失败 ${res.status}`)
        reportUnavailable(`webrtc_sdp_${res.status}`)
        // eslint-disable-next-line no-console
        console.warn('webrtc_sdp_failed', res.status, detail.slice(0, 300))
        cleanup()
        return
      }
      const answerSdp = await res.text()
      await pc.setRemoteDescription(new webrtc.RTCSessionDescription({ type: 'answer', sdp: answerSdp }))
    } catch (e: unknown) {
      if (!hadResponseRef.current) { reportUnavailable('webrtc_start_failed'); cleanup(); return }
      updateStatus('error', `Error: ${e instanceof Error ? e.message : 'unknown'}`)
    }
  }, [cleanup, handleMessage, language, patientId, persona, reportUnavailable, send, updateStatus, waitForIceGathering])

  const stopConversation = useCallback(() => {
    // Ask for a graceful goodbye if the session is live and we haven't started closing; otherwise tear down.
    if (dcRef.current?.readyState === 'open' && !closingRef.current && hadResponseRef.current && persona === 'screening') {
      requestGoodbye()
      updateStatus('idle', 'Wrapping up...')
      if (endTimerRef.current) clearTimeout(endTimerRef.current)
      endTimerRef.current = setTimeout(() => { setEnded(true); cleanup() }, 6000)
      return
    }
    cleanup()
    setEnded(true)
  }, [cleanup, persona, requestGoodbye, updateStatus])

  useEffect(() => cleanup, [cleanup])

  return {
    mode: 'webrtc',
    statusKind,
    statusText,
    messages,
    startConversation,
    stopConversation,
    connecting,
    sessionActive,
    userSpeaking,
    ended,
  }
}
