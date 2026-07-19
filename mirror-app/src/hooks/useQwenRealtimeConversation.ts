import { useCallback, useEffect, useRef, useState } from 'react'
import { Platform } from 'react-native'

import {
  CAPTURE_SAMPLE_RATE,
  PLAYBACK_SAMPLE_RATE,
  getRealtimeWsUrl,
} from '../constants/realtime'
import { randomId } from '../utils/id'

export type ChatRole = 'system' | 'user' | 'assistant'
export type StatusKind = 'idle' | 'listening' | 'processing' | 'speaking' | 'error'

export type ChatMessage = {
  id: string
  role: ChatRole
  text: string
  streaming?: boolean
}

type Options = {
  patientId?: string
  language?: string
}

/**
 * Realtime voice conversation against the Qwen Omni relay (server/*.mjs), replacing
 * the old OpenAI WebRTC hook. Web-first: uses the browser Web Audio pipeline ported
 * from REFLEXION's clinic browser client (convertTo16kPcm capture, gapless 24k
 * playback, half-duplex capture hold). Native audio (MirrorAudio module) is TODO.
 */
export function useQwenRealtimeConversation(options: Options = {}) {
  const patientId = options.patientId ?? 'demo-patient'
  const [language] = useState(options.language ?? 'en')

  const [statusKind, setStatusKind] = useState<StatusKind>('idle')
  const [statusText, setStatusText] = useState('Ready to start')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [connecting, setConnecting] = useState(false)
  const [sessionActive, setSessionActive] = useState(false)
  const [userSpeaking, setUserSpeaking] = useState(false)

  const socketRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<any>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const mediaSourceRef = useRef<any>(null)
  const processorRef = useRef<any>(null)
  const muteGainRef = useRef<any>(null)
  const isRecordingRef = useRef(false)
  const sessionReadyRef = useRef(false)
  const playbackCursorRef = useRef(0)
  const holdUntilRef = useRef(0)
  const assistantStreamIdRef = useRef<string | null>(null)
  const assistantTextRef = useRef('')

  const updateStatus = useCallback((kind: StatusKind, text: string) => {
    setStatusKind(kind)
    setStatusText(text)
  }, [])

  const appendAssistantStreaming = useCallback((text: string) => {
    if (!assistantStreamIdRef.current) {
      const id = randomId('assistant')
      assistantStreamIdRef.current = id
      setMessages((prev) => [...prev, { id, role: 'assistant', text, streaming: true }])
      return
    }
    const id = assistantStreamIdRef.current
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, text } : m)))
  }, [])

  const finalizeAssistant = useCallback((text: string) => {
    const id = assistantStreamIdRef.current
    assistantStreamIdRef.current = null
    assistantTextRef.current = ''
    const clean = text.trim()
    if (!clean) return
    if (id) setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, text: clean, streaming: false } : m)))
    else setMessages((prev) => [...prev, { id: randomId('assistant'), role: 'assistant', text: clean }])
  }, [])

  const shouldSuppressCapture = useCallback(() => {
    const ctx = audioContextRef.current
    if (!ctx) return false
    return ctx.currentTime < holdUntilRef.current
  }, [])

  const playAssistantAudio = useCallback((base64Audio: string) => {
    const ctx = audioContextRef.current
    if (!ctx || !base64Audio) return
    const bytes = base64ToBytes(base64Audio)
    const pcm16 = new Int16Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))
    if (pcm16.length === 0) return

    const audioBuffer = ctx.createBuffer(1, pcm16.length, PLAYBACK_SAMPLE_RATE)
    const channel = audioBuffer.getChannelData(0)
    for (let i = 0; i < pcm16.length; i += 1) channel[i] = pcm16[i] / 0x8000

    const source = ctx.createBufferSource()
    source.buffer = audioBuffer
    source.connect(ctx.destination)
    const startAt = Math.max(ctx.currentTime, playbackCursorRef.current || 0)
    source.start(startAt)
    playbackCursorRef.current = startAt + audioBuffer.duration
    holdUntilRef.current = Math.max(holdUntilRef.current || 0, playbackCursorRef.current + 0.12)
  }, [])

  const handleMessage = useCallback(
    (payload: any) => {
      const type = String(payload?.type || '')
      switch (type) {
        case 'reflexion.session.ready':
          sessionReadyRef.current = true
          setSessionActive(true)
          setConnecting(false)
          isRecordingRef.current = true
          updateStatus('listening', 'Listening...')
          break
        case 'reflexion.session.degraded':
          updateStatus('error', `Live relay unavailable: ${payload?.reason || 'unknown'}`)
          break
        case 'reflexion.voice.selected':
          updateStatus('listening', `Listening... (${payload?.language || ''})`)
          break
        case 'input_audio_buffer.speech_started':
          setUserSpeaking(true)
          updateStatus('listening', 'Listening...')
          break
        case 'input_audio_buffer.speech_stopped':
          setUserSpeaking(false)
          updateStatus('processing', 'Thinking...')
          break
        case 'conversation.item.input_audio_transcription.completed': {
          const transcript = String(payload?.transcript || '').trim()
          if (transcript) setMessages((prev) => [...prev, { id: randomId('user'), role: 'user', text: transcript }])
          break
        }
        case 'response.created':
          updateStatus('speaking', 'Speaking...')
          break
        case 'response.audio.delta':
          playAssistantAudio(String(payload?.delta || ''))
          updateStatus('speaking', 'Speaking...')
          break
        case 'response.audio_transcript.delta':
        case 'response.text.delta':
          assistantTextRef.current += String(payload?.delta || '')
          appendAssistantStreaming(assistantTextRef.current)
          break
        case 'response.audio_transcript.done':
        case 'response.text.done':
          finalizeAssistant(String(payload?.transcript ?? payload?.text ?? assistantTextRef.current))
          break
        case 'response.done':
          updateStatus('listening', 'Listening...')
          break
        default:
          break
      }
    },
    [appendAssistantStreaming, finalizeAssistant, playAssistantAudio, updateStatus],
  )

  const cleanup = useCallback(() => {
    isRecordingRef.current = false
    sessionReadyRef.current = false
    try { processorRef.current?.disconnect() } catch {}
    try { mediaSourceRef.current?.disconnect() } catch {}
    try { muteGainRef.current?.disconnect() } catch {}
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop())
    try { audioContextRef.current?.close() } catch {}
    processorRef.current = null
    mediaSourceRef.current = null
    muteGainRef.current = null
    mediaStreamRef.current = null
    audioContextRef.current = null
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      try { socketRef.current.send(JSON.stringify({ type: 'reflexion.close' })) } catch {}
    }
    try { socketRef.current?.close() } catch {}
    socketRef.current = null
    playbackCursorRef.current = 0
    holdUntilRef.current = 0
    assistantStreamIdRef.current = null
    assistantTextRef.current = ''
    setSessionActive(false)
    setConnecting(false)
    setUserSpeaking(false)
  }, [])

  const initAudioPipeline = useCallback((stream: MediaStream) => {
    const w = window as any
    const AudioContextCtor = w.AudioContext || w.webkitAudioContext
    const ctx = new AudioContextCtor()
    audioContextRef.current = ctx
    void ctx.resume?.()

    const source = ctx.createMediaStreamSource(stream)
    const processor = ctx.createScriptProcessor(4096, 1, 1)
    const muteGain = ctx.createGain()
    muteGain.gain.value = 0
    mediaSourceRef.current = source
    processorRef.current = processor
    muteGainRef.current = muteGain

    processor.onaudioprocess = (event: any) => {
      const socket = socketRef.current
      if (
        !isRecordingRef.current ||
        !sessionReadyRef.current ||
        !socket ||
        socket.readyState !== WebSocket.OPEN ||
        shouldSuppressCapture()
      ) {
        return
      }
      const input = event.inputBuffer.getChannelData(0)
      const pcm16 = convertTo16kPcm(input, ctx.sampleRate)
      if (!pcm16 || pcm16.length === 0) return
      socket.send(JSON.stringify({
        type: 'input_audio_buffer.append',
        audio: bytesToBase64(new Uint8Array(pcm16.buffer)),
      }))
    }

    source.connect(processor)
    processor.connect(muteGain)
    muteGain.connect(ctx.destination)
  }, [shouldSuppressCapture])

  const startConversation = useCallback(async () => {
    if (Platform.OS !== 'web') {
      updateStatus('error', 'Realtime audio currently runs on web only (native MirrorAudio module is TODO).')
      return
    }
    if (socketRef.current) return
    setConnecting(true)
    updateStatus('processing', 'Connecting...')
    setMessages([])

    try {
      const stream = await (navigator as any).mediaDevices.getUserMedia({ audio: true, video: false })
      mediaStreamRef.current = stream

      const url = getRealtimeWsUrl(patientId, language)
      const socket = new WebSocket(url)
      socketRef.current = socket

      socket.onopen = () => {
        initAudioPipeline(stream)
      }
      socket.onmessage = (event) => {
        try { handleMessage(JSON.parse(String(event.data))) } catch { /* ignore malformed */ }
      }
      socket.onerror = () => {
        updateStatus('error', 'Relay connection error. Is the relay running on :8787?')
      }
      socket.onclose = () => {
        if (sessionReadyRef.current) updateStatus('idle', 'Conversation ended')
        cleanup()
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error'
      cleanup()
      updateStatus('error', `Error: ${message}`)
    }
  }, [cleanup, handleMessage, initAudioPipeline, language, patientId, updateStatus])

  const stopConversation = useCallback(async () => {
    cleanup()
    updateStatus('idle', 'Conversation ended')
  }, [cleanup, updateStatus])

  useEffect(() => cleanup, [cleanup])

  return {
    statusKind,
    statusText,
    messages,
    startConversation,
    stopConversation,
    connecting,
    sessionActive,
    userSpeaking,
    language,
  }
}

function convertTo16kPcm(channelData: Float32Array, inputSampleRate: number): Int16Array | null {
  if (!channelData?.length) return null
  const ratio = inputSampleRate / CAPTURE_SAMPLE_RATE
  const length = Math.max(1, Math.round(channelData.length / ratio))
  const pcm = new Int16Array(length)
  let offsetResult = 0
  let offsetBuffer = 0
  while (offsetResult < pcm.length) {
    const nextOffsetBuffer = Math.min(channelData.length, Math.round((offsetResult + 1) * ratio))
    let acc = 0
    let count = 0
    for (let i = offsetBuffer; i < nextOffsetBuffer; i += 1) { acc += channelData[i]; count += 1 }
    const sample = count > 0 ? acc / count : channelData[offsetBuffer] || 0
    const clamped = Math.max(-1, Math.min(1, sample))
    pcm[offsetResult] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff
    offsetResult += 1
    offsetBuffer = nextOffsetBuffer
  }
  return pcm
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = ''
  const chunkSize = 0x8000
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize))
  }
  return (globalThis as any).btoa(binary)
}

function base64ToBytes(base64: string): Uint8Array {
  const binary = (globalThis as any).atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)
  return bytes
}
