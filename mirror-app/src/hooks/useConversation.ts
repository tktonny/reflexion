import { useCallback, useRef, useState } from 'react'
import { Platform } from 'react-native'

import { CONVERSATION_MODE } from '../config/conversationMode'
import type { ConversationApi } from './conversationTypes'
import { useDirectRealtimeConversation } from './useDirectRealtimeConversation'
import { useWebRtcRealtimeConversation } from './useWebRtcRealtimeConversation'
import { useQwenRealtimeConversation } from './useQwenRealtimeConversation'
import { useTurnBasedConversation } from './useTurnBasedConversation'
import { useTurnBasedConversationNative } from './useTurnBasedConversationNative'

type Options = { patientId?: string; language?: string; persona?: 'screening' | 'companion'; pushToTalk?: boolean }

/**
 * Version selector. Picks the conversation implementation from EXPO_PUBLIC_CONVERSATION_MODE:
 *   relay -> 版本一 Node 中继 (useQwenRealtimeConversation)   [web + native]
 *   http  -> 版本二 回合制 HTTP (useTurnBasedConversation)    [web + native*]
 *   ws    -> 版本三 原生直连实时 WS + 自动降级                  [native; web falls back to relay]
 *
 * CONVERSATION_MODE and Platform.OS are build-time constants, so exactly one branch runs for the
 * app's lifetime — the conditional hook calls are stable (rules-of-hooks safe in practice).
 */
export function useConversation(opts: Options = {}): ConversationApi {
  /* eslint-disable react-hooks/rules-of-hooks */
  if (CONVERSATION_MODE === 'http') {
    // web uses the Web-Audio pipeline; native uses expo-audio (record file -> ASR).
    return Platform.OS === 'web' ? useTurnBasedConversation(opts) : useTurnBasedConversationNative(opts)
  }
  if (CONVERSATION_MODE === 'ws' && Platform.OS !== 'web') {
    // websocket-v0.0.0: omni realtime (v3) primary, turn-based (v2) automatic fallback.
    return useResilientConversation(opts)
  }
  if (CONVERSATION_MODE === 'webrtc' && Platform.OS !== 'web') {
    // webrtc-v0.0.0: native WebRTC realtime (built-in AEC). Direct, no auto-fallback in v0.0.0.
    return useWebRtcRealtimeConversation(opts)
  }
  // relay (default); also the web fallback for 'ws'/'webrtc' (browser WS/WebRTC-auth constraints).
  const api = useQwenRealtimeConversation(opts)
  return { ...api, mode: CONVERSATION_MODE === 'ws' || CONVERSATION_MODE === 'webrtc' ? CONVERSATION_MODE : 'relay' }
  /* eslint-enable react-hooks/rules-of-hooks */
}

/**
 * Omni-primary conversation with automatic runtime fallback (用户意图: omni 语音 → 失败 → vl+tts+plus).
 * Runs the direct-realtime (v3) hook as primary and the turn-based native (v2) hook as a warm standby.
 * If omni is unavailable at STARTUP — connect timeout, region block, handshake reject, or an error
 * before any response — v3 reports it via onUnavailable and we swap to v2, auto-starting it with the
 * same options (language carried through). The post-conversation screening (qwen-vl-max + qwen-plus)
 * is transport-agnostic, so it runs identically whichever engine produced the transcript.
 */
function useResilientConversation(opts: Options): ConversationApi {
  const [usingFallback, setUsingFallback] = useState(false)
  const fellBackRef = useRef(false)
  const swapRef = useRef<() => void>(() => {})

  // Stable callback handed to v3. It only records the intent + flips state; the actual fallback
  // start is delegated to swapRef, which is wired below once both child hooks exist.
  const onUnavailable = useCallback((_reason: string) => {
    if (fellBackRef.current) return
    fellBackRef.current = true
    setUsingFallback(true)
    swapRef.current()
  }, [])

  const primary = useDirectRealtimeConversation({ ...opts, onUnavailable })
  const fallback = useTurnBasedConversationNative(opts)

  // v3 already tears itself down at the failure point; here we just kick off the standby session.
  swapRef.current = () => { void fallback.startConversation() }

  const active = usingFallback ? fallback : primary
  return { ...active, mode: usingFallback ? 'http' : 'ws' }
}
