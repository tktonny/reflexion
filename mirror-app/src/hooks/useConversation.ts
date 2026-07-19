import { Platform } from 'react-native'

import { CONVERSATION_MODE } from '../config/conversationMode'
import type { ConversationApi } from './conversationTypes'
import { useDirectRealtimeConversation } from './useDirectRealtimeConversation'
import { useQwenRealtimeConversation } from './useQwenRealtimeConversation'
import { useTurnBasedConversation } from './useTurnBasedConversation'

type Options = { patientId?: string; language?: string }

/**
 * Version selector. Picks the conversation implementation from EXPO_PUBLIC_CONVERSATION_MODE:
 *   relay -> 版本一 Node 中继 (useQwenRealtimeConversation)   [web + native]
 *   http  -> 版本二 回合制 HTTP (useTurnBasedConversation)    [web + native*]
 *   ws    -> 版本三 原生直连实时 WS                            [native; falls back to relay for now]
 *
 * CONVERSATION_MODE is a build-time constant, so exactly one hook is used for the app's
 * lifetime — the conditional hook call is stable (rules-of-hooks safe in practice).
 */
export function useConversation(opts: Options = {}): ConversationApi {
  // CONVERSATION_MODE and Platform.OS are build-time constants → exactly one hook is used for
  // the app's lifetime, so the conditional hook calls are stable.
  /* eslint-disable react-hooks/rules-of-hooks */
  if (CONVERSATION_MODE === 'http') {
    return useTurnBasedConversation(opts)
  }
  if (CONVERSATION_MODE === 'ws' && Platform.OS !== 'web') {
    return useDirectRealtimeConversation(opts)
  }
  // relay (default); also the web fallback for 'ws' (browser WS can't set the auth header).
  const api = useQwenRealtimeConversation(opts)
  return { ...api, mode: CONVERSATION_MODE === 'ws' ? 'ws' : 'relay' }
  /* eslint-enable react-hooks/rules-of-hooks */
}
