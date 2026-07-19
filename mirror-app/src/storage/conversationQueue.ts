import AsyncStorage from '@react-native-async-storage/async-storage'

import {
  saveConversation,
  type SaveConversationInput,
} from '../api/conversation'

const PENDING_CONVERSATIONS_STORAGE_KEY = 'reflexion:pendingConversations'

export async function loadPendingConversations(): Promise<SaveConversationInput[]> {
  const raw = await AsyncStorage.getItem(PENDING_CONVERSATIONS_STORAGE_KEY)
  if (!raw) return []

  try {
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? (parsed as SaveConversationInput[]) : []
  } catch {
    await AsyncStorage.removeItem(PENDING_CONVERSATIONS_STORAGE_KEY)
    return []
  }
}

export async function queuePendingConversation(payload: SaveConversationInput) {
  const pending = await loadPendingConversations()
  if (payload.clientSessionId && pending.some((item) => item.clientSessionId === payload.clientSessionId)) {
    return pending.length
  }

  pending.push(payload)
  await AsyncStorage.setItem(PENDING_CONVERSATIONS_STORAGE_KEY, JSON.stringify(pending))
  return pending.length
}

export async function clearPendingConversations() {
  await AsyncStorage.removeItem(PENDING_CONVERSATIONS_STORAGE_KEY)
}

export async function flushPendingConversations() {
  const pending = await loadPendingConversations()
  if (pending.length === 0) return { synced: 0, remaining: 0 }

  const remaining: SaveConversationInput[] = []
  let synced = 0

  for (const payload of pending) {
    try {
      const result = await saveConversation(payload)
      if (result.success) {
        synced += 1
      } else {
        remaining.push(payload)
      }
    } catch {
      remaining.push(payload)
    }
  }

  if (remaining.length > 0) {
    await AsyncStorage.setItem(PENDING_CONVERSATIONS_STORAGE_KEY, JSON.stringify(remaining))
  } else {
    await AsyncStorage.removeItem(PENDING_CONVERSATIONS_STORAGE_KEY)
  }

  return { synced, remaining: remaining.length }
}
