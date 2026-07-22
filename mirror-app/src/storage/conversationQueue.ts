import AsyncStorage from '@react-native-async-storage/async-storage'

import { uploadPendingSessionCompletion, type PendingSessionCompletion } from '../api/sessionSync'

const PENDING_CONVERSATIONS_STORAGE_KEY = 'reflexion:pendingV1SessionCompletions'

export async function loadPendingConversations(): Promise<PendingSessionCompletion[]> {
  const raw = await AsyncStorage.getItem(PENDING_CONVERSATIONS_STORAGE_KEY)
  if (!raw) return []
  try {
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed as PendingSessionCompletion[] : []
  } catch {
    await AsyncStorage.removeItem(PENDING_CONVERSATIONS_STORAGE_KEY)
    return []
  }
}

export async function queuePendingConversation(payload: PendingSessionCompletion) {
  const pending = await loadPendingConversations()
  if (!pending.some((item) => item.sessionId === payload.sessionId)) pending.push(payload)
  await AsyncStorage.setItem(PENDING_CONVERSATIONS_STORAGE_KEY, JSON.stringify(pending))
  return pending.length
}

export async function clearPendingConversations() {
  await AsyncStorage.removeItem(PENDING_CONVERSATIONS_STORAGE_KEY)
}

export async function flushPendingConversations() {
  const pending = await loadPendingConversations()
  const remaining: PendingSessionCompletion[] = []
  let synced = 0
  for (const payload of pending) {
    try { await uploadPendingSessionCompletion(payload); synced++ } catch { remaining.push(payload) }
  }
  if (remaining.length) await AsyncStorage.setItem(PENDING_CONVERSATIONS_STORAGE_KEY, JSON.stringify(remaining))
  else await AsyncStorage.removeItem(PENDING_CONVERSATIONS_STORAGE_KEY)
  return { synced, remaining: remaining.length }
}
