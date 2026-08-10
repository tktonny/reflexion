import AsyncStorage from '@react-native-async-storage/async-storage'

import { abandonMirrorSessionById, uploadPendingSessionCompletion, type PendingSessionCompletion } from '../api/sessionSync'

const PENDING_CONVERSATIONS_STORAGE_KEY = 'reflexion:pendingV1SessionCompletions'
const PARKED_CONVERSATIONS_STORAGE_KEY = 'reflexion:parkedV1SessionCompletions'

// A queued session-completion is retried with exponential backoff and is PARKED (not retried, not
// lost) after MAX_ATTEMPTS so a permanently-unsendable payload can never grow the outbox forever
// (implementation baseline §7 Phase 7 — the poison-queue fix). An entry can also represent an
// abandon (a session that produced no transcript) so a zero-transcript session is never queued as an
// uncompletable completion.
const MAX_ATTEMPTS = 8
const BASE_BACKOFF_MS = 60_000
const MAX_BACKOFF_MS = 6 * 60 * 60 * 1000

type QueueEntry = {
  kind: 'complete' | 'abandon'
  sessionId: string
  payload?: PendingSessionCompletion
  reason?: string
  idempotencyKey?: string
  attempts: number
  nextAttemptAt: number
  firstQueuedAt: number
}

function nowMs() { return Date.now() }
function backoffMs(attempts: number) { return Math.min(BASE_BACKOFF_MS * 2 ** attempts, MAX_BACKOFF_MS) }

// Tolerates the legacy shape (a bare PendingSessionCompletion[]) by wrapping each as a complete entry.
function normalizeEntries(parsed: unknown): QueueEntry[] {
  if (!Array.isArray(parsed)) return []
  return parsed.map((item): QueueEntry | null => {
    if (item && typeof item === 'object' && 'kind' in item && 'sessionId' in item) return item as QueueEntry
    if (item && typeof item === 'object' && 'events' in item && 'sessionId' in item) {
      const payload = item as PendingSessionCompletion
      return { kind: 'complete', sessionId: payload.sessionId, payload, attempts: 0, nextAttemptAt: 0, firstQueuedAt: nowMs() }
    }
    return null
  }).filter((entry): entry is QueueEntry => entry != null)
}

async function loadEntries(key = PENDING_CONVERSATIONS_STORAGE_KEY): Promise<QueueEntry[]> {
  const raw = await AsyncStorage.getItem(key)
  if (!raw) return []
  try {
    return normalizeEntries(JSON.parse(raw))
  } catch {
    await AsyncStorage.removeItem(key)
    return []
  }
}

async function saveEntries(entries: QueueEntry[], key = PENDING_CONVERSATIONS_STORAGE_KEY) {
  if (entries.length) await AsyncStorage.setItem(key, JSON.stringify(entries))
  else await AsyncStorage.removeItem(key)
}

/** Legacy-compatible read: the pending completion payloads still waiting to upload. */
export async function loadPendingConversations(): Promise<PendingSessionCompletion[]> {
  return (await loadEntries()).filter((entry) => entry.kind === 'complete' && entry.payload).map((entry) => entry.payload!)
}

export async function queuePendingConversation(payload: PendingSessionCompletion) {
  const entries = await loadEntries()
  if (!entries.some((entry) => entry.sessionId === payload.sessionId)) {
    entries.push({ kind: 'complete', sessionId: payload.sessionId, payload, attempts: 0, nextAttemptAt: 0, firstQueuedAt: nowMs() })
  }
  await saveEntries(entries)
  return entries.length
}

/** Queue a durable abandon for a session that produced no transcript, instead of an unsendable payload. */
export async function queueAbandon(sessionId: string, reason: string, idempotencyKey: string) {
  const entries = await loadEntries()
  if (!entries.some((entry) => entry.sessionId === sessionId)) {
    entries.push({ kind: 'abandon', sessionId, reason, idempotencyKey, attempts: 0, nextAttemptAt: 0, firstQueuedAt: nowMs() })
  }
  await saveEntries(entries)
  return entries.length
}

export async function clearPendingConversations() {
  await AsyncStorage.removeItem(PENDING_CONVERSATIONS_STORAGE_KEY)
}

async function runEntry(entry: QueueEntry): Promise<void> {
  if (entry.kind === 'abandon') {
    await abandonMirrorSessionById(entry.sessionId, entry.reason || 'transport_failed', entry.idempotencyKey)
    return
  }
  if (entry.payload) await uploadPendingSessionCompletion(entry.payload)
}

/**
 * Drains due entries. Failures are rescheduled with exponential backoff; entries that exhaust
 * MAX_ATTEMPTS are moved to a parked store (retained for diagnostics, never retried). Safe to call
 * on app start and on network reconnect.
 */
export async function flushPendingConversations() {
  const entries = await loadEntries()
  if (!entries.length) return { synced: 0, remaining: 0, parked: 0 }
  const now = nowMs()
  const remaining: QueueEntry[] = []
  const parked: QueueEntry[] = []
  let synced = 0
  for (const entry of entries) {
    if (entry.nextAttemptAt > now) { remaining.push(entry); continue }
    try {
      await runEntry(entry)
      synced++
    } catch {
      const attempts = entry.attempts + 1
      if (attempts >= MAX_ATTEMPTS) parked.push({ ...entry, attempts })
      else remaining.push({ ...entry, attempts, nextAttemptAt: now + backoffMs(attempts) })
    }
  }
  await saveEntries(remaining)
  if (parked.length) {
    const existingParked = await loadEntries(PARKED_CONVERSATIONS_STORAGE_KEY)
    await saveEntries([...existingParked, ...parked], PARKED_CONVERSATIONS_STORAGE_KEY)
    console.warn(`[conversationQueue] parked ${parked.length} unsendable session(s) after ${MAX_ATTEMPTS} attempts`)
  }
  return { synced, remaining: remaining.length, parked: parked.length }
}
