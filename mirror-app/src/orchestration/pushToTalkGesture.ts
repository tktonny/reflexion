export type PushToTalkGestureState = 'idle' | 'preparing' | 'recording'
export type PushToTalkRelease = 'ignored' | 'cancelled' | 'discard' | 'send'

type PushToTalkGestureOptions = {
  minimumRecordingMs?: number
  now?: () => number
}

/**
 * Small generation-guarded controller for a press-and-hold microphone gesture.
 *
 * Native recorder preparation is asynchronous. A user can release before preparation finishes, or
 * press again while the first preparation promise is still settling. Tokens make those stale
 * continuations harmless: only the latest held gesture may transition to `recording`.
 */
export function createPushToTalkGesture(options: PushToTalkGestureOptions = {}) {
  let state: PushToTalkGestureState = 'idle'
  let generation = 0
  let recordingStartedAt = 0
  const minimumRecordingMs = options.minimumRecordingMs ?? 0
  const now = options.now ?? Date.now

  return {
    begin(): number | null {
      if (state !== 'idle') return null
      generation += 1
      state = 'preparing'
      return generation
    },

    ready(token: number): boolean {
      if (token !== generation || state !== 'preparing') return false
      state = 'recording'
      recordingStartedAt = now()
      return true
    },

    release(): PushToTalkRelease {
      if (state === 'idle') return 'ignored'
      generation += 1
      if (state === 'preparing') {
        state = 'idle'
        return 'cancelled'
      }
      const heldForMs = now() - recordingStartedAt
      state = 'idle'
      recordingStartedAt = 0
      return heldForMs < minimumRecordingMs ? 'discard' : 'send'
    },

    cancel(token?: number): void {
      if (token !== undefined && token !== generation) return
      generation += 1
      state = 'idle'
      recordingStartedAt = 0
    },

    reset(): void {
      generation += 1
      state = 'idle'
      recordingStartedAt = 0
    },

    getState(): PushToTalkGestureState {
      return state
    },
  }
}
