export type PlaybackCompletionStatus = {
  currentTime?: number
  didJustFinish?: boolean
  duration?: number
  error?: string | null
  isLoaded?: boolean
  playing?: boolean
}

export type PlaybackCompletionPlayer = {
  addListener: (
    event: 'playbackStatusUpdate',
    listener: (status: PlaybackCompletionStatus) => void,
  ) => { remove: () => void }
  play: () => void
}

/**
 * Start one expo-audio player and resolve only after the native player reports completion.
 * The timeout is a failure bound; it is never treated as evidence that playback finished.
 */
export function playAndWaitForCompletion(
  player: PlaybackCompletionPlayer,
  timeoutMs = 45_000,
  signal?: AbortSignal,
): Promise<void> {
  return new Promise((resolve, reject) => {
    let settled = false
    let playbackStarted = false
    let subscription: { remove: () => void } | null = null
    let timeoutId: ReturnType<typeof setTimeout> | null = null
    let abort = () => {}

    const finish = (error?: Error) => {
      if (settled) return
      settled = true
      if (timeoutId) clearTimeout(timeoutId)
      subscription?.remove()
      signal?.removeEventListener('abort', abort)
      if (error) reject(error)
      else resolve()
    }

    abort = () => finish(new Error('Audio playback was cancelled by a newer conversation.'))

    timeoutId = setTimeout(() => {
      finish(new Error('Audio playback did not report completion before the safety timeout.'))
    }, timeoutMs)

    subscription = player.addListener('playbackStatusUpdate', (status) => {
      if (status.error) {
        finish(new Error(`Audio playback failed: ${status.error}`))
        return
      }
      if (status.playing || (status.currentTime ?? 0) > 0) playbackStarted = true
      const reachedDuration =
        playbackStarted &&
        status.isLoaded !== false &&
        !status.playing &&
        (status.duration ?? 0) > 0 &&
        (status.currentTime ?? 0) >= Math.max(0, (status.duration ?? 0) - 0.05)
      if (status.didJustFinish || reachedDuration) finish()
    })

    if (signal?.aborted) {
      abort()
      return
    }
    signal?.addEventListener('abort', abort, { once: true })

    try {
      player.play()
    } catch (error) {
      finish(error instanceof Error ? error : new Error('Audio playback could not start.'))
    }
  })
}
