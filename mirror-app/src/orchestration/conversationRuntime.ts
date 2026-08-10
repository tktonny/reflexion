export type ConversationRuntimeLease = {
  ownerId: string
  generation: number
  isCurrent: () => boolean
  release: () => void
}

type RuntimeOwner = {
  ownerId: string
  generation: number
  terminate: () => void
}

let activeOwner: RuntimeOwner | null = null
let generationSequence = 0

/**
 * Acquire the process-wide microphone/speaker lease. A new owner synchronously terminates the old
 * owner before it becomes active, so Expo Router screens and transport fallback cannot play at once.
 */
export function acquireConversationRuntime(
  ownerId: string,
  terminate: () => void,
): ConversationRuntimeLease {
  const previous = activeOwner
  activeOwner = null
  if (previous) {
    if (typeof __DEV__ !== 'undefined' && __DEV__) {
      console.info(`[conversation-runtime] superseding ${previous.ownerId} -> ${ownerId}`)
    }
    try { previous.terminate() } catch {}
  }

  const generation = generationSequence + 1
  generationSequence = generation
  activeOwner = { ownerId, generation, terminate }

  const isCurrent = () =>
    activeOwner?.ownerId === ownerId && activeOwner.generation === generation
  const release = () => {
    if (isCurrent()) {
      activeOwner = null
      if (typeof __DEV__ !== 'undefined' && __DEV__) {
        console.info(`[conversation-runtime] released ${ownerId}`)
      }
    }
  }
  return { ownerId, generation, isCurrent, release }
}
