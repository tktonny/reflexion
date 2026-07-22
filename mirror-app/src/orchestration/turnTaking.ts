export type TurnTakingPhase =
  | 'idle'
  | 'connecting'
  | 'configuring'
  | 'listening'
  | 'user_speaking'
  | 'thinking'
  | 'assistant_generating'
  | 'assistant_playing'
  | 'playback_guard'
  | 'closing'
  | 'ended'
  | 'error'

export type TurnTakingEvent =
  | { type: 'reset' }
  | { type: 'connect_started' }
  | { type: 'session_configuring' }
  | { type: 'session_ready' }
  | { type: 'user_speech_started' }
  | { type: 'user_speech_stopped' }
  | { type: 'response_requested'; closing?: boolean }
  | { type: 'response_created' }
  | { type: 'audio_delta' }
  | { type: 'input_rejected' }
  | { type: 'response_done' }
  | { type: 'playback_drained' }
  | { type: 'mic_reopened' }
  | { type: 'close_requested' }
  | { type: 'finished' }
  | { type: 'failed'; reason: string }

export type TurnTakingState = {
  phase: TurnTakingPhase
  responseInFlight: boolean
  awaitingPlayback: boolean
  captureMuted: boolean
  closingRequested: boolean
  audioReceived: boolean
  sequence: number
  violations: string[]
}

export type PostPlaybackAction = 'listen' | 'steer_recall' | 'request_goodbye' | 'finish'

export type PostPlaybackContext = {
  persona: 'screening' | 'companion'
  closingResponse: boolean
  manualCloseRequested: boolean
  spontaneousGoodbye: boolean
  recallAnswered: boolean
  recallProbeIssued: boolean
  recallForced: boolean
  turnCount: number
  recallDeadlineTurn: number
  hardMaxTurn: number
}

export type PlaybackDrainDecision = {
  drained: boolean
  timedOut: boolean
}

export function createTurnTakingState(): TurnTakingState {
  return {
    phase: 'idle',
    responseInFlight: false,
    awaitingPlayback: false,
    captureMuted: true,
    closingRequested: false,
    audioReceived: false,
    sequence: 0,
    violations: [],
  }
}

function withViolation(state: TurnTakingState, message: string): TurnTakingState {
  return { ...state, sequence: state.sequence + 1, violations: [...state.violations, message] }
}

/**
 * Pure turn lifecycle reducer. Provider generation completion and device playback completion are
 * deliberately separate events: response_done can never reopen capture or advance the agenda.
 */
export function reduceTurnTaking(state: TurnTakingState, event: TurnTakingEvent): TurnTakingState {
  if (event.type === 'reset') return createTurnTakingState()
  const next = { ...state, sequence: state.sequence + 1 }

  switch (event.type) {
    case 'connect_started':
      return { ...next, phase: 'connecting', captureMuted: true }
    case 'session_configuring':
      return { ...next, phase: 'configuring', captureMuted: true }
    case 'session_ready':
      return { ...next, phase: 'listening', captureMuted: false }
    case 'user_speech_started':
      if (state.captureMuted || state.responseInFlight || state.awaitingPlayback) {
        return withViolation(state, 'user speech accepted while assistant owned the audio turn')
      }
      return { ...next, phase: 'user_speaking' }
    case 'user_speech_stopped':
      return { ...next, phase: 'thinking' }
    case 'response_requested':
      if (state.responseInFlight || state.awaitingPlayback) {
        return withViolation(state, 'a second response was requested before playback completed')
      }
      return {
        ...next,
        phase: event.closing ? 'closing' : 'assistant_generating',
        responseInFlight: true,
        captureMuted: true,
        closingRequested: state.closingRequested || Boolean(event.closing),
        audioReceived: false,
      }
    case 'response_created':
      if (state.awaitingPlayback) {
        return withViolation(state, 'provider created a response before previous playback completed')
      }
      return {
        ...next,
        phase: state.closingRequested ? 'closing' : 'assistant_generating',
        responseInFlight: true,
        captureMuted: true,
        audioReceived: false,
      }
    case 'audio_delta':
      if (!state.responseInFlight) return withViolation(state, 'audio arrived without an active response')
      return {
        ...next,
        phase: state.closingRequested ? 'closing' : 'assistant_playing',
        captureMuted: true,
        audioReceived: true,
      }
    case 'input_rejected':
      if (!state.responseInFlight) return withViolation(state, 'input was rejected without a pending turn')
      return {
        ...next,
        phase: state.closingRequested ? 'closing' : 'playback_guard',
        responseInFlight: false,
        awaitingPlayback: false,
        captureMuted: true,
        audioReceived: false,
      }
    case 'response_done':
      if (!state.responseInFlight) return withViolation(state, 'response_done arrived without an active response')
      return {
        ...next,
        phase: state.closingRequested ? 'closing' : 'playback_guard',
        responseInFlight: false,
        awaitingPlayback: true,
        captureMuted: true,
      }
    case 'playback_drained':
      if (!state.awaitingPlayback) return withViolation(state, 'playback drained without a completed response')
      return {
        ...next,
        phase: state.closingRequested ? 'closing' : 'playback_guard',
        awaitingPlayback: false,
        captureMuted: true,
      }
    case 'mic_reopened':
      if (state.responseInFlight || state.awaitingPlayback || state.closingRequested) {
        return withViolation(state, 'capture reopened before the assistant turn was safe')
      }
      return { ...next, phase: 'listening', captureMuted: false }
    case 'close_requested':
      return { ...next, phase: 'closing', captureMuted: true, closingRequested: true }
    case 'finished':
      return {
        ...next,
        phase: 'ended',
        responseInFlight: false,
        awaitingPlayback: false,
        captureMuted: true,
        closingRequested: true,
      }
    case 'failed':
      return {
        ...next,
        phase: 'error',
        responseInFlight: false,
        awaitingPlayback: false,
        captureMuted: true,
      }
  }
}

/** Decide what may happen only after device playback has drained. */
export function choosePostPlaybackAction(context: PostPlaybackContext): PostPlaybackAction {
  if (context.closingResponse) return 'finish'
  if (context.persona === 'companion' && context.spontaneousGoodbye) return 'finish'
  if (context.manualCloseRequested) return 'request_goodbye'
  if (context.persona === 'screening') {
    if (context.recallAnswered) return 'request_goodbye'
    const recallIsDue =
      context.turnCount >= context.recallDeadlineTurn || context.turnCount >= context.hardMaxTurn
    if (
      !context.recallProbeIssued &&
      !context.recallForced &&
      recallIsDue
    ) {
      return 'steer_recall'
    }
  }
  return 'listen'
}

/** A timeout is a failure signal, never permission to reopen the microphone. */
export function playbackDrainDecision(
  backlogMs: number,
  elapsedMs: number,
  thresholdMs: number,
  maxWaitMs: number,
): PlaybackDrainDecision {
  if (elapsedMs >= maxWaitMs) return { drained: false, timedOut: true }
  return { drained: Math.max(0, backlogMs) <= thresholdMs, timedOut: false }
}

export function replayTurnTaking(events: TurnTakingEvent[]): TurnTakingState {
  return events.reduce(reduceTurnTaking, createTurnTakingState())
}
