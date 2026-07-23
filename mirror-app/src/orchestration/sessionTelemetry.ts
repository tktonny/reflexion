// Per-turn + per-session telemetry recorder for the mirror (implementation baseline §3.2/§3.3, doc
// "Signal-to-Status Algorithm" §3 raw signals). The mirror computes NO status — it only observes and
// reports timing. This recorder is a pure state machine driven by one-liner calls from the realtime
// hook at existing lifecycle points; it never influences turn-taking control flow. Clocks are passed
// in (epoch ms) so it is unit-testable and deterministic.

export type TurnTelemetry = {
  turnId: string
  role: 'patient' | 'assistant'
  questionId?: string
  startedAt: string
  endedAt: string
  speechMs?: number
  ariaPromptEndAt?: string
  userSpeechStartAt?: string
  responseLatencyMs?: number | null
  firstAudioAt?: string
  responseDoneAt?: string
  playbackFinishedAt?: string
  interrupted?: boolean
  interruptionReason?: string
}

export type SessionTelemetry = {
  startedAt: string
  endedAt: string
  patientSpeechMs: number
  ariaSpeechMs: number
  patientTurns: number
  ariaTurns: number
  repromptCount: number
  reconnectCount: number
  medianResponseLatencyMs: number | null
  turns: TurnTelemetry[]
}

const iso = (ms: number) => new Date(ms).toISOString()

export function createSessionTelemetry() {
  let patientSpeechMs = 0
  let ariaSpeechMs = 0
  let patientTurns = 0
  let ariaTurns = 0
  let repromptCount = 0
  let reconnectCount = 0
  const latencies: number[] = []
  const turns: TurnTelemetry[] = []

  let userSpeechStartAt: number | null = null
  let lastAriaPlaybackEndAt: number | null = null
  let aStart: number | null = null
  let aFirstAudio: number | null = null
  let aDone: number | null = null
  let aInterrupted = false
  let aReason: string | undefined

  return {
    reset() {
      patientSpeechMs = 0; ariaSpeechMs = 0; patientTurns = 0; ariaTurns = 0; repromptCount = 0; reconnectCount = 0
      latencies.length = 0; turns.length = 0
      userSpeechStartAt = null; lastAriaPlaybackEndAt = null
      aStart = null; aFirstAudio = null; aDone = null; aInterrupted = false; aReason = undefined
    },
    onUserSpeechStart(now: number) { userSpeechStartAt = now },
    onReprompt() { repromptCount += 1 },
    onReconnect() { reconnectCount += 1 },
    // A user turn is finalized when its transcript arrives. `now` is that moment (end of the utterance).
    onUserTurn(questionId: string | undefined, now: number) {
      const start = userSpeechStartAt ?? now
      const speechMs = Math.max(0, now - start)
      // A no-response turn contributes no latency (doc §10.5) — only count when we saw both edges.
      const latency = lastAriaPlaybackEndAt != null && userSpeechStartAt != null
        ? Math.max(0, userSpeechStartAt - lastAriaPlaybackEndAt) : null
      patientTurns += 1
      patientSpeechMs += speechMs
      if (latency != null) latencies.push(latency)
      turns.push({
        turnId: `p${patientTurns}`, role: 'patient', questionId,
        startedAt: iso(start), endedAt: iso(now), speechMs,
        ariaPromptEndAt: lastAriaPlaybackEndAt != null ? iso(lastAriaPlaybackEndAt) : undefined,
        userSpeechStartAt: iso(start), responseLatencyMs: latency,
      })
      userSpeechStartAt = null
    },
    onAriaResponseCreated(now: number) { aStart = now; aFirstAudio = null; aDone = null; aInterrupted = false; aReason = undefined },
    onAriaFirstAudio(now: number) { if (aFirstAudio == null) aFirstAudio = now },
    onAriaResponseDone(now: number) { aDone = now },
    onInterrupt(reason: string) { aInterrupted = true; aReason = reason },
    onAriaPlaybackFinished(now: number) {
      ariaTurns += 1
      const spokeMs = aFirstAudio != null ? Math.max(0, now - aFirstAudio) : 0
      ariaSpeechMs += spokeMs
      lastAriaPlaybackEndAt = now
      turns.push({
        turnId: `a${ariaTurns}`, role: 'assistant',
        startedAt: iso(aStart ?? now), endedAt: iso(now), speechMs: spokeMs,
        firstAudioAt: aFirstAudio != null ? iso(aFirstAudio) : undefined,
        responseDoneAt: aDone != null ? iso(aDone) : undefined, playbackFinishedAt: iso(now),
        interrupted: aInterrupted, interruptionReason: aReason,
      })
      aStart = null; aFirstAudio = null; aDone = null; aInterrupted = false; aReason = undefined
    },
    snapshot(startedAtMs: number, endedAtMs: number): SessionTelemetry {
      const sorted = [...latencies].sort((a, b) => a - b)
      const median = sorted.length
        ? sorted.length % 2 ? sorted[(sorted.length - 1) / 2] : Math.round((sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2)
        : null
      return {
        startedAt: iso(startedAtMs), endedAt: iso(endedAtMs),
        patientSpeechMs, ariaSpeechMs, patientTurns, ariaTurns, repromptCount, reconnectCount,
        medianResponseLatencyMs: median, turns: [...turns],
      }
    },
  }
}

export type SessionTelemetryRecorder = ReturnType<typeof createSessionTelemetry>
