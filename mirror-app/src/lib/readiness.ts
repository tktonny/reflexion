import type { HardwareReport } from './hardwareCheck'

// Turns the self-check report into a DECISION plus elder-readable copy.
//
// Before this, the launch self-check only console.logged: a mirror with a denied microphone would show
// "Microphone ready ✓", start a check-in, capture nothing, and end with a generic failure — the elder
// had no idea why and the caregiver saw a missed check-in rather than a fixable device problem. The
// point of this module is that a blocking hardware problem must become something a person can act on.
//
// Wording rules (product): warm, plain, never clinical, never blame the elder, and always say what to
// do next. Copy is written for the person standing at the mirror, with a caregiver-facing hint under it.

export type BlockingProblem = 'microphone' | 'audio_module' | 'backend'

export type ReadinessVerdict = {
  /** True when a conversation would fail or capture nothing — do not start one. */
  blocked: boolean
  problem: BlockingProblem | null
  /** Large line for the elder. */
  title: string
  /** One supporting sentence. */
  body: string
  /** Label for the primary action, when the app itself can fix it. */
  actionLabel: string | null
}

const OK: ReadinessVerdict = { blocked: false, problem: null, title: '', body: '', actionLabel: null }

/**
 * Decide whether the mirror can hold a conversation right now.
 *
 * Deliberately NOT blocking on: camera (unused by the voice check-in), relay (dev-only transport),
 * speaker (a dead speaker is serious, but blocking on it would strand a mirror that can still listen —
 * it is surfaced to the caregiver through the heartbeat instead), and network (the boot flow already
 * has a real backend ping and its own offline screen).
 */
export function evaluateReadiness(report: HardwareReport | null): ReadinessVerdict {
  if (!report) return OK
  const byKey = Object.fromEntries(report.checks.map((c) => [c.key, c]))

  if (byKey.mic?.status === 'fail') {
    return {
      blocked: true,
      problem: 'microphone',
      title: 'Aria cannot hear you yet',
      body: 'The microphone is switched off for this app, so Aria would not hear your answers.',
      actionLabel: 'Turn on the microphone',
    }
  }

  if (byKey.rtaudio?.status === 'fail' && byKey.turnaudio?.status === 'fail') {
    // Both audio engines missing — nothing can carry a conversation on this build.
    return {
      blocked: true,
      problem: 'audio_module',
      title: 'This mirror needs an update',
      body: 'Its voice system is not installed. Your family can reinstall the Reflexion app to fix this.',
      actionLabel: null,
    }
  }

  return OK
}
