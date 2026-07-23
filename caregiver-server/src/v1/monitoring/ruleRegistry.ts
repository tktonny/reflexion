// Versioned, single-source thresholds for the deterministic caregiver reassurance status engine.
// Implementation baseline §2.4 + the doc's "Reflexion Signal-to-Status Algorithm" and "Updated
// Metrics". Every threshold lives here so it can be tuned without editing engine logic, and (later)
// overlaid from the `ruleRegistry` collection without a code deploy. This module is pure and imports
// nothing from the database.

export type ReassuranceRules = {
  ruleVersion: string
  baseline: {
    windowDays: number
    minSessions: number
    recomputeEveryDays: number
    ewmaAlpha: number
    // Section 8.4 guardrails: a metric whose baseline is below its minimum is suppressed (neutral).
    guardrails: { patientSpeechMs: number; patientTurns: number; medianResponseLatencyMs: number; wordCount: number }
  }
  // M4 speech duration, M6 turn count — lower is worse (ratio = today / baseline median).
  speechDuration: { amberBelow: number; redBelow: number; redConsecutive: number }
  turnCount: { amberBelow: number; redBelow: number; redConsecutive: number }
  // M7 response latency — higher is worse (ratio = today / baseline median).
  latency: { amberAbove: number; redAbove: number; redConsecutive: number }
  // M3 routine window (minutes from midnight). window = median ± max(k·MAD, floor), capped at ±cap.
  routineWindow: { madMultiplier: number; floorMinutes: number; capMinutes: number; unusualStartMin: number; unusualEndMin: number; amberOffWindowDays: number }
  // M5 weekly engagement frequency (freq_ratio = sessions_7d / max(baselineWeekly, floor)).
  weeklyEngagement: { floorSessions: number; amberBelow: number; redBelow: number }
  // M1/M2 completion + missed-day streak.
  missedStreak: { amberAt: number; redAt: number }
  // Device technical separation.
  technical: { unreachableAfterMinutes: number }
}

export const REASSURANCE_RULES: ReassuranceRules = {
  ruleVersion: 'reassurance-rules-v1',
  baseline: {
    windowDays: 14,
    minSessions: 7,
    recomputeEveryDays: 7,
    ewmaAlpha: 0.1,
    guardrails: { patientSpeechMs: 30_000, patientTurns: 3, medianResponseLatencyMs: 1, wordCount: 20 },
  },
  speechDuration: { amberBelow: 0.7, redBelow: 0.5, redConsecutive: 3 },
  turnCount: { amberBelow: 0.7, redBelow: 0.5, redConsecutive: 3 },
  latency: { amberAbove: 1.5, redAbove: 2.0, redConsecutive: 3 },
  routineWindow: { madMultiplier: 2, floorMinutes: 90, capMinutes: 180, unusualStartMin: 22 * 60, unusualEndMin: 5 * 60, amberOffWindowDays: 2 },
  weeklyEngagement: { floorSessions: 3, amberBelow: 0.8, redBelow: 0.5 },
  missedStreak: { amberAt: 2, redAt: 3 },
  technical: { unreachableAfterMinutes: 15 },
}

let cachedOverrides: Partial<ReassuranceRules> | null | undefined

// Returns the active rules. Overrides from the `ruleRegistry` collection (if any were loaded via
// loadReassuranceRuleOverrides) are shallow-merged over the frozen defaults so tuning does not require
// a redeploy. The default path is pure and synchronous.
export function getReassuranceRules(): ReassuranceRules {
  if (!cachedOverrides) return REASSURANCE_RULES
  return { ...REASSURANCE_RULES, ...cachedOverrides, ruleVersion: cachedOverrides.ruleVersion || REASSURANCE_RULES.ruleVersion }
}

export function setReassuranceRuleOverrides(overrides: Partial<ReassuranceRules> | null) {
  cachedOverrides = overrides
}
