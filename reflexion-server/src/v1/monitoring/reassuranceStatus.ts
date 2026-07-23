// Pure deterministic caregiver-status evaluator (implementation baseline §4, doc "Signal-to-Status
// Algorithm"). Takes already-gathered signals and returns the caregiver-facing status + ranked
// reasons. NO database, NO clock ambiguity, NO research/anomaly signals — SHADOW ISOLATION is
// structural: this function's input type simply has no field for anomaly scores or the longitudinal
// baseline, so caregiver colour can never be driven by them.

import { getReassuranceRules, type ReassuranceRules } from './ruleRegistry.js'

export type CaregiverStatus = 'establishing' | 'doing_well' | 'worth_checking' | 'needs_attention'
export type TechnicalState = 'ok' | 'possible_issue' | 'unreachable' | 'unknown'

export type MetricAggregate = { median: number; mad: number; sampleCount: number }

export type TodaySessionMetrics = {
  patientSpeechMs?: number
  patientTurns?: number
  medianResponseLatencyMs?: number
  sessionStartMinuteOfDay?: number
}

export type ReassuranceBaseline = {
  patientSpeechMs?: MetricAggregate
  patientTurns?: MetricAggregate
  medianResponseLatencyMs?: MetricAggregate
  sessionStartMinuteOfDay?: MetricAggregate
  weeklyRate?: number
}

// Consecutive-day breach counts sourced from persisted daily_statuses history (the finalize job),
// used to escalate a single-day amber to red only when the doc's persistence rule is satisfied.
export type PersistenceCounts = {
  speechBelowRed?: number
  turnsBelowRed?: number
  latencyAboveRed?: number
  offWindowDays7d?: number
  weeklyBelowRedWindows?: number
}

export type ReassuranceInputs = {
  baselineComplete: boolean
  baselineSessionCount: number
  completedToday: boolean
  today: TodaySessionMetrics | null
  baseline: ReassuranceBaseline | null
  // Away-adjusted consecutive missed days (away days already removed by the caller).
  missedStreak: number
  weeklyCompletedCount: number
  technicalState: TechnicalState
  awayActive: boolean
  manualFlag: 'worth_checking' | 'needs_attention' | null
  persistence?: PersistenceCounts
  rules?: ReassuranceRules
}

export type ReassuranceResult = {
  status: CaregiverStatus
  primaryReason: string
  secondaryReasons: string[]
  ruleVersion: string
  metricEvaluations: Array<{ metric: string; ratio?: number; verdict: 'green' | 'amber' | 'red' | 'neutral'; reason?: string }>
}

// Red reasons, strongest first (doc §11.1). Amber reasons, strongest first (doc §11.2).
const RED_PRIORITY = [
  'CAREGIVER_FLAG_NEEDS_ATTENTION',
  'CHECKIN_MISSED_3_DAYS',
  'DEVICE_UNREACHABLE_PERSISTENT',
  'SPOKE_LESS_THAN_USUAL',
  'FEWER_RESPONSES',
  'SLOWER_TO_RESPOND',
  'WEEKLY_ENGAGEMENT_DOWN',
]
const AMBER_PRIORITY = [
  'CHECKIN_MISSED_TODAY',
  'CHECKIN_MISSED_REPEATEDLY',
  'DEVICE_UNREACHABLE',
  'CAREGIVER_FLAG_WORTH_CHECKING',
  'SPOKE_LESS_THAN_USUAL',
  'FEWER_RESPONSES',
  'SLOWER_TO_RESPOND',
  'CHECKIN_OUTSIDE_USUAL_WINDOW',
  'WEEKLY_ENGAGEMENT_DOWN',
]

function strongest(reasons: string[], priority: string[]): string {
  for (const code of priority) if (reasons.includes(code)) return code
  return reasons[0]
}

function usable(aggregate: MetricAggregate | undefined, guardrailMin: number): aggregate is MetricAggregate {
  return Boolean(aggregate && aggregate.sampleCount >= 1 && aggregate.median >= guardrailMin && aggregate.median > 0)
}

export function evaluateReassuranceStatus(inputs: ReassuranceInputs): ReassuranceResult {
  const rules = inputs.rules ?? getReassuranceRules()
  const red: string[] = []
  const amber: string[] = []
  const secondary: string[] = []
  const metricEvaluations: ReassuranceResult['metricEvaluations'] = []
  const persistence = inputs.persistence ?? {}

  // --- Day-1 signals (no baseline required) ---
  if (inputs.manualFlag === 'needs_attention') red.push('CAREGIVER_FLAG_NEEDS_ATTENTION')
  else if (inputs.manualFlag === 'worth_checking') amber.push('CAREGIVER_FLAG_WORTH_CHECKING')

  // Missed-day streak (M2). Away days are already excluded from missedStreak by the caller, and an
  // active away period suppresses missed reasons entirely.
  if (!inputs.awayActive) {
    if (inputs.missedStreak >= rules.missedStreak.redAt) red.push('CHECKIN_MISSED_3_DAYS')
    else if (inputs.missedStreak >= rules.missedStreak.amberAt) amber.push('CHECKIN_MISSED_REPEATEDLY')
    else if (inputs.missedStreak >= 1 && !inputs.completedToday) secondary.push('CHECKIN_MISSED_TODAY')
  }

  // Technical separation (M13): a device issue is never framed as decline.
  if (inputs.technicalState === 'unreachable') {
    if (!inputs.completedToday) amber.push('DEVICE_UNREACHABLE')
    else secondary.push('DEVICE_UNREACHABLE')
  }

  // --- Baseline-dependent signals (only when baseline complete and today's session exists) ---
  if (inputs.baselineComplete && inputs.completedToday && inputs.today && inputs.baseline) {
    const t = inputs.today
    const b = inputs.baseline
    const g = rules.baseline.guardrails

    // M4 speech duration — lower is worse.
    if (t.patientSpeechMs !== undefined && usable(b.patientSpeechMs, g.patientSpeechMs)) {
      const ratio = t.patientSpeechMs / b.patientSpeechMs.median
      if (ratio < rules.speechDuration.redBelow && (persistence.speechBelowRed ?? 0) + 1 >= rules.speechDuration.redConsecutive) {
        red.push('SPOKE_LESS_THAN_USUAL'); metricEvaluations.push({ metric: 'M4_speech', ratio, verdict: 'red', reason: 'SPOKE_LESS_THAN_USUAL' })
      } else if (ratio < rules.speechDuration.amberBelow) {
        amber.push('SPOKE_LESS_THAN_USUAL'); metricEvaluations.push({ metric: 'M4_speech', ratio, verdict: 'amber', reason: 'SPOKE_LESS_THAN_USUAL' })
      } else metricEvaluations.push({ metric: 'M4_speech', ratio, verdict: 'green' })
    } else metricEvaluations.push({ metric: 'M4_speech', verdict: 'neutral' })

    // M6 turn count — lower is worse.
    if (t.patientTurns !== undefined && usable(b.patientTurns, g.patientTurns)) {
      const ratio = t.patientTurns / b.patientTurns.median
      if (ratio < rules.turnCount.redBelow && (persistence.turnsBelowRed ?? 0) + 1 >= rules.turnCount.redConsecutive) {
        red.push('FEWER_RESPONSES'); metricEvaluations.push({ metric: 'M6_turns', ratio, verdict: 'red', reason: 'FEWER_RESPONSES' })
      } else if (ratio < rules.turnCount.amberBelow) {
        amber.push('FEWER_RESPONSES'); metricEvaluations.push({ metric: 'M6_turns', ratio, verdict: 'amber', reason: 'FEWER_RESPONSES' })
      } else metricEvaluations.push({ metric: 'M6_turns', ratio, verdict: 'green' })
    } else metricEvaluations.push({ metric: 'M6_turns', verdict: 'neutral' })

    // M7 response latency — higher is worse.
    if (t.medianResponseLatencyMs !== undefined && usable(b.medianResponseLatencyMs, g.medianResponseLatencyMs)) {
      const ratio = t.medianResponseLatencyMs / b.medianResponseLatencyMs.median
      if (ratio > rules.latency.redAbove && (persistence.latencyAboveRed ?? 0) + 1 >= rules.latency.redConsecutive) {
        red.push('SLOWER_TO_RESPOND'); metricEvaluations.push({ metric: 'M7_latency', ratio, verdict: 'red', reason: 'SLOWER_TO_RESPOND' })
      } else if (ratio > rules.latency.amberAbove) {
        amber.push('SLOWER_TO_RESPOND'); metricEvaluations.push({ metric: 'M7_latency', ratio, verdict: 'amber', reason: 'SLOWER_TO_RESPOND' })
      } else metricEvaluations.push({ metric: 'M7_latency', ratio, verdict: 'green' })
    } else metricEvaluations.push({ metric: 'M7_latency', verdict: 'neutral' })

    // M3 routine window — a single off-window day is informational; amber only on a sustained pattern.
    if (t.sessionStartMinuteOfDay !== undefined && b.sessionStartMinuteOfDay && b.sessionStartMinuteOfDay.sampleCount >= 3) {
      const rw = rules.routineWindow
      const halfWidth = Math.min(Math.max(rw.madMultiplier * b.sessionStartMinuteOfDay.mad, rw.floorMinutes), rw.capMinutes)
      const start = t.sessionStartMinuteOfDay
      const offWindow = start < b.sessionStartMinuteOfDay.median - halfWidth || start > b.sessionStartMinuteOfDay.median + halfWidth
      if (offWindow && (persistence.offWindowDays7d ?? 0) + 1 >= rw.amberOffWindowDays) {
        amber.push('CHECKIN_OUTSIDE_USUAL_WINDOW'); metricEvaluations.push({ metric: 'M3_window', verdict: 'amber', reason: 'CHECKIN_OUTSIDE_USUAL_WINDOW' })
      } else if (offWindow) {
        secondary.push('CHECKIN_OUTSIDE_USUAL_WINDOW'); metricEvaluations.push({ metric: 'M3_window', verdict: 'green', reason: 'off_window_single_day' })
      } else metricEvaluations.push({ metric: 'M3_window', verdict: 'green' })
    } else metricEvaluations.push({ metric: 'M3_window', verdict: 'neutral' })
  }

  // M5 weekly engagement frequency — evaluated once baseline is complete (does not need today's session).
  if (inputs.baselineComplete && inputs.baseline?.weeklyRate) {
    const floor = Math.max(inputs.baseline.weeklyRate, rules.weeklyEngagement.floorSessions)
    const ratio = inputs.weeklyCompletedCount / floor
    if (ratio < rules.weeklyEngagement.redBelow && (persistence.weeklyBelowRedWindows ?? 0) + 1 >= 2) {
      red.push('WEEKLY_ENGAGEMENT_DOWN'); metricEvaluations.push({ metric: 'M5_weekly', ratio, verdict: 'red', reason: 'WEEKLY_ENGAGEMENT_DOWN' })
    } else if (ratio < rules.weeklyEngagement.amberBelow) {
      amber.push('WEEKLY_ENGAGEMENT_DOWN'); metricEvaluations.push({ metric: 'M5_weekly', ratio, verdict: 'amber', reason: 'WEEKLY_ENGAGEMENT_DOWN' })
    } else metricEvaluations.push({ metric: 'M5_weekly', ratio, verdict: 'green' })
  }

  // --- Resolve status by priority: Red > Amber > Green, establishing as the base when incomplete ---
  let status: CaregiverStatus
  let primaryReason: string
  if (red.length) {
    status = 'needs_attention'
    primaryReason = strongest(red, RED_PRIORITY)
  } else if (amber.length) {
    status = 'worth_checking'
    primaryReason = strongest(amber, AMBER_PRIORITY)
  } else if (inputs.baselineComplete) {
    status = 'doing_well'
    primaryReason = 'DAILY_PATTERN_ON_TRACK'
  } else {
    status = 'establishing'
    primaryReason = 'LEARNING_PERSONAL_ROUTINE'
  }

  if (inputs.completedToday) secondary.push('CHECKIN_COMPLETED_TODAY')
  if (inputs.awayActive) secondary.push('AWAY_PERIOD_ACTIVE')

  const secondaryReasons = [...new Set([...red, ...amber, ...secondary])].filter((code) => code !== primaryReason)

  return { status, primaryReason, secondaryReasons, ruleVersion: rules.ruleVersion, metricEvaluations }
}
