export type NumericFeatures = Record<string, number | undefined>

export type ScalarAggregate = {
  median: number
  mad: number
  sampleCount: number
  missingRate: number
  minimum: number
  maximum: number
}

export type StructuredBaseline = Record<string, ScalarAggregate>

export const FEATURE_DIRECTIONS: Record<string, 'higher_is_worse' | 'lower_is_worse' | 'two_sided'> = {
  'speechAcoustic.speechRate': 'lower_is_worse',
  'speechAcoustic.responseLatencyMs': 'higher_is_worse',
  'speechLanguage.lexicalDiversity': 'lower_is_worse',
  'speechLanguage.wordFindingMarkers': 'higher_is_worse',
  'interaction.patientTurns': 'lower_is_worse',
  'interaction.interruptionCount': 'higher_is_worse',
}

export function median(values: number[]) {
  if (!values.length) return undefined
  const sorted = [...values].sort((a, b) => a - b)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
}

export function mad(values: number[], center = median(values)) {
  if (!values.length || center === undefined) return undefined
  return median(values.map((value) => Math.abs(value - center)))
}

export function robustZ(value: number, aggregate: Pick<ScalarAggregate, 'median' | 'mad'>, epsilon = 1e-6) {
  return 0.6745 * (value - aggregate.median) / Math.max(aggregate.mad, epsilon)
}

export function buildStructuredBaseline(samples: NumericFeatures[]): StructuredBaseline {
  const keys = new Set(samples.flatMap((sample) => Object.keys(sample)))
  const baseline: StructuredBaseline = {}
  for (const key of keys) {
    const values = samples.map((sample) => sample[key]).filter((value): value is number => Number.isFinite(value))
    if (!values.length) continue
    const center = median(values)!
    baseline[key] = {
      median: center,
      mad: mad(values, center) || 0,
      sampleCount: values.length,
      missingRate: 1 - values.length / samples.length,
      minimum: Math.min(...values),
      maximum: Math.max(...values),
    }
  }
  return baseline
}

export function scoreStructured(features: NumericFeatures, baseline: StructuredBaseline) {
  const reasons: Array<Record<string, unknown>> = []
  const contributions: number[] = []
  for (const [feature, direction] of Object.entries(FEATURE_DIRECTIONS)) {
    const value = features[feature]
    const aggregate = baseline[feature]
    if (value === undefined || !aggregate || aggregate.sampleCount < 5 || aggregate.missingRate > 0.5) continue
    const z = robustZ(value, aggregate)
    const directionalZ = direction === 'higher_is_worse' ? z : direction === 'lower_is_worse' ? -z : Math.abs(z)
    if (directionalZ <= 0) continue
    const contribution = Math.min(directionalZ / 4, 1)
    contributions.push(contribution)
    if (directionalZ >= 2) reasons.push({
      code: reasonCode(feature, direction), feature, direction: 'worse', current: value,
      baselineMedian: aggregate.median, robustZ: round(z), contribution: round(contribution),
    })
  }
  return {
    value: contributions.length ? contributions.reduce((sum, value) => sum + value, 0) / contributions.length : undefined,
    reasons,
    observedFeatures: contributions.length,
  }
}

export function operationalEligibility(completedAt: Date[], now = new Date()) {
  const start = new Date(now.getTime() - 14 * DAY_MS)
  const withinWindow = completedAt.filter((date) => date >= start && date <= now)
  const oldest = completedAt.length ? Math.min(...completedAt.map((date) => date.getTime())) : now.getTime()
  const observedDays = Math.min(14, Math.floor((now.getTime() - oldest) / DAY_MS) + 1)
  return {
    eligible: withinWindow.length >= 7 && observedDays >= 14,
    completedSessions: withinWindow.length,
    requiredSessions: 7,
    windowDays: 14,
    observedDays,
  }
}

export function researchEligibility(capturedAt: Date[], qualityCoverage?: number) {
  if (!capturedAt.length) return { eligible: false, usableSessions: 0, spanDays: 0, distinctWeeks: 0, qualityCoverage: 0 }
  const sorted = [...capturedAt].sort((a, b) => a.getTime() - b.getTime())
  const spanDays = Math.floor((sorted.at(-1)!.getTime() - sorted[0].getTime()) / DAY_MS) + 1
  const distinctWeeks = new Set(sorted.map(weekKey)).size
  const coverage = qualityCoverage ?? Math.min(sorted.length / Math.max(spanDays, 1), 1)
  return { eligible: sorted.length >= 12 && spanDays >= 28 && distinctWeeks >= 3 && coverage >= 0.7,
    usableSessions: sorted.length, spanDays, distinctWeeks, qualityCoverage: coverage }
}

export function ewma(values: number[], alpha = 0.1) {
  if (!values.length) return undefined
  return values.slice(1).reduce((current, value) => alpha * value + (1 - alpha) * current, values[0])
}

export function cosineDistance(a: number[], b: number[]) {
  if (!a.length || a.length !== b.length) return undefined
  let dot = 0; let normA = 0; let normB = 0
  for (let index = 0; index < a.length; index++) {
    dot += a[index] * b[index]
    normA += a[index] ** 2
    normB += b[index] ** 2
  }
  if (!normA || !normB) return undefined
  return 1 - dot / (Math.sqrt(normA) * Math.sqrt(normB))
}

export function normalizedCentroid(vectors: number[][]) {
  if (!vectors.length || !vectors[0].length || vectors.some((vector) => vector.length !== vectors[0].length)) return undefined
  const centroid = Array(vectors[0].length).fill(0) as number[]
  for (const vector of vectors) for (let index = 0; index < vector.length; index++) centroid[index] += vector[index] / vectors.length
  const norm = Math.sqrt(centroid.reduce((sum, value) => sum + value ** 2, 0))
  return norm ? centroid.map((value) => value / norm) : undefined
}

export function anomalyBand(score: number | undefined, confidence: number, persistenceCount = 1) {
  if (score === undefined || confidence < 0.5) return 'insufficient_data'
  if (score >= 0.8 && persistenceCount >= 2) return 'priority_review'
  if (score >= 0.6 && persistenceCount >= 2) return 'review_recommended'
  if (score >= 0.35) return 'watch'
  return 'within_personal_range'
}

function reasonCode(feature: string, direction: string) {
  return `${feature.replace(/([a-z])([A-Z])/g, '$1_$2').replaceAll('.', '_').toUpperCase()}_${direction === 'higher_is_worse' ? 'INCREASE' : direction === 'lower_is_worse' ? 'DECREASE' : 'CHANGE'}`
}

function weekKey(date: Date) {
  const day = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()))
  day.setUTCDate(day.getUTCDate() + 4 - (day.getUTCDay() || 7))
  const yearStart = new Date(Date.UTC(day.getUTCFullYear(), 0, 1))
  return `${day.getUTCFullYear()}-${Math.ceil((((day.getTime() - yearStart.getTime()) / DAY_MS) + 1) / 7)}`
}

function round(value: number) { return Math.round(value * 1000) / 1000 }
const DAY_MS = 24 * 60 * 60 * 1000
