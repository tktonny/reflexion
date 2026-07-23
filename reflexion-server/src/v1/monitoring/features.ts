import type { NumericFeatures } from './algorithms.js'

export type TranscriptTurn = { role?: string; text?: string; startedAt?: Date; endedAt?: Date }

export function extractStructuredFeatures(turns: TranscriptTurn[], acquisition: Record<string, unknown> = {}) {
  const patientTurns = turns.filter((turn) => turn.role === 'patient' && turn.text?.trim())
  const tokens = patientTurns.flatMap((turn) => tokenize(turn.text || ''))
  const uniqueTokens = new Set(tokens.map((token) => token.toLocaleLowerCase()))
  const speechMs = finiteNumber(acquisition.patientSpeechMs)
  const interruptionCount = finiteNumber(acquisition.interruptionCount)
  const responseLatencies: number[] = []
  for (let index = 1; index < turns.length; index++) {
    if (turns[index].role !== 'patient' || turns[index - 1].role !== 'assistant') continue
    const start = turns[index].startedAt?.getTime()
    const previousEnd = turns[index - 1].endedAt?.getTime()
    if (start !== undefined && previousEnd !== undefined && start >= previousEnd) responseLatencies.push(start - previousEnd)
  }
  const wordFindingMarkers = patientTurns.reduce((count, turn) => count + markerCount(turn.text || ''), 0)
  const features: NumericFeatures = {
    'speechAcoustic.speechRate': speechMs && speechMs > 0 ? tokens.length / (speechMs / 60_000) : undefined,
    'speechAcoustic.responseLatencyMs': responseLatencies.length ? responseLatencies.reduce((a, b) => a + b, 0) / responseLatencies.length : undefined,
    'speechLanguage.lexicalDiversity': tokens.length ? uniqueTokens.size / tokens.length : undefined,
    'speechLanguage.wordFindingMarkers': tokens.length ? wordFindingMarkers : undefined,
    'interaction.patientTurns': patientTurns.length || undefined,
    'interaction.interruptionCount': interruptionCount,
  }
  return {
    schemaVersion: 'home-features-v1',
    featureGroups: unflatten(features),
    flattened: features,
    evidence: { patientTurns: patientTurns.length, tokenCount: tokens.length, patientSpeechMs: speechMs },
  }
}

export function flattenFeatureGroups(groups: Record<string, unknown>) {
  const result: NumericFeatures = {}
  for (const [group, groupValue] of Object.entries(groups)) {
    if (!groupValue || typeof groupValue !== 'object' || Array.isArray(groupValue)) continue
    for (const [feature, value] of Object.entries(groupValue as Record<string, unknown>)) {
      if (typeof value === 'number' && Number.isFinite(value)) result[`${group}.${feature}`] = value
    }
  }
  return result
}

function unflatten(features: NumericFeatures) {
  const groups: Record<string, Record<string, number>> = {}
  for (const [path, value] of Object.entries(features)) {
    if (value === undefined) continue
    const [group, feature] = path.split('.')
    if (!groups[group]) groups[group] = {}
    groups[group][feature] = Math.round(value * 10_000) / 10_000
  }
  return groups
}

function tokenize(text: string) {
  const segmenter = new Intl.Segmenter(undefined, { granularity: 'word' })
  return Array.from(segmenter.segment(text)).filter((segment) => segment.isWordLike).map((segment) => segment.segment)
}

function markerCount(text: string) {
  const matches = text.toLocaleLowerCase().match(/\b(?:um+|uh+|er+|thing(?:y|amajig)?)\b|(?:那个|这个|怎么说|想不起来|叫什么)/g)
  return matches?.length || 0
}

function finiteNumber(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}
