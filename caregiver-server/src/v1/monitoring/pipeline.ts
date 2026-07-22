import type { Db } from 'mongodb'
import { collections } from '../platform/collections.js'
import { newId } from '../platform/ids.js'
import { getObjectStore } from '../platform/objectStore.js'
import { appendOutbox } from '../platform/outbox.js'
import { anomalyBand, buildStructuredBaseline, cosineDistance, mad, median, normalizedCentroid, operationalEligibility, researchEligibility, robustZ, scoreStructured } from './algorithms.js'
import { configuredEmbeddingProvider } from './embeddings.js'
import { extractStructuredFeatures, flattenFeatureGroups, type TranscriptTurn } from './features.js'

const PIPELINE_VERSION = process.env.MONITORING_PIPELINE_VERSION || 'longitudinal-v1'
const RULE_VERSION = process.env.MONITORING_RULE_VERSION || 'transparent-rules-v1'

export async function verifyArtifact(db: Db, artifactId: string, correlationId: string) {
  const artifact = await db.collection<any>(collections.artifacts).findOne({ _id: artifactId })
  if (!artifact || artifact.state === 'verified') return
  const verified = await getObjectStore().verify({
    objectKey: String(artifact.objectKey), hash: String(artifact.hash), sizeBytes: Number(artifact.sizeBytes),
  })
  if (!verified) throw new Error(`Artifact ${artifactId} could not be verified in object storage.`)
  await db.collection<any>(collections.artifacts).updateOne({ _id: artifactId }, { $set: { state: 'verified', verifiedAt: new Date(), updatedAt: new Date() } })
  await appendOutbox(db, { eventType: 'artifact.verified', tenantId: String(artifact.tenantId), patientId: String(artifact.patientId),
    aggregateType: 'artifact', aggregateId: artifactId, correlationId, payload: { sessionId: artifact.sessionId } })
}

export async function processCompletedSession(db: Db, sessionId: string, correlationId: string) {
  const session = await db.collection<any>(collections.sessions).findOne({ _id: sessionId })
  if (!session || ['completed', 'excluded', 'review_pending'].includes(String(session.state))) return
  await updateProcessingStage(db, sessionId, 'quality_identity', {
    state: 'processing', startedAt: session.processingSummary?.startedAt || new Date(), retryable: false,
  })
  const artifacts = await db.collection<any>(collections.artifacts).find({ _id: { $in: session.artifactIds || [] }, sessionId }).toArray()
  if (artifacts.some((artifact) => artifact.state !== 'verified')) throw new Error(`Session ${sessionId} is waiting for artifact verification.`)

  const turns = await db.collection<any>(collections.transcriptTurns).find({ sessionId }).sort({ sequence: 1 }).toArray() as TranscriptTurn[]
  const revision = Number(session.latestProcessingRevision || 0) + 1
  const consentGate = await evaluateConsent(db, session)
  const identity = await evaluateIdentity(db, session)
  const quality = evaluateSessionQuality(session, turns)
  await Promise.all([
    recordRun(db, session, revision, 'consent_gate', consentGate.verdict, consentGate),
    db.collection<any>(collections.identityLinks).updateOne({ sessionId, revision }, { $setOnInsert: {
      tenantId: session.tenantId, patientId: session.patientId, sessionId, revision, ...identity,
      pipelineVersion: PIPELINE_VERSION, createdAt: new Date(),
    } }, { upsert: true }),
    db.collection<any>(collections.qualityAssessments).updateOne({ sessionId, revision }, { $setOnInsert: {
      tenantId: session.tenantId, patientId: session.patientId, sessionId, revision, gate: quality.verdict,
      scores: quality.scores, flags: quality.flags, coverage: quality.coverage,
      pipelineVersion: PIPELINE_VERSION, createdAt: new Date(),
    } }, { upsert: true }),
  ])

  const inclusion = combineGates(consentGate.verdict, identity.verdict, quality.verdict)
  if (!['include', 'include_with_caveats'].includes(inclusion)) {
    await db.collection<any>(collections.sessions).updateOne({ _id: sessionId }, { $set: {
      state: 'excluded', latestProcessingRevision: revision,
      processingSummary: { state: 'completed', stage: 'complete', inclusion, qualityFlags: quality.flags,
        identityVerdict: identity.verdict, completedAt: new Date(), retryable: false },
      updatedAt: new Date(),
    }, $inc: { stateVersion: 1 } })
    await upsertMonitoringWindow(db, session, inclusion, undefined, quality.coverage, correlationId)
    return
  }

  await updateProcessingStage(db, sessionId, 'feature_extraction')
  const extracted = extractStructuredFeatures(turns, session.acquisition || {})
  const featureSnapshot = {
    _id: `feat_${sessionId}_${revision}`, tenantId: session.tenantId, patientId: session.patientId,
    sessionId, capturedAt: session.localCompletedAt || new Date(), schemaVersion: extracted.schemaVersion,
    sessionType: session.type,
    pipelineVersion: PIPELINE_VERSION, featureGroups: extracted.featureGroups, evidence: extracted.evidence,
    inclusion, revision, createdAt: new Date(),
  }
  await db.collection<any>(collections.featureSnapshots).updateOne({
    sessionId, schemaVersion: extracted.schemaVersion, pipelineVersion: PIPELINE_VERSION,
  }, { $setOnInsert: featureSnapshot }, { upsert: true })
  await recordRun(db, session, revision, 'feature_extraction', 'completed', { featureSnapshotId: featureSnapshot._id })
  await appendOutbox(db, { eventType: 'features.extracted', tenantId: String(session.tenantId), patientId: String(session.patientId),
    aggregateType: 'session', aggregateId: sessionId, correlationId, payload: { featureSnapshotId: featureSnapshot._id } })

  let embedding: Record<string, any> | null = null
  await updateProcessingStage(db, sessionId, 'embedding')
  const provider = configuredEmbeddingProvider()
  const patientText = turns.filter((turn) => turn.role === 'patient').map((turn) => turn.text?.trim()).filter(Boolean).join('\n')
  if (provider && patientText) {
    try {
      const result = await provider.embed(patientText)
      embedding = {
        _id: `emb_${sessionId}_${result.family}_${result.modelId.replace(/[^A-Za-z0-9_-]/g, '_')}`,
        tenantId: session.tenantId, patientId: session.patientId, sessionId,
        capturedAt: featureSnapshot.capturedAt, family: result.family, modelId: result.modelId,
        modelVersion: process.env.EMBEDDING_MODEL_VERSION || 'configured', dimensions: result.dimensions,
        vector: result.vector, quality: quality.scores.overall, inclusion, createdAt: new Date(),
      }
      await db.collection<any>(collections.featureEmbeddings).updateOne({ _id: embedding._id }, { $setOnInsert: embedding }, { upsert: true })
      await recordRun(db, session, revision, 'embedding', 'completed', { embeddingId: embedding._id, modelId: result.modelId })
      await appendOutbox(db, { eventType: 'embedding.created', tenantId: String(session.tenantId), patientId: String(session.patientId),
        aggregateType: 'session', aggregateId: sessionId, correlationId, payload: { embeddingId: embedding._id } })
    } catch (error) {
      // A configured embedding provider is an enrichment dependency, not a gate for durable
      // transcript ingestion. Structured monitoring remains available while the failure is visible.
      await recordRun(db, session, revision, 'embedding', 'failed', {
        retryable: true,
        reason: error instanceof Error ? error.message.slice(0, 500) : 'embedding_provider_failed',
      })
    }
  } else {
    await recordRun(db, session, revision, 'embedding', 'not_configured', { evidenceCreated: false })
  }

  // Companion conversations are retained as session observations, but they must not silently alter
  // the validated daily-check-in baseline or trigger a review case.
  if (session.type !== 'daily_checkin') {
    await db.collection<any>(collections.sessions).updateOne({ _id: sessionId }, { $set: {
      state: 'completed', latestProcessingRevision: revision,
      processingSummary: { state: 'completed', stage: 'complete', inclusion, qualityFlags: quality.flags,
        identityVerdict: identity.verdict, monitoringUse: 'observation_only', completedAt: new Date(), retryable: false },
      updatedAt: new Date(),
    }, $inc: { stateVersion: 1 } })
    return
  }

  await updateProcessingStage(db, sessionId, 'monitoring')
  await rebuildOperationalBaseline(db, session, correlationId)
  const anomaly = await scoreAgainstResearchBaseline(db, session, featureSnapshot, embedding, quality, identity, correlationId)
  const window = await upsertMonitoringWindow(db, session, inclusion, anomaly, quality.coverage, correlationId)
  const needsReview = anomaly && ['review_recommended', 'priority_review'].includes(String(anomaly.status))
  if (needsReview) await createReviewCase(db, session, anomaly!, correlationId)
  await db.collection<any>(collections.sessions).updateOne({ _id: sessionId }, { $set: {
    state: needsReview ? 'review_pending' : 'completed', latestProcessingRevision: revision,
    processingSummary: { state: 'completed', stage: 'complete', inclusion, qualityFlags: quality.flags,
      identityVerdict: identity.verdict, anomalyState: anomaly?.status || 'baseline_building',
      monitoringWindowId: window._id, completedAt: new Date(), retryable: false },
    updatedAt: new Date(),
  }, $inc: { stateVersion: 1 } })
}

async function updateProcessingStage(db: Db, sessionId: string, stage: string, extra: Record<string, unknown> = {}) {
  await db.collection<any>(collections.sessions).updateOne({ _id: sessionId }, { $set: {
    state: 'processing',
    processingSummary: { state: 'processing', stage, ...extra },
    updatedAt: new Date(),
  } })
}

async function evaluateConsent(db: Db, session: Record<string, any>) {
  if (session.type === 'companion' || session.type === 'device_test') return { verdict: 'include', reason: 'not_required_for_session_type' }
  if (!session.consentRef?.consentId) return { verdict: 'exclude', reason: 'missing_capture_consent' }
  const consent = await db.collection<any>(collections.consents).findOne({
    _id: session.consentRef.consentId, tenantId: session.tenantId, patientId: session.patientId,
    purpose: session.consentRef.purpose, status: 'granted', withdrawnAt: null,
  })
  return consent ? { verdict: 'include', reason: 'active_consent' } : { verdict: 'exclude', reason: 'consent_withdrawn_or_invalid' }
}

async function evaluateIdentity(db: Db, session: Record<string, any>) {
  if (!session.deviceId) return { verdict: 'manual_review', confidence: 0.3, reasons: ['NO_DEVICE_ASSIGNMENT'] }
  const assignment = await db.collection<any>(collections.assignments).findOne({
    tenantId: session.tenantId, patientId: session.patientId, deviceId: session.deviceId, status: 'active',
  })
  return assignment
    ? { verdict: 'linked', confidence: 0.7, reasons: ['ACTIVE_DEVICE_ASSIGNMENT'], enrollmentRef: null }
    : { verdict: 'exclude', confidence: 0, reasons: ['DEVICE_ASSIGNMENT_INVALID'], enrollmentRef: null }
}

export function evaluateSessionQuality(session: Record<string, any>, turns: TranscriptTurn[]) {
  const acquisition = session.acquisition || {}
  const patientTurns = turns.filter((turn) => turn.role === 'patient' && turn.text?.trim())
  const tokenCount = patientTurns.flatMap((turn) => Array.from(new Intl.Segmenter(undefined, { granularity: 'word' }).segment(turn.text || '')).filter((part) => part.isWordLike)).length
  const speechMs = typeof acquisition.patientSpeechMs === 'number' ? acquisition.patientSpeechMs : undefined
  const flags: string[] = []
  if (patientTurns.length < 3) flags.push('PATIENT_TURNS_INSUFFICIENT')
  if (tokenCount < 15) flags.push('TRANSCRIPT_COVERAGE_INSUFFICIENT')
  if (speechMs === undefined) flags.push('PATIENT_SPEECH_DURATION_UNAVAILABLE')
  else if (speechMs < 15_000) flags.push('PATIENT_SPEECH_DURATION_INSUFFICIENT')
  if (acquisition.languageMismatch === true) flags.push('LANGUAGE_MISMATCH')
  if (typeof acquisition.caregiverSpeechRatio === 'number' && acquisition.caregiverSpeechRatio > 0.35) flags.push('CAREGIVER_SPEECH_HIGH')
  const severe = flags.some((flag) => ['PATIENT_TURNS_INSUFFICIENT', 'TRANSCRIPT_COVERAGE_INSUFFICIENT', 'PATIENT_SPEECH_DURATION_INSUFFICIENT', 'LANGUAGE_MISMATCH'].includes(flag))
  const overall = Math.max(0, 1 - flags.length * 0.2)
  return { verdict: severe ? 'repeat_requested' : flags.length ? 'include_with_caveats' : 'include',
    scores: { overall, patientTurns: patientTurns.length, tokenCount, patientSpeechMs: speechMs }, flags,
    coverage: Math.min(patientTurns.length / 7, 1) }
}

function combineGates(consent: string, identity: string, quality: string) {
  if (consent === 'exclude' || identity === 'exclude') return 'exclude'
  if (identity === 'manual_review') return 'manual_review'
  if (quality === 'repeat_requested') return 'repeat_requested'
  if (quality === 'include_with_caveats' || identity !== 'linked') return 'include_with_caveats'
  return 'include'
}

async function rebuildOperationalBaseline(db: Db, session: Record<string, any>, correlationId: string) {
  const rows = await db.collection<any>(collections.sessions).find({
    tenantId: session.tenantId, patientId: session.patientId, type: 'daily_checkin', localCompletedAt: { $exists: true },
  }).project({ localCompletedAt: 1, acquisition: 1 }).sort({ localCompletedAt: 1 }).toArray()
  const completedAt = rows.map((row) => new Date(row.localCompletedAt)).filter((date) => !Number.isNaN(date.getTime()))
  const eligibility = operationalEligibility(completedAt)
  const previous = await db.collection<any>(collections.operationalBaselines).findOne({
    tenantId: session.tenantId, patientId: session.patientId, baselineType: 'reassurance_mvp', state: { $in: ['establishing', 'complete'] },
  }, { sort: { revision: -1 } })
  const metrics = rows.slice(-14).map((row) => row.acquisition || {})
  const document = {
    _id: newId('base'), tenantId: session.tenantId, patientId: session.patientId, baselineType: 'reassurance_mvp',
    state: eligibility.eligible ? 'complete' : 'establishing', sessionCount: eligibility.completedSessions,
    window: { days: 14, requiredSessions: 7, observedDays: eligibility.observedDays },
    metricAggregates: operationalAggregates(metrics), algorithmVersion: 'reassurance-ewma-v1', alpha: 0.1,
    revision: Number(previous?.revision || 0) + 1, sourceSessionId: session._id, createdAt: new Date(),
  }
  await db.collection<any>(collections.operationalBaselines).updateMany({
    tenantId: session.tenantId, patientId: session.patientId, baselineType: 'reassurance_mvp', state: { $in: ['establishing', 'complete'] },
  }, { $set: { state: 'superseded', supersededAt: new Date() } })
  await db.collection<any>(collections.operationalBaselines).insertOne(document)
  await appendOutbox(db, { eventType: 'baseline.rebuilt', tenantId: String(session.tenantId), patientId: String(session.patientId),
    aggregateType: 'operational_baseline', aggregateId: document._id, correlationId,
    payload: { baselineType: document.baselineType, state: document.state, revision: document.revision } })
  return document
}

function operationalAggregates(rows: Array<Record<string, unknown>>) {
  const names = ['durationMs', 'patientSpeechMs', 'patientTurns']
  return Object.fromEntries(names.map((name) => {
    const values = rows.map((row) => row[name]).filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
    return [name, values.length ? { median: median(values), mad: mad(values), sampleCount: values.length } : { sampleCount: 0 }]
  }))
}

async function scoreAgainstResearchBaseline(
  db: Db, session: Record<string, any>, snapshot: Record<string, any>, embedding: Record<string, any> | null,
  quality: Record<string, any>, identity: Record<string, any>, correlationId: string,
) {
  let baseline = await db.collection<any>(collections.baselineModels).findOne({
    tenantId: session.tenantId, patientId: session.patientId, state: 'active', baselineType: 'longitudinal_research',
    featureSchemaVersion: snapshot.schemaVersion, pipelineVersion: PIPELINE_VERSION,
  }, { sort: { revision: -1 } })
  if (!baseline) {
    const history = await db.collection<any>(collections.featureSnapshots).find({
      tenantId: session.tenantId, patientId: session.patientId, sessionId: { $ne: session._id },
      sessionType: 'daily_checkin',
      schemaVersion: snapshot.schemaVersion, pipelineVersion: PIPELINE_VERSION, inclusion: { $in: ['include', 'include_with_caveats'] },
    }).sort({ capturedAt: 1 }).toArray()
    const capturedAt = history.map((item) => new Date(item.capturedAt))
    const qualityCoverage = await plannedSessionCoverage(db, session, capturedAt)
    const eligibility = researchEligibility(capturedAt, qualityCoverage)
    if (!eligibility.eligible) return null
    const scalarAggregates = buildStructuredBaseline(history.map((item) => flattenFeatureGroups(item.featureGroups || {})))
    const embeddingBaseline = await buildEmbeddingBaseline(db, session, history)
    baseline = {
      _id: newId('base'), baselineId: newId('base'), tenantId: session.tenantId, patientId: session.patientId,
      baselineType: 'longitudinal_research', state: 'active', eligibility,
      featureSchemaVersion: snapshot.schemaVersion, pipelineVersion: PIPELINE_VERSION,
      scalarAggregates, embeddingBaseline, validFrom: history[0].capturedAt,
      validTo: history.at(-1)?.capturedAt, revision: 1, algorithmVersion: RULE_VERSION, createdAt: new Date(),
    }
    await db.collection<any>(collections.baselineModels).insertOne(baseline)
    await appendOutbox(db, { eventType: 'baseline.rebuilt', tenantId: String(session.tenantId), patientId: String(session.patientId),
      aggregateType: 'baseline_model', aggregateId: String(baseline._id), correlationId,
      payload: { baselineType: 'longitudinal_research', revision: 1 } })
  }
  const structured = scoreStructured(flattenFeatureGroups(snapshot.featureGroups || {}), baseline.scalarAggregates || {})
  const embeddingComponent = scoreEmbedding(embedding, baseline.embeddingBaseline)
  const available = [structured.value, embeddingComponent?.value].filter((value): value is number => value !== undefined)
  const raw = available.length ? available.reduce((sum, value) => sum + value, 0) / available.length : undefined
  const confidence = Math.min(Number(quality.scores?.overall || 0), Number(identity.confidence || 0), Number(baseline.eligibility?.qualityCoverage || 1))
  const recent = raw === undefined ? [] : await db.collection<any>(collections.anomalyScores).find({
    tenantId: session.tenantId, patientId: session.patientId,
  }).project({ overall: 1 }).sort({ createdAt: -1 }).limit(2).toArray()
  let persistenceCount = raw !== undefined && raw >= 0.35 ? 1 : 0
  for (const previous of recent) {
    if (typeof previous.overall !== 'number' || previous.overall < 0.35) break
    persistenceCount++
  }
  const status = anomalyBand(raw, confidence, persistenceCount)
  const score = {
    _id: newId('score'), tenantId: session.tenantId, patientId: session.patientId, sessionId: session._id,
    capturedAt: snapshot.capturedAt, baselineRef: baseline._id, baselineRevision: baseline.revision,
    scoreVersion: RULE_VERSION, components: { structuredDeviation: structured.value, embeddingDeviation: embeddingComponent?.value },
    overall: raw, status, confidence, reasons: [...structured.reasons, ...(embeddingComponent?.reasons || [])],
    context: { missingFamilies: embedding ? [] : ['embedding'] }, createdAt: new Date(),
  }
  await db.collection<any>(collections.anomalyScores).updateOne({
    sessionId: session._id, baselineRevision: baseline.revision, scoreVersion: RULE_VERSION,
  }, { $setOnInsert: score }, { upsert: true })
  await appendOutbox(db, { eventType: 'anomaly.scored', tenantId: String(session.tenantId), patientId: String(session.patientId),
    aggregateType: 'session', aggregateId: String(session._id), correlationId, payload: { scoreId: score._id, status } })
  return score
}

async function buildEmbeddingBaseline(db: Db, session: Record<string, any>, history: Record<string, any>[]) {
  const sessionIds = history.map((item) => item.sessionId)
  const vectors = await db.collection<any>(collections.featureEmbeddings).find({
    tenantId: session.tenantId, patientId: session.patientId, sessionId: { $in: sessionIds }, inclusion: { $in: ['include', 'include_with_caveats'] },
  }).toArray()
  if (vectors.length < 12) return null
  const first = vectors[0]
  const homogeneous = vectors.filter((item) => item.family === first.family && item.modelId === first.modelId && item.dimensions === first.dimensions)
  if (homogeneous.length < 12) return null
  const centroid = normalizedCentroid(homogeneous.map((item) => item.vector as number[]))
  if (!centroid) return null
  const distances = homogeneous.map((item) => cosineDistance(item.vector as number[], centroid)).filter((value): value is number => value !== undefined)
  return { family: first.family, modelId: first.modelId, dimensions: first.dimensions, centroid,
    distanceMedian: median(distances), distanceMad: mad(distances), sampleCount: homogeneous.length }
}

async function plannedSessionCoverage(db: Db, session: Record<string, any>, usableDates: Date[]) {
  if (!usableDates.length) return 0
  const sorted = [...usableDates].sort((a, b) => a.getTime() - b.getTime())
  const scheduledWindowSessions = await db.collection<any>(collections.sessions).countDocuments({
    tenantId: session.tenantId,
    patientId: session.patientId,
    type: 'daily_checkin',
    localCompletedAt: { $gte: sorted[0], $lte: sorted.at(-1) },
  })
  return scheduledWindowSessions ? Math.min(usableDates.length / scheduledWindowSessions, 1) : 0
}

function scoreEmbedding(embedding: Record<string, any> | null, baseline: Record<string, any> | null | undefined) {
  if (!embedding || !baseline || embedding.family !== baseline.family || embedding.modelId !== baseline.modelId || embedding.dimensions !== baseline.dimensions) return undefined
  const distance = cosineDistance(embedding.vector, baseline.centroid)
  if (distance === undefined || baseline.distanceMedian === undefined) return undefined
  const z = robustZ(distance, { median: baseline.distanceMedian, mad: baseline.distanceMad || 0 })
  const value = Math.min(Math.max(z, 0) / 4, 1)
  return { value, reasons: z >= 2 ? [{ code: 'SEMANTIC_EMBEDDING_DISTANCE_INCREASE', family: embedding.family,
    current: distance, baselineMedian: baseline.distanceMedian, robustZ: z, contribution: value }] : [] }
}

async function upsertMonitoringWindow(db: Db, session: Record<string, any>, inclusion: string, anomaly: Record<string, any> | null | undefined, coverage: number, correlationId: string) {
  const displayState = inclusion === 'repeat_requested' ? 'repeat_needed'
    : inclusion === 'manual_review' || ['review_recommended', 'priority_review'].includes(String(anomaly?.status)) ? 'review_pending'
      : anomaly ? 'on_track' : 'building'
  const windowEnd = new Date(session.localCompletedAt || Date.now())
  const windowStart = new Date(windowEnd.getTime() - 7 * 24 * 60 * 60 * 1000)
  const document = {
    _id: newId('win'), tenantId: session.tenantId, patientId: session.patientId, sessionId: session._id,
    windowStart, windowEnd, baselineState: anomaly ? 'complete' : 'building', coverage: { qualityCoverage: coverage },
    inclusion, status: displayState, action: displayState === 'repeat_needed' ? 'repeat_session' : displayState === 'review_pending' ? 'provider_review' : 'none',
    ruleVersion: RULE_VERSION, sourceScoreId: anomaly?._id, updatedAt: new Date(),
  }
  await db.collection<any>(collections.monitoringWindows).updateOne({
    tenantId: session.tenantId, patientId: session.patientId, windowEnd, ruleVersion: RULE_VERSION,
  }, { $set: document }, { upsert: true })
  return document
}

async function createReviewCase(db: Db, session: Record<string, any>, anomaly: Record<string, any>, correlationId: string) {
  const existing = await db.collection<any>(collections.reviewCases).findOne({ tenantId: session.tenantId, sourceRefs: session._id })
  if (existing) return existing
  const reviewCase = {
    _id: newId('case'), tenantId: session.tenantId, patientId: session.patientId,
    reason: anomaly.status === 'priority_review' ? 'persistent_high_confidence_change' : 'persistent_change_review_recommended',
    priority: anomaly.status === 'priority_review' ? 'urgent' : 'elevated', state: 'open',
    sourceRefs: [session._id, anomaly._id], createdAt: new Date(),
  }
  await db.collection<any>(collections.reviewCases).insertOne(reviewCase)
  await appendOutbox(db, { eventType: 'review_case.created', tenantId: String(session.tenantId), patientId: String(session.patientId),
    aggregateType: 'review_case', aggregateId: reviewCase._id, correlationId, payload: { priority: reviewCase.priority } })
  return reviewCase
}

async function recordRun(db: Db, session: Record<string, any>, revision: number, stage: string, state: string, checkpoint: Record<string, unknown>) {
  await db.collection<any>(collections.processingRuns).updateOne({
    sessionId: session._id, stage, pipelineVersion: PIPELINE_VERSION, revision,
  }, { $setOnInsert: {
    _id: newId('run'), tenantId: session.tenantId, patientId: session.patientId, sessionId: session._id,
    stage, pipelineVersion: PIPELINE_VERSION, revision, state, attempts: 1, checkpoint,
    timestamps: { startedAt: new Date(), completedAt: new Date() }, createdAt: new Date(),
  } }, { upsert: true })
}
