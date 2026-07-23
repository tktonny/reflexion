import type { Db, IndexDescription } from 'mongodb'
import { getDb } from '../../lib/mongo.js'
import { collections as c } from './collections.js'

type IndexSet = [string, IndexDescription[]]

const indexes: IndexSet[] = [
  [c.users, [
    { key: { tenantId: 1, authSubject: 1 }, unique: true },
    { key: { tenantId: 1, emailNormalized: 1 }, unique: true, partialFilterExpression: { emailNormalized: { $type: 'string' } } },
  ]],
  [c.authSessions, [{ key: { refreshExpiresAt: 1 }, expireAfterSeconds: 0 }, { key: { tenantId: 1, userId: 1, status: 1 } }]],
  [c.passwordResetTokens, [{ key: { tokenDigest: 1 }, unique: true }, { key: { expiresAt: 1 }, expireAfterSeconds: 0 }]],
  [c.patients, [{ key: { tenantId: 1, _id: 1 }, unique: true }, { key: { tenantId: 1, status: 1 } }]],
  [c.careRelationships, [
    { key: { tenantId: 1, userId: 1, status: 1 } },
    { key: { tenantId: 1, patientId: 1, userId: 1, status: 1 }, unique: true, partialFilterExpression: { status: 'active' } },
  ]],
  [c.consents, [{ key: { tenantId: 1, patientId: 1, purpose: 1, status: 1 } }]],
  [c.programEnrollments, [{ key: { tenantId: 1, patientId: 1, status: 1, enrolledAt: -1 } }]],
  [c.devices, [{ key: { serialHash: 1 }, unique: true }, { key: { tenantId: 1, status: 1 } }]],
  [c.pairings, [
    { key: { expiresAt: 1 }, expireAfterSeconds: 0 },
    { key: { codeHash: 1, state: 1 }, unique: true, partialFilterExpression: { state: 'pending' } },
  ]],
  [c.assignments, [
    { key: { tenantId: 1, deviceId: 1, status: 1 }, unique: true, partialFilterExpression: { status: 'active' } },
    { key: { tenantId: 1, patientId: 1, assignmentType: 1, status: 1 }, unique: true, partialFilterExpression: { status: 'active', assignmentType: 'primary' } },
  ]],
  [c.credentials, [{ key: { deviceId: 1, status: 1 } }, { key: { refreshExpiresAt: 1 }, expireAfterSeconds: 0 }]],
  [c.deviceConfigurations, [{ key: { deviceId: 1, configVersion: 1 }, unique: true }]],
  [c.deviceTelemetry, [
    { key: { 'meta.deviceId': 1, 'measurements.heartbeatId': 1 }, unique: true },
    { key: { 'meta.tenantId': 1, recordedAt: -1 } },
  ]],
  [c.carePlans, [{ key: { tenantId: 1, patientId: 1, status: 1 } }]],
  [c.medicationPlans, [{ key: { tenantId: 1, patientId: 1, status: 1 } }]],
  [c.reminderOccurrences, [
    { key: { tenantId: 1, sourceId: 1, scheduledAt: 1 }, unique: true },
    { key: { tenantId: 1, patientId: 1, scheduledAt: 1 } },
  ]],
  [c.caregiverTasks, [{ key: { tenantId: 1, patientId: 1, status: 1, dueAt: 1 } }]],
  [c.sessions, [
    { key: { tenantId: 1, _id: 1 }, unique: true },
    { key: { tenantId: 1, patientId: 1, createdAt: -1 } },
    { key: { tenantId: 1, deviceId: 1, createdAt: -1 } },
    { key: { deviceId: 1, clientSessionId: 1 }, unique: true, partialFilterExpression: { clientSessionId: { $type: 'string' } } },
  ]],
  [c.sessionEvents, [
    { key: { sessionId: 1, eventId: 1 }, unique: true },
    { key: { sessionId: 1, sequence: 1 }, unique: true },
  ]],
  [c.transcriptTurns, [{ key: { sessionId: 1, turnId: 1 }, unique: true }, { key: { tenantId: 1, patientId: 1, startedAt: 1 } }]],
  [c.artifacts, [
    { key: { sessionId: 1, clientArtifactId: 1 }, unique: true },
    { key: { sessionId: 1, kind: 1, hash: 1 }, unique: true },
    { key: { state: 1, createdAt: 1 } },
  ]],
  [c.processingRuns, [{ key: { sessionId: 1, stage: 1, pipelineVersion: 1, revision: 1 }, unique: true }]],
  [c.qualityAssessments, [{ key: { sessionId: 1, revision: 1 }, unique: true }]],
  [c.identityLinks, [{ key: { sessionId: 1, revision: 1 }, unique: true }]],
  [c.featureSnapshots, [
    { key: { sessionId: 1, schemaVersion: 1, pipelineVersion: 1 }, unique: true },
    { key: { tenantId: 1, patientId: 1, sessionType: 1, capturedAt: 1 } },
  ]],
  [c.featureEmbeddings, [{ key: { tenantId: 1, patientId: 1, family: 1, modelId: 1, capturedAt: 1 } }]],
  [c.operationalBaselines, [{ key: { tenantId: 1, patientId: 1, baselineType: 1, state: 1 } }]],
  [c.dailyStatuses, [{ key: { tenantId: 1, patientId: 1, localDate: 1 }, unique: true }]],
  [c.awayPeriods, [{ key: { tenantId: 1, patientId: 1, startsOn: 1, endsOn: 1 } }]],
  [c.manualFlags, [{ key: { tenantId: 1, patientId: 1, state: 1, createdAt: -1 } }]],
  [c.baselineModels, [{ key: { tenantId: 1, patientId: 1, baselineId: 1, revision: 1 }, unique: true }]],
  [c.anomalyScores, [{ key: { sessionId: 1, baselineRevision: 1, scoreVersion: 1 }, unique: true }, { key: { tenantId: 1, patientId: 1, createdAt: -1 } }]],
  [c.monitoringWindows, [{ key: { tenantId: 1, patientId: 1, windowEnd: 1, ruleVersion: 1 }, unique: true }]],
  [c.reviewCases, [{ key: { tenantId: 1, state: 1, priority: 1, createdAt: 1 } }, { key: { tenantId: 1, patientId: 1, createdAt: -1 } }]],
  [c.reviewDispositions, [{ key: { tenantId: 1, caseId: 1, createdAt: 1 } }]],
  [c.notifications, [
    { key: { tenantId: 1, recipientUserId: 1, dedupeKey: 1 }, unique: true },
    { key: { tenantId: 1, recipientUserId: 1, state: 1, createdAt: -1 } },
  ]],
  [c.outboxEvents, [{ key: { state: 1, nextAttemptAt: 1 } }, { key: { aggregateType: 1, aggregateId: 1, occurredAt: 1 } }]],
  [c.eventConsumptions, [{ key: { eventId: 1, consumerName: 1 }, unique: true }]],
  [c.auditEvents, [{ key: { tenantId: 1, occurredAt: -1 } }]],
  [c.idempotencyRecords, [
    { key: { actorKey: 1, routeKey: 1, key: 1 }, unique: true },
    { key: { expiresAt: 1 }, expireAfterSeconds: 0 },
  ]],
  [c.toolInvocations, [{ key: { tenantId: 1, patientId: 1, sessionId: 1, createdAt: -1 } }, { key: { state: 1, createdAt: 1 } }]],
  [c.protocolRegistry, [{ key: { sessionType: 1, status: 1, activatedAt: -1 } }]],
  [c.featureRegistry, [{ key: { schemaVersion: 1, status: 1 }, unique: true, partialFilterExpression: { status: 'active' } }]],
  [c.modelRegistry, [{ key: { provider: 1, modelId: 1, version: 1 }, unique: true }]],
  [c.ruleRegistry, [{ key: { ruleType: 1, version: 1 }, unique: true }]],
]

export async function ensureV1Indexes(db?: Db) {
  const database = db || await getDb()
  for (const [name, definitions] of indexes) {
    await database.collection<any>(name).createIndexes(definitions)
  }
}
