import type { Db } from 'mongodb'
import { collections } from '../platform/collections.js'
import { getObjectStore } from '../platform/objectStore.js'

/**
 * Executes a confirmed, queued deletion request. Every filter is tenant + patient scoped. The request is
 * deliberately resumable: a worker retry sees `processing` and repeats only idempotent deletes. Object-store
 * cleanup is reported separately when storage is not configured, rather than claiming that media vanished.
 */
export async function processDataDeletionRequest(db: Db, requestId: string) {
  const request = await db.collection<any>(collections.dataDeletionRequests).findOne({ _id: requestId })
  if (!request || request.state === 'completed') return request
  const tenantId = String(request.tenantId)
  const patientId = String(request.patientId)
  const categories = new Set<string>(Array.isArray(request.categories) ? request.categories.map(String) : [])
  await db.collection<any>(collections.dataDeletionRequests).updateOne(
    { _id: requestId, state: { $in: ['queued', 'processing', 'partial'] } },
    { $set: { state: 'processing', updatedAt: new Date() } },
  )

  const remainingObjectKeys: string[] = []
  if (categories.has('sessions')) {
    const sessions = await db.collection<any>(collections.sessions).find({ tenantId, patientId }).project({ _id: 1 }).toArray()
    const sessionIds = sessions.map((session) => String(session._id))
    if (sessionIds.length) {
      const artifacts = await db.collection<any>(collections.artifacts).find({ tenantId, patientId, sessionId: { $in: sessionIds } }).project({ objectKey: 1 }).toArray()
      const store = getObjectStore()
      for (const artifact of artifacts) {
        const objectKey = typeof artifact.objectKey === 'string' ? artifact.objectKey : ''
        if (!objectKey) continue
        try {
          if (!await store.deleteObject(objectKey)) remainingObjectKeys.push(objectKey)
        } catch {
          remainingObjectKeys.push(objectKey)
        }
      }
      const dependentCollections = [
        collections.sessionEvents, collections.transcriptTurns, collections.artifacts,
        collections.processingRuns, collections.qualityAssessments, collections.identityLinks,
        collections.featureSnapshots, collections.featureEmbeddings, collections.sessionObservations,
      ]
      for (const collection of dependentCollections) await db.collection<any>(collection).deleteMany({ tenantId, patientId, sessionId: { $in: sessionIds } })
      await db.collection<any>(collections.sessions).deleteMany({ tenantId, patientId, _id: { $in: sessionIds } })
    }
    await db.collection<any>(collections.dailySummaries).deleteMany({ tenantId, patientId })
    // Long-term soft continuity memory is derived from those conversations. Remove it with the selected
    // session data so a later Mirror request cannot resurrect facts the caregiver asked us to delete.
    await db.collection<any>(collections.patientMemory).deleteOne({ _id: patientId, tenantId })
  }
  if (categories.has('messages')) {
    await db.collection<any>(collections.familyMessages).deleteMany({ tenantId, patientId })
  }
  if (categories.has('routine-responses')) {
    await db.collection<any>(collections.reminderOccurrences).deleteMany({ tenantId, patientId })
    await db.collection<any>(collections.caregiverTasks).deleteMany({ tenantId, patientId })
  }
  if (categories.has('device-events')) {
    const assignments = await db.collection<any>(collections.assignments).find({ tenantId, patientId }).project({ deviceId: 1 }).toArray()
    const deviceIds = [...new Set(assignments.map((assignment) => String(assignment.deviceId)).filter(Boolean))]
    if (deviceIds.length) await db.collection<any>(collections.deviceTelemetry).deleteMany({ 'meta.tenantId': tenantId, 'meta.deviceId': { $in: deviceIds } })
  }

  const now = new Date()
  const state = remainingObjectKeys.length ? 'partial' : 'completed'
  const setFields: Record<string, unknown> = {
    state,
    updatedAt: now,
    remainingObjectKeys,
    error: state === 'partial' ? 'OBJECT_STORE_CLEANUP_PENDING' : null,
  }
  if (state === 'completed') setFields.completedAt = now
  await db.collection<any>(collections.dataDeletionRequests).updateOne(
    { _id: requestId },
    state === 'completed'
      ? { $set: setFields }
      : { $set: setFields, $unset: { completedAt: '' } },
  )
  return { ...request, state, remainingObjectKeys }
}
