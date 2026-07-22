import type { Db } from 'mongodb'
import { collections } from '../platform/collections.js'
import { newId } from '../platform/ids.js'

export function reviewCaseNotificationDedupeKey(caseId: string) {
  return `review_case:${caseId}`
}

/** Materializes one durable in-app notification per authorized caregiver. */
export async function materializeReviewCaseNotifications(db: Db, caseId: string) {
  const reviewCase = await db.collection<any>(collections.reviewCases).findOne({ _id: caseId })
  if (!reviewCase) return { created: 0 }
  const relationships = await db.collection<any>(collections.careRelationships).find({
    tenantId: reviewCase.tenantId,
    patientId: reviewCase.patientId,
    status: 'active',
    scopes: 'monitoring:read',
    $or: [{ validTo: null }, { validTo: { $gt: new Date() } }, { validTo: { $exists: false } }],
  }).project({ userId: 1 }).toArray()
  let created = 0
  for (const relationship of relationships) {
    const dedupeKey = reviewCaseNotificationDedupeKey(caseId)
    const result = await db.collection<any>(collections.notifications).updateOne({
      tenantId: reviewCase.tenantId,
      recipientUserId: relationship.userId,
      dedupeKey,
    }, { $setOnInsert: {
      _id: newId('notif'),
      tenantId: reviewCase.tenantId,
      recipientUserId: relationship.userId,
      patientId: reviewCase.patientId,
      type: reviewCase.priority === 'urgent' ? 'needs_attention' : 'worth_checking',
      state: 'unread',
      title: reviewCase.priority === 'urgent' ? 'This may need attention' : 'Worth checking in',
      body: 'A new review item is available for this patient.',
      dedupeKey,
      source: { type: 'review_case', id: caseId },
      createdAt: new Date(),
      updatedAt: new Date(),
    } }, { upsert: true })
    created += result.upsertedCount
  }
  return { created }
}
