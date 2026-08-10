import type { Db, ObjectId } from 'mongodb'
import { DAILY_STATUS_COLLECTION } from './constants.js'
import { computeDailyStatusFromConversations } from './statusEngine.js'
import type { DailyPatientStatus } from './types.js'

export async function getDailyStatus(db: Db, patientId: ObjectId, date: string) {
  return db.collection<DailyPatientStatus>(DAILY_STATUS_COLLECTION).findOne({ patientId, date })
}

export async function getDailyStatusesForRange(db: Db, patientId: ObjectId, dates: string[]) {
  return db.collection<DailyPatientStatus>(DAILY_STATUS_COLLECTION).find({
    patientId,
    date: { $in: dates },
  }).toArray()
}

export async function refreshDailyStatusForDate(db: Db, patientId: ObjectId, date: string) {
  const computed = await computeDailyStatusFromConversations(db, patientId, date)
  return upsertDailyStatus(db, computed)
}

export async function upsertDailyStatus(db: Db, status: DailyPatientStatus) {
  const now = new Date()
  const update = {
    ...status,
    updatedAt: now,
  }
  delete update._id
  const result = await db.collection<DailyPatientStatus>(DAILY_STATUS_COLLECTION).findOneAndUpdate(
    { patientId: status.patientId, date: status.date },
    {
      $set: update,
      $unset: {
        duration: '',
        evaluated7pmAt: '',
        finalizedAt: '',
        reasons: '',
      },
      $setOnInsert: { createdAt: now },
    },
    {
      returnDocument: 'after',
      upsert: true,
    },
  )

  if (!result) {
    throw new Error('Unable to upsert daily patient status.')
  }

  return result
}
