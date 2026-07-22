import { Router } from 'express'
import { ObjectId, type Db } from 'mongodb'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { DB_NAME, NURSE_CONFIG_COLLECTION } from '../../lib/constants.js'
import { getMissedDays } from '../../lib/dates.js'
import { withMongo } from '../../lib/mongo.js'
import { getLatestV1SessionByPatientIds } from '../../lib/v1Conversations.js'
import type { StoredPatient } from '../../lib/types.js'

type PatientStatus = 'doing_well' | 'worth_checking' | 'needs_attention'

export const latestConfigRouter = Router()

latestConfigRouter.get('/', asyncHandler(async (request, response) => {
  const nurseId = typeof request.query.nurseId === 'string' ? request.query.nurseId : ''
  if (nurseId && !ObjectId.isValid(nurseId)) {
    response.status(400).json({ error: 'Nurse id is invalid.' })
    return
  }

  await withMongo(async (client) => {
    const db = client.db(DB_NAME)
    const filter = nurseId ? { _id: new ObjectId(nurseId) } : {}
    const document = await db.collection(NURSE_CONFIG_COLLECTION).findOne(filter, {
      sort: nurseId ? undefined : { createdAt: -1 },
      projection: {
        alertSensitivity: 1,
        email: 1,
        name: 1,
        patients: 1,
        phoneNumber: 1,
        preferredDailySummaryTime: 1,
        pushNotificationsEnabled: 1,
      },
    })

    const patients = await returnPatientsWithStatuses(db, (document?.patients || []) as StoredPatient[])

    response.json({
      nurseId: document?._id?.toHexString?.() || '',
      caregiverName: document?.name || '',
      email: document?.email || '',
      phoneNumber: document?.phoneNumber || '',
      pushNotificationsEnabled: Boolean(document?.pushNotificationsEnabled),
      alertSensitivity: document?.alertSensitivity || 'only_important_changes',
      preferredDailySummaryTime: document?.preferredDailySummaryTime || '09:00',
      patients,
    })
  })
}))

async function returnPatientsWithStatuses(db: Db, storedPatients: StoredPatient[]) {
  const patientIds = storedPatients
    .map((patient) => patient._id)
    .filter((patientId): patientId is ObjectId => Boolean(patientId))
  const latestConversationByPatientId = await getLatestConversationByPatientId(db, patientIds)

  return storedPatients.map((patient, index) => {
    const patientId = patient._id?.toHexString?.() || String(index)
    const latestConversation = latestConversationByPatientId.get(patientId)
    const status = getPatientStatus(latestConversation?.createdAt || null)

    return {
      id: patientId,
      patientId,
      name: patient.name || `Person ${index + 1}`,
      phoneNumber: patient.phoneNumber || '',
      age: patient.age || 0,
      preferredLanguage: patient.preferredLanguage || '',
      speechSpeed: patient.speechSpeed || 'Slow',
      mirrorName: patient.mirrorName || `Mirror ${index + 1}`,
      photoUrl: patient.photoUrl || '',
      status,
      statusLabel: getStatusLabel(status),
      lastSpokenAt: latestConversation?.createdAt?.toISOString?.() || null,
      lastSpokenLabel: formatLastSpoken(latestConversation?.createdAt || null),
      duration: latestConversation?.duration || 0,
    }
  })
}

// Per-patient "last conversation" now comes from the v1 `sessions` pipeline (the mirror no longer writes
// the legacy Conversation collection). lastSpoken = localCompletedAt (conversation end) ?? createdAt;
// duration = acquisition.durationMs / 1000 (seconds, matching the app's formatter).
async function getLatestConversationByPatientId(db: Db, patientIds: ObjectId[]) {
  const latestByPatientId = new Map<string, { id: string; duration: number; createdAt: Date | null }>()
  if (!patientIds.length) return latestByPatientId

  const latestSessions = await getLatestV1SessionByPatientIds(db, patientIds.map((patientId) => patientId.toHexString()))
  for (const [patientId, session] of latestSessions) {
    const durationMs = session.acquisition?.durationMs
    latestByPatientId.set(patientId, {
      id: session._id,
      duration: durationMs ? Math.round(durationMs / 1000) : 0,
      createdAt: session.localCompletedAt || session.createdAt || null,
    })
  }

  return latestByPatientId
}

function getPatientStatus(createdAt: Date | null): PatientStatus {
  const missedDays = getMissedDays(createdAt)
  if (missedDays === 0) return 'doing_well'
  if (missedDays <= 2) return 'worth_checking'
  return 'needs_attention'
}

function getStatusLabel(status: PatientStatus) {
  if (status === 'doing_well') return 'Doing well'
  if (status === 'worth_checking') return 'Worth checking'
  return 'Needs attention'
}

function formatLastSpoken(createdAt: Date | null) {
  const missedDays = getMissedDays(createdAt)
  if (!createdAt || missedDays >= 999) return 'No interaction yet'

  const time = new Intl.DateTimeFormat('en-SG', {
    hour: 'numeric',
    minute: '2-digit',
    timeZone: 'Asia/Singapore',
  }).format(createdAt).replace(/\s/g, '').toLowerCase()

  if (missedDays === 0) return `Today, ${time}`
  if (missedDays === 1) return `Yesterday, ${time}`
  return `No interaction for ${missedDays} days`
}
