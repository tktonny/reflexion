import { Router } from 'express'
import { ObjectId } from 'mongodb'
import { asyncHandler } from '../../lib/asyncHandler.js'
import {
  DB_NAME,
  MIRROR_MAP_COLLECTION,
  NURSE_CONFIG_COLLECTION,
  PAIRING_COLLECTION,
} from '../../lib/constants.js'
import { withMongo } from '../../lib/mongo.js'
import {
  buildPatientDocuments,
  getPairedMirrorIds,
  type PatientBody,
  replaceMirrorAssignments,
  markPairingsPaired,
} from '../../lib/patients.js'
import { hashPassword } from '../../lib/password.js'
import {
  ALERT_SENSITIVITIES,
  GENDERS,
  isOneOf,
  LANGUAGES,
  RELATIONSHIPS,
  SUMMARY_TIMES,
  TOPICS,
} from '../../lib/validation.js'

type AccountBody = {
  name?: string
  email?: string
  password?: string
  phoneNumber?: string
  relationshipToElderly?: string
}

type NotificationsBody = {
  pushNotificationsEnabled?: boolean
  alertSensitivity?: string
  preferredDailySummaryTime?: string
}

type CreateConfigBody = {
  account?: AccountBody
  patients?: PatientBody[]
  notifications?: NotificationsBody
}

export const createConfigRouter = Router()

createConfigRouter.post('/', asyncHandler(async (request, response) => {
  const body = request.body as CreateConfigBody
  const validationError = validateBody(body)
  if (validationError) {
    response.status(400).json({ error: validationError })
    return
  }

  const now = new Date()
  const account = body.account as Required<AccountBody>
  const notifications = body.notifications as Required<NotificationsBody>
  const patients = body.patients as PatientBody[]
  const nurseId = new ObjectId()

  await withMongo(async (client) => {
    const db = client.db(DB_NAME)
    const patientDocuments = await buildPatientDocuments(db, patients)
    const newlyPairedMirrorIds = getPairedMirrorIds(patientDocuments)
    await replaceMirrorAssignments(db, newlyPairedMirrorIds)

    const document = {
      _id: nurseId,
      name: account.name.trim(),
      email: account.email.trim().toLowerCase(),
      passwordHash: hashPassword(account.password),
      phoneNumber: account.phoneNumber.trim(),
      relationshipToElderly: account.relationshipToElderly,
      pushNotificationsEnabled: notifications.pushNotificationsEnabled,
      alertSensitivity: notifications.alertSensitivity,
      preferredDailySummaryTime: notifications.preferredDailySummaryTime,
      patients: patientDocuments,
      createdAt: now,
      updatedAt: now,
    }
    const result = await db.collection(NURSE_CONFIG_COLLECTION).insertOne(document)

    const mirrorMaps = document.patients.map((patient) => ({
      mirrorId: patient.mirrorId,
      nurseId,
      patientId: patient._id,
      mirrorName: patient.mirrorName,
      patientName: patient.name,
      pairingCode: patient.mirrorPairingCode,
      createdAt: now,
      updatedAt: now,
    })).filter((item) => item.mirrorId)
    if (mirrorMaps.length > 0) {
      await db.collection(MIRROR_MAP_COLLECTION).insertMany(mirrorMaps)
    }

    await markPairingsPaired(db.collection(PAIRING_COLLECTION), document.patients, nurseId)

    response.json({
      insertedId: result.insertedId.toHexString(),
      nurseId: nurseId.toHexString(),
      name: document.name,
      email: document.email,
      mirrorMapCount: document.patients.length,
      patientCount: document.patients.length,
    })
  })
}))

function validateBody(body: CreateConfigBody) {
  const account = body.account
  if (!account) return 'Account details are required.'
  if (!account.name?.trim()) return 'Name is required.'
  if (!account.email?.trim() || !account.email.includes('@')) return 'A valid email is required.'
  if (!account.password || account.password.length < 8) return 'Password must be at least 8 characters.'
  if (!account.phoneNumber?.trim()) return 'Phone number is required.'
  if (!isOneOf(account.relationshipToElderly, RELATIONSHIPS)) {
    return 'Relationship to elderly person is invalid.'
  }

  const patientsError = validatePatients(body.patients)
  if (patientsError) return patientsError

  const notifications = body.notifications
  if (!notifications) return 'Notification settings are required.'
  if (typeof notifications.pushNotificationsEnabled !== 'boolean') {
    return 'Push notification setting is required.'
  }
  if (!isOneOf(notifications.alertSensitivity, ALERT_SENSITIVITIES)) {
    return 'Alert sensitivity is invalid.'
  }
  if (!isOneOf(notifications.preferredDailySummaryTime, SUMMARY_TIMES)) {
    return 'Preferred daily summary time is invalid.'
  }

  return ''
}

function validatePatients(patients: PatientBody[] | undefined) {
  if (!Array.isArray(patients) || patients.length === 0) {
    return 'At least one elderly profile is required.'
  }

  for (const patient of patients) {
    if (!patient.name?.trim()) return 'Each elderly profile needs a name.'
    if (
      typeof patient.age !== 'number' ||
      !Number.isInteger(patient.age) ||
      patient.age < 1 ||
      patient.age > 130
    ) {
      return 'Each elderly profile needs a valid age.'
    }
    if (!isOneOf(patient.gender, GENDERS)) return 'Patient gender is invalid.'
    if (!isOneOf(patient.preferredLanguage, LANGUAGES)) return 'Patient preferred language is invalid.'
    if (!patient.usualWakeTime?.trim()) return 'Each elderly profile needs a usual wake time.'
    if (!Array.isArray(patient.keyTopics) || patient.keyTopics.length === 0) {
      return 'Each elderly profile needs at least one key topic.'
    }
    if (patient.keyTopics.some((topic) => !isOneOf(topic, TOPICS))) {
      return 'One or more key topics are invalid.'
    }
    if (patient.keyTopics.includes('others') && !patient.keyTopicsOtherText?.trim()) {
      return 'Other topic text is required when Others is selected.'
    }
  }

  return ''
}
