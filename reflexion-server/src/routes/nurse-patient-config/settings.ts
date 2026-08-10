import { Router } from 'express'
import { ObjectId, type Db } from 'mongodb'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { DB_NAME, NURSE_CONFIG_COLLECTION } from '../../lib/constants.js'
import { withMongo } from '../../lib/mongo.js'
import { collections } from '../../v1/platform/collections.js'
import {
  ALERT_SENSITIVITIES,
  GENDERS,
  isOneOf,
  LANGUAGES,
  SUMMARY_TIMES,
  TOPICS,
} from '../../lib/validation.js'
import type { StoredPatient } from '../../lib/types.js'

// Caregiver-editable settings for the legacy NursePatientConfig document, which is still where the
// caregiver profile and the per-patient conversation preferences live (usualWakeTime, keyTopics,
// speechOrHearingConditions and the notification preferences have no v1 equivalent). Both handlers are
// PARTIAL updates: only keys present in the body are written, so a client that has not loaded a field
// can never blank it. `nurseId` is required — these routes deliberately do NOT inherit the
// "fall back to the most recent config" behaviour, which leaks one caregiver's record to another.

export const settingsRouter = Router()

type NurseSettingsBody = {
  nurseId?: string
  name?: unknown
  phoneNumber?: unknown
  pushNotificationsEnabled?: unknown
  alertSensitivity?: unknown
  preferredDailySummaryTime?: unknown
  storeSessionSummaries?: unknown
}

settingsRouter.patch('/', asyncHandler(async (request, response) => {
  const body = (request.body || {}) as NurseSettingsBody
  if (!body.nurseId || !ObjectId.isValid(String(body.nurseId))) {
    response.status(400).json({ error: 'A valid nurse id is required.' })
    return
  }

  const update: Record<string, unknown> = {}
  if ('name' in body) {
    const name = String(body.name ?? '').trim()
    if (!name) {
      response.status(400).json({ error: 'Your name cannot be empty.' })
      return
    }
    update.name = name.slice(0, 120)
  }
  if ('phoneNumber' in body) update.phoneNumber = String(body.phoneNumber ?? '').trim().slice(0, 40)
  if ('pushNotificationsEnabled' in body) {
    if (typeof body.pushNotificationsEnabled !== 'boolean') {
      response.status(400).json({ error: 'Push notification setting is invalid.' })
      return
    }
    update.pushNotificationsEnabled = body.pushNotificationsEnabled
  }
  if ('alertSensitivity' in body) {
    if (!isOneOf(body.alertSensitivity, ALERT_SENSITIVITIES)) {
      response.status(400).json({ error: 'Alert sensitivity is invalid.' })
      return
    }
    update.alertSensitivity = body.alertSensitivity
  }
  if ('preferredDailySummaryTime' in body) {
    if (!isOneOf(body.preferredDailySummaryTime, SUMMARY_TIMES)) {
      response.status(400).json({ error: 'Preferred daily summary time is invalid.' })
      return
    }
    update.preferredDailySummaryTime = body.preferredDailySummaryTime
  }
  if ('storeSessionSummaries' in body) {
    if (typeof body.storeSessionSummaries !== 'boolean') {
      response.status(400).json({ error: 'Store session summaries setting is invalid.' })
      return
    }
    update.storeSessionSummaries = body.storeSessionSummaries
  }

  if (!Object.keys(update).length) {
    response.status(400).json({ error: 'No settings were provided.' })
    return
  }

  await withMongo(async (client) => {
    const db = client.db(DB_NAME)
    const collection = db.collection(NURSE_CONFIG_COLLECTION)
    const nurseObjectId = new ObjectId(String(body.nurseId))
    const result = await collection.findOneAndUpdate(
      { _id: nurseObjectId },
      { $set: { ...update, updatedAt: new Date() } },
      { returnDocument: 'after' },
    )
    if (!result) {
      response.status(404).json({ error: 'Nurse config not found' })
      return
    }
    await mirrorCaregiverSettingsToV1(db, nurseObjectId.toHexString(), update)
    response.json(serializeSettings(result))
  })
}))

type PatientSettingsBody = { nurseId?: string } & Partial<Record<keyof StoredPatient, unknown>>

settingsRouter.patch('/patients/:patientId', asyncHandler(async (request, response) => {
  const body = (request.body || {}) as PatientSettingsBody
  const patientId = String(request.params.patientId || '')
  if (!body.nurseId || !ObjectId.isValid(String(body.nurseId))) {
    response.status(400).json({ error: 'A valid nurse id is required.' })
    return
  }
  if (!ObjectId.isValid(patientId)) {
    response.status(400).json({ error: 'A valid patient id is required.' })
    return
  }

  const patientUpdate: Record<string, unknown> = {}
  const fail = (message: string) => {
    response.status(400).json({ error: message })
    return null
  }

  if ('name' in body) {
    const name = String(body.name ?? '').trim()
    if (!name) return void fail('Their name cannot be empty.')
    patientUpdate.name = name.slice(0, 120)
  }
  if ('phoneNumber' in body) patientUpdate.phoneNumber = String(body.phoneNumber ?? '').trim().slice(0, 40)
  if ('age' in body) {
    const age = Number(body.age)
    if (!Number.isInteger(age) || age < 1 || age > 130) return void fail('Age must be a whole number between 1 and 130.')
    patientUpdate.age = age
  }
  if ('gender' in body) {
    if (!isOneOf(body.gender, GENDERS)) return void fail('Gender is invalid.')
    patientUpdate.gender = body.gender
  }
  if ('preferredLanguage' in body) {
    if (!isOneOf(body.preferredLanguage, LANGUAGES)) return void fail('Preferred language is invalid.')
    patientUpdate.preferredLanguage = body.preferredLanguage
  }
  if ('usualWakeTime' in body) {
    const usualWakeTime = String(body.usualWakeTime ?? '').trim()
    if (!usualWakeTime) return void fail('A usual wake time is required.')
    patientUpdate.usualWakeTime = usualWakeTime.slice(0, 40)
  }
  if ('speechOrHearingConditions' in body) {
    patientUpdate.speechOrHearingConditions = String(body.speechOrHearingConditions ?? '').trim().slice(0, 500) || null
  }
  if ('photoUrl' in body) patientUpdate.photoUrl = String(body.photoUrl ?? '')
  if ('keyTopics' in body) {
    const keyTopics = body.keyTopics
    if (!Array.isArray(keyTopics) || !keyTopics.length) return void fail('Pick at least one topic they enjoy.')
    if (keyTopics.some((topic) => !isOneOf(topic, TOPICS))) return void fail('One or more topics are invalid.')
    patientUpdate.keyTopics = keyTopics
    const otherText = String(('keyTopicsOtherText' in body ? body.keyTopicsOtherText : '') ?? '').trim()
    if (keyTopics.includes('others') && !otherText) return void fail('Tell us the other topic they enjoy.')
    patientUpdate.keyTopicsOtherText = otherText.slice(0, 200) || null
  } else if ('keyTopicsOtherText' in body) {
    patientUpdate.keyTopicsOtherText = String(body.keyTopicsOtherText ?? '').trim().slice(0, 200) || null
  }

  if (!Object.keys(patientUpdate).length) {
    response.status(400).json({ error: 'No profile changes were provided.' })
    return
  }

  await withMongo(async (client) => {
    const db = client.db(DB_NAME)
    const collection = db.collection(NURSE_CONFIG_COLLECTION)
    const nurseObjectId = new ObjectId(String(body.nurseId))
    const patientObjectId = new ObjectId(patientId)
    const result = await collection.findOneAndUpdate(
      { _id: nurseObjectId, 'patients._id': patientObjectId },
      {
        $set: {
          ...Object.fromEntries(Object.entries(patientUpdate).map(([key, value]) => [`patients.$[target].${key}`, value])),
          updatedAt: new Date(),
        },
      },
      { arrayFilters: [{ 'target._id': patientObjectId }], returnDocument: 'after' },
    )
    if (!result) {
      response.status(404).json({ error: 'Loved one profile not found' })
      return
    }

    const patients = (result.patients || []) as StoredPatient[]
    const patient = patients.find((item) => item._id?.toHexString?.() === patientId)
    if (!patient) {
      response.status(404).json({ error: 'Loved one profile not found' })
      return
    }
    await syncV1Patient(db, patientId, patientUpdate)
    response.json({ patient: serializePatient(patient) })
  })
}))

/**
 * Mirrors a legacy caregiver-settings write into the v1 `users` document.
 *
 * Both API surfaces are live during the migration, and the sign-in bridge now SEEDS these fields rather than
 * re-asserting them (otherwise it would revert a v1 edit). That leaves exactly one gap: a legacy write would
 * not reach v1. Writing both here closes it, so whichever surface the caregiver used, the other agrees —
 * the same rule the password reset follows.
 */
async function mirrorCaregiverSettingsToV1(db: Db, userId: string, update: Record<string, unknown>) {
  const v1Update: Record<string, unknown> = {}
  if (typeof update.name === 'string') v1Update.name = update.name
  if (typeof update.phoneNumber === 'string') v1Update.phoneNumber = update.phoneNumber
  if (typeof update.pushNotificationsEnabled === 'boolean') {
    v1Update['notificationPreferences.pushNotificationsEnabled'] = update.pushNotificationsEnabled
  }
  if (typeof update.alertSensitivity === 'string') {
    v1Update['notificationPreferences.alertSensitivity'] = update.alertSensitivity
  }
  if (typeof update.preferredDailySummaryTime === 'string') {
    v1Update['notificationPreferences.preferredDailySummaryTime'] = update.preferredDailySummaryTime
  }
  if (!Object.keys(v1Update).length) return
  v1Update.updatedAt = new Date()
  // Update-only: a caregiver with no v1 user yet gets one from the sign-in bridge, not from here.
  await db.collection<any>(collections.users).updateOne({ _id: userId }, { $set: v1Update })
}

/**
 * Keeps the v1 `patients` read model from drifting on the two fields it shares with the legacy profile.
 * Updates only — never upserts: a v1 patient row (and its care relationship) is created by pairing a
 * mirror, and editing a name must not mint a new v1 identity as a side effect.
 */
async function syncV1Patient(db: Db, patientId: string, patientUpdate: Record<string, unknown>) {
  const v1Update: Record<string, unknown> = {}
  if (typeof patientUpdate.name === 'string') v1Update.displayName = patientUpdate.name
  if (typeof patientUpdate.preferredLanguage === 'string') v1Update.preferredLanguage = patientUpdate.preferredLanguage
  if (typeof patientUpdate.age === 'number') {
    v1Update.ageBand = ageBand(patientUpdate.age)
    v1Update['profile.age'] = patientUpdate.age
  }
  // The display-only profile now has a v1 home, so a legacy edit must land there as well or the two
  // surfaces disagree about the same loved one.
  if (typeof patientUpdate.gender === 'string') v1Update['profile.gender'] = patientUpdate.gender
  if (typeof patientUpdate.photoUrl === 'string') v1Update['profile.photoUrl'] = patientUpdate.photoUrl || null
  if (typeof patientUpdate.phoneNumber === 'string') v1Update['profile.phoneNumber'] = patientUpdate.phoneNumber || null
  if (!Object.keys(v1Update).length) return
  await db.collection<any>(collections.patients).updateOne(
    { _id: patientId },
    { $set: { ...v1Update, updatedAt: new Date() } },
  )
}

function ageBand(age: number): string | null {
  if (!Number.isFinite(age) || age <= 0) return null
  if (age < 65) return 'under_65'
  if (age < 75) return '65_74'
  if (age < 85) return '75_84'
  return '85_plus'
}

/** Mirrors the shape `GET /nurse-patient-config/latest` returns so the app can reuse one normalizer. */
export function serializePatient(patient: StoredPatient) {
  const patientId = patient._id?.toHexString?.() || ''
  return {
    id: patientId,
    patientId,
    name: patient.name || '',
    phoneNumber: patient.phoneNumber || '',
    age: patient.age || 0,
    gender: patient.gender || '',
    preferredLanguage: patient.preferredLanguage || '',
    usualWakeTime: patient.usualWakeTime || '',
    speechOrHearingConditions: patient.speechOrHearingConditions || '',
    speechSpeed: patient.speechSpeed || 'Slow',
    mirrorName: patient.mirrorName || '',
    photoUrl: patient.photoUrl || '',
    keyTopics: Array.isArray(patient.keyTopics) ? patient.keyTopics : [],
    keyTopicsOtherText: patient.keyTopicsOtherText || '',
  }
}

function serializeSettings(document: Record<string, any>) {
  return {
    nurseId: document._id?.toHexString?.() || '',
    caregiverName: document.name || '',
    email: document.email || '',
    phoneNumber: document.phoneNumber || '',
    pushNotificationsEnabled: Boolean(document.pushNotificationsEnabled),
    alertSensitivity: document.alertSensitivity || 'only_important_changes',
    preferredDailySummaryTime: document.preferredDailySummaryTime || '09:00',
    storeSessionSummaries: document.storeSessionSummaries !== false,
    patients: ((document.patients || []) as StoredPatient[]).map(serializePatient),
  }
}
