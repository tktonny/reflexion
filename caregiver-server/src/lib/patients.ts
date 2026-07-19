import { ObjectId, type Collection, type Db } from 'mongodb'
import {
  MIRROR_MAP_COLLECTION,
  NURSE_CONFIG_COLLECTION,
  PAIRING_COLLECTION,
} from './constants.js'

export type PatientBody = {
  name?: string
  phoneNumber?: string
  age?: number
  gender?: string
  preferredLanguage?: string
  usualWakeTime?: string
  speechOrHearingConditions?: string
  photoUrl?: string
  keyTopics?: string[]
  keyTopicsOtherText?: string
  mirrorName?: string
  mirrorPairingCode?: string
  timezone?: string
}

export function normalizePairingCode(value?: string) {
  return value?.replace(/\D/g, '').slice(0, 6) || ''
}

export async function resolveMirrorPairing(collection: Collection, rawCode?: string) {
  const pairingCode = normalizePairingCode(rawCode)
  if (!pairingCode) return null

  const session = await collection.findOne({
    pairingCode,
    status: 'pending',
    expiresAt: { $gt: new Date() },
  })
  if (!session?.deviceId || !ObjectId.isValid(String(session.deviceId))) {
    throw new Error(`Pairing code ${pairingCode} is not valid or has expired.`)
  }

  return {
    deviceId: String(session.deviceId),
    pairingCode,
    authToken: typeof session.authToken === 'string' ? session.authToken : undefined,
    timezone: typeof session.timezone === 'string' ? session.timezone : undefined,
  }
}

export async function buildPatientDocuments(
  db: Db,
  patients: PatientBody[],
  existingCount = 0,
) {
  const now = new Date()
  const pairingCollection = db.collection(PAIRING_COLLECTION)

  return Promise.all(
    patients.map(async (patient, index) => {
      const pairing = await resolveMirrorPairing(pairingCollection, patient.mirrorPairingCode)
      const mirrorId = pairing?.deviceId ? new ObjectId(pairing.deviceId) : null

      return {
        _id: new ObjectId(),
        name: patient.name?.trim(),
        phoneNumber: patient.phoneNumber?.trim() || undefined,
        age: patient.age,
        gender: patient.gender,
        preferredLanguage: patient.preferredLanguage,
        usualWakeTime: patient.usualWakeTime?.trim(),
        speechOrHearingConditions: patient.speechOrHearingConditions?.trim() || undefined,
        photoUrl: patient.photoUrl?.trim() || undefined,
        keyTopics: patient.keyTopics,
        keyTopicsOtherText: patient.keyTopicsOtherText?.trim() || undefined,
        mirrorId,
        mirrorName: patient.mirrorName?.trim() || `Mirror ${existingCount + index + 1}`,
        mirrorVerified: Boolean(pairing),
        mirrorPairingStatus: pairing ? 'paired' : 'awaiting_pairing',
        mirrorPairingCode: pairing?.pairingCode || normalizePairingCode(patient.mirrorPairingCode) || undefined,
        mirrorPairedAt: pairing ? now : undefined,
        deviceAuthToken: pairing?.authToken || undefined,
        timezone: patient.timezone?.trim() || pairing?.timezone || 'Asia/Singapore',
      }
    }),
  )
}

export async function clearExistingMirrorAssignments(collection: Collection, mirrorIds: ObjectId[]) {
  await collection.updateMany(
    { 'patients.mirrorId': { $in: mirrorIds } },
    {
      $set: {
        'patients.$[patient].mirrorId': null,
        'patients.$[patient].mirrorVerified': false,
        'patients.$[patient].mirrorPairingStatus': 'replaced',
      },
      $unset: {
        'patients.$[patient].mirrorPairingCode': '',
        'patients.$[patient].deviceAuthToken': '',
      },
    },
    {
      arrayFilters: [{ 'patient.mirrorId': { $in: mirrorIds } }],
    },
  )
}

export async function markPairingsPaired(
  collection: Collection,
  patients: Array<{ mirrorPairingCode?: string; mirrorId?: ObjectId | null; _id: ObjectId }>,
  nurseId: ObjectId,
) {
  await Promise.all(
    patients
      .filter((patient) => patient.mirrorPairingCode && patient.mirrorId)
      .map((patient) =>
        collection.updateOne(
          { pairingCode: patient.mirrorPairingCode },
          {
            $set: {
              status: 'paired',
              nurseId,
              patientId: patient._id,
              pairedAt: new Date(),
              updatedAt: new Date(),
            },
          },
        ),
      ),
  )
}

export async function replaceMirrorAssignments(db: Db, mirrorIds: ObjectId[]) {
  if (mirrorIds.length === 0) return

  await clearExistingMirrorAssignments(db.collection(NURSE_CONFIG_COLLECTION), mirrorIds)
  await db.collection(MIRROR_MAP_COLLECTION).deleteMany({
    mirrorId: { $in: mirrorIds },
  })
}

export function getPairedMirrorIds(patients: Array<{ mirrorId?: ObjectId | null }>) {
  return patients
    .map((patient) => patient.mirrorId)
    .filter((mirrorId): mirrorId is ObjectId => Boolean(mirrorId))
}

export function validatePatients(patients: PatientBody[] | undefined) {
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
  }

  return ''
}
