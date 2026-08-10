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
  clearExistingMirrorAssignments,
  getPairedMirrorIds,
  markPairingsPaired,
  type PatientBody,
} from '../../lib/patients.js'
import { GENDERS, isOneOf, LANGUAGES, TOPICS } from '../../lib/validation.js'

type AddPatientsBody = {
  nurseId?: string
  patients?: PatientBody[]
}

export const addPatientsRouter = Router()

addPatientsRouter.patch('/', asyncHandler(async (request, response) => {
  const body = request.body as AddPatientsBody
  const validationError = validateBody(body)
  if (validationError) {
    response.status(400).json({ error: validationError })
    return
  }

  await withMongo(async (client) => {
    const db = client.db(DB_NAME)
    const collection = db.collection(NURSE_CONFIG_COLLECTION)
    const config = body.nurseId
      ? await collection.findOne({ _id: new ObjectId(body.nurseId) })
      : await collection.findOne({}, { sort: { createdAt: -1 } })

    if (!config?._id) {
      response.status(404).json({ error: 'Nurse config not found' })
      return
    }

    const nurseId = config._id as ObjectId
    const now = new Date()
    const existingCount = Array.isArray(config.patients) ? config.patients.length : 0
    const newPatients = await buildPatientDocuments(db, body.patients as PatientBody[], existingCount)
    const newlyPairedMirrorIds = getPairedMirrorIds(newPatients)
    const existingPatients = Array.isArray(config.patients)
      ? config.patients.map((patient: { mirrorId?: ObjectId }) =>
          newlyPairedMirrorIds.some((mirrorId) => patient.mirrorId?.toHexString?.() === mirrorId.toHexString())
            ? {
                ...patient,
                mirrorId: null,
                mirrorVerified: false,
                mirrorPairingStatus: 'replaced',
                mirrorPairingCode: undefined,
                deviceAuthToken: undefined,
              }
            : patient,
        )
      : []

    if (newlyPairedMirrorIds.length > 0) {
      await clearExistingMirrorAssignments(collection, newlyPairedMirrorIds)
      await db.collection(MIRROR_MAP_COLLECTION).deleteMany({
        mirrorId: { $in: newlyPairedMirrorIds },
      })
    }

    await collection.updateOne(
      { _id: nurseId },
      {
        $set: {
          patients: [...existingPatients, ...newPatients],
          updatedAt: now,
        },
      },
    )

    const mirrorMaps = newPatients.map((patient) => ({
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

    await markPairingsPaired(db.collection(PAIRING_COLLECTION), newPatients, nurseId)

    response.json({
      nurseId: nurseId.toHexString(),
      patientCount: newPatients.length,
    })
  })
}))

function validateBody(body: AddPatientsBody) {
  if (body.nurseId && !ObjectId.isValid(body.nurseId)) {
    return 'Nurse id is invalid.'
  }

  const patients = body.patients
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
