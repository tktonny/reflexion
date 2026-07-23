import { Router } from 'express'
import { ObjectId, type Collection } from 'mongodb'
import { asyncHandler } from '../../../lib/asyncHandler.js'
import {
  DB_NAME,
  MIRROR_MAP_COLLECTION,
  NURSE_CONFIG_COLLECTION,
  PAIRING_COLLECTION,
} from '../../../lib/constants.js'
import { withMongo } from '../../../lib/mongo.js'
import {
  objectIdFromUnknown,
  objectIdToString,
  uniqueObjectIds,
  uniqueStrings,
} from '../../../lib/objectId.js'

type MirrorPatchBody = {
  action?: 'unlink'
  nurseId?: string
  patientId?: string
}

export const mirrorsRouter = Router()

mirrorsRouter.get('/', asyncHandler(async (request, response) => {
  const nurseId = typeof request.query.nurseId === 'string' ? request.query.nurseId : ''
  if (!nurseId || !ObjectId.isValid(nurseId)) {
    response.status(400).json({ error: 'Nurse id is required.' })
    return
  }

  await withMongo(async (client) => {
    const config = await client.db(DB_NAME).collection(NURSE_CONFIG_COLLECTION).findOne({
      _id: new ObjectId(nurseId),
    })

    response.json({
      nurseId,
      patients: Array.isArray(config?.patients)
        ? config.patients.map((patient: Record<string, unknown>, index: number) => ({
            patientId: objectIdToString(patient._id) || String(index),
            patientName: typeof patient.name === 'string' ? patient.name : `Person ${index + 1}`,
            mirrorId: objectIdToString(patient.mirrorId),
            mirrorName: typeof patient.mirrorName === 'string' ? patient.mirrorName : '',
            mirrorVerified: Boolean(patient.mirrorVerified),
            mirrorPairingStatus:
              typeof patient.mirrorPairingStatus === 'string' ? patient.mirrorPairingStatus : 'unpaired',
            mirrorPairingCode: typeof patient.mirrorPairingCode === 'string' ? patient.mirrorPairingCode : '',
            mirrorPairedAt: dateToISOString(patient.mirrorPairedAt),
            deviceAuthTokenPresent: Boolean(patient.deviceAuthToken),
            timezone: typeof patient.timezone === 'string' ? patient.timezone : 'Asia/Singapore',
          }))
        : [],
    })
  })
}))

mirrorsRouter.patch('/', asyncHandler(async (request, response) => {
  const body = request.body as MirrorPatchBody | null
  const validationError = validatePatchBody(body)
  if (validationError) {
    response.status(400).json({ error: validationError })
    return
  }

  await withMongo(async (client) => {
    const db = client.db(DB_NAME)
    const configCollection = db.collection(NURSE_CONFIG_COLLECTION)
    const pairingCollection = db.collection(PAIRING_COLLECTION)
    const mapCollection = db.collection(MIRROR_MAP_COLLECTION)
    const nurseId = new ObjectId(body!.nurseId)
    const patientId = new ObjectId(body!.patientId)
    const now = new Date()

    const config = await configCollection.findOne({ _id: nurseId, 'patients._id': patientId })
    const patient = Array.isArray(config?.patients)
      ? config.patients.find((candidate: { _id?: ObjectId }) => candidate._id?.toHexString?.() === patientId.toHexString())
      : null

    if (!config || !patient) {
      response.status(404).json({ error: 'Patient was not found.' })
      return
    }

    const unlinkResult = await unlinkMirror({
      configCollection,
      mapCollection,
      pairingCollection,
      nurseId,
      patient,
      patientId,
      now,
    })

    response.json({
      success: true,
      patientId: patientId.toHexString(),
      mirrorPairingStatus: '',
      ...unlinkResult,
    })
  })
}))

function validatePatchBody(body: MirrorPatchBody | null) {
  if (!body) return 'Request body is required.'
  if (body.action !== 'unlink') return 'Action is invalid.'
  if (!body.nurseId || !ObjectId.isValid(body.nurseId)) return 'Nurse id is invalid.'
  if (!body.patientId || !ObjectId.isValid(body.patientId)) return 'Patient id is invalid.'
  return ''
}

async function unlinkMirror({
  configCollection,
  mapCollection,
  pairingCollection,
  nurseId,
  patient,
  patientId,
  now,
}: {
  configCollection: Collection
  mapCollection: Collection
  pairingCollection: Collection
  nurseId: ObjectId
  patient: Record<string, unknown>
  patientId: ObjectId
  now: Date
}) {
  const mirrorId = objectIdFromUnknown(patient.mirrorId)
  const pairingCode = typeof patient.mirrorPairingCode === 'string' ? patient.mirrorPairingCode : ''
  const deviceAuthToken = typeof patient.deviceAuthToken === 'string' ? patient.deviceAuthToken : ''
  const initialMirrorIdValues = mirrorId ? [mirrorId, mirrorId.toHexString()] : []
  const mapRows = await mapCollection.find({
    $or: [
      { nurseId, patientId },
      ...(initialMirrorIdValues.length > 0 ? [{ mirrorId: { $in: initialMirrorIdValues } }] : []),
    ],
  }).toArray()
  const mirrorIds = uniqueObjectIds([
    mirrorId,
    ...mapRows.map((row) => objectIdFromUnknown(row.mirrorId)),
  ])
  const pairingCodes = uniqueStrings([
    pairingCode,
    ...mapRows.map((row) => (typeof row.pairingCode === 'string' ? row.pairingCode : '')),
  ])
  const mirrorIdStrings = mirrorIds.map((id) => id.toHexString())
  const mirrorIdValues = [...mirrorIds, ...mirrorIdStrings]

  await configCollection.updateOne(
    { _id: nurseId, 'patients._id': patientId },
    {
      $set: {
        'patients.$.mirrorId': null,
        'patients.$.mirrorName': '',
        'patients.$.mirrorVerified': false,
        'patients.$.mirrorPairingStatus': '',
        'patients.$.mirrorPairingCode': '',
        'patients.$.mirrorPairedAt': null,
        'patients.$.deviceAuthToken': '',
        updatedAt: now,
      },
    },
  )

  if (mirrorIdValues.length > 0) {
    await configCollection.updateMany(
      { 'patients.mirrorId': { $in: mirrorIdValues } },
      {
        $set: {
          'patients.$[patient].mirrorId': null,
          'patients.$[patient].mirrorName': '',
          'patients.$[patient].mirrorVerified': false,
          'patients.$[patient].mirrorPairingStatus': '',
          'patients.$[patient].mirrorPairingCode': '',
          'patients.$[patient].mirrorPairedAt': null,
          'patients.$[patient].deviceAuthToken': '',
          updatedAt: now,
        },
      },
      {
        arrayFilters: [{ 'patient.mirrorId': { $in: mirrorIdValues } }],
      },
    )
  }

  const mapDelete = await mapCollection.deleteMany({
    $or: [
      { nurseId, patientId },
      ...(mirrorIdValues.length > 0 ? [{ mirrorId: { $in: mirrorIdValues } }] : []),
    ],
  })
  const pairingDelete = await pairingCollection.deleteMany({
    $or: [
      ...(mirrorIdValues.length > 0 ? [{ mirrorId: { $in: mirrorIdValues } }] : []),
      ...(mirrorIdValues.length > 0 ? [{ deviceId: { $in: mirrorIdValues } }] : []),
      ...(pairingCodes.length > 0 ? [{ pairingCode: { $in: pairingCodes } }] : []),
      ...(deviceAuthToken ? [{ authToken: deviceAuthToken }] : []),
      { nurseId, patientId },
    ],
  })

  return {
    deletedMirrorMapCount: mapDelete.deletedCount || 0,
    deletedPairingSessionCount: pairingDelete.deletedCount || 0,
  }
}

function dateToISOString(value: unknown) {
  if (value instanceof Date) return value.toISOString()
  if (typeof value === 'string') return value
  return null
}
