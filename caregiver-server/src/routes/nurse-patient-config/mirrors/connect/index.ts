import { Router } from 'express'
import { ObjectId } from 'mongodb'
import { asyncHandler } from '../../../../lib/asyncHandler.js'
import {
  DB_NAME,
  MIRROR_MAP_COLLECTION,
  NURSE_CONFIG_COLLECTION,
  PAIRING_COLLECTION,
} from '../../../../lib/constants.js'
import { withMongo } from '../../../../lib/mongo.js'
import { normalizePairingCode, resolveMirrorPairing } from '../../../../lib/patients.js'

type ConnectMirrorBody = {
  nurseId?: string
  patientId?: string
  pairingCode?: string
  mirrorName?: string
  timezone?: string
}

export const mirrorConnectRouter = Router()

mirrorConnectRouter.post('/', asyncHandler(async (request, response) => {
  const body = request.body as ConnectMirrorBody | null
  const validationError = validateBody(body)
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

    if (hasExistingMirrorConnection(patient)) {
      response.status(409).json({ error: 'This patient already has a linked mirror. Delete the existing connection first.' })
      return
    }

    const existingPatientMap = await mapCollection.findOne({ nurseId, patientId })
    if (existingPatientMap) {
      response.status(409).json({ error: 'This patient already has a mirror map. Delete the existing connection first.' })
      return
    }

    const pairing = await resolveMirrorPairing(pairingCollection, body!.pairingCode).catch((error: unknown) => {
      return error instanceof Error ? error : new Error('Pairing code is not valid or has expired.')
    })
    if (pairing instanceof Error || !pairing) {
      response.status(400).json({ error: pairing instanceof Error ? pairing.message : 'Pairing code is not valid or has expired.' })
      return
    }

    const mirrorId = new ObjectId(pairing.deviceId)
    const existingMirrorMap = await mapCollection.findOne({ mirrorId })
    if (existingMirrorMap) {
      response.status(409).json({ error: 'This mirror is already linked to a patient. Delete that connection first.' })
      return
    }

    const existingMirrorPatient = await configCollection.findOne({ 'patients.mirrorId': mirrorId })
    if (existingMirrorPatient) {
      response.status(409).json({ error: 'This mirror is already assigned to a patient. Delete that connection first.' })
      return
    }

    const mirrorName =
      body!.mirrorName?.trim() ||
      (typeof patient.mirrorName === 'string' && patient.mirrorName.trim()) ||
      `Mirror for ${patient.name || 'patient'}`
    const timezone = body!.timezone?.trim() || pairing.timezone || patient.timezone || 'Asia/Singapore'

    await configCollection.updateOne(
      { _id: nurseId, 'patients._id': patientId },
      {
        $set: {
          'patients.$.mirrorId': mirrorId,
          'patients.$.mirrorName': mirrorName,
          'patients.$.mirrorVerified': true,
          'patients.$.mirrorPairingStatus': 'paired',
          'patients.$.mirrorPairingCode': pairing.pairingCode,
          'patients.$.mirrorPairedAt': now,
          'patients.$.deviceAuthToken': pairing.authToken || '',
          'patients.$.timezone': timezone,
          updatedAt: now,
        },
      },
    )

    await mapCollection.insertOne({
      mirrorId,
      nurseId,
      patientId,
      mirrorName,
      patientName: patient.name || '',
      pairingCode: pairing.pairingCode,
      createdAt: now,
      updatedAt: now,
    })

    await Promise.all([
      mapCollection.createIndex(
        { patientId: 1 },
        {
          unique: true,
          partialFilterExpression: { patientId: { $exists: true } },
        },
      ),
      mapCollection.createIndex(
        { mirrorId: 1 },
        {
          unique: true,
          partialFilterExpression: { mirrorId: { $exists: true } },
        },
      ),
    ]).catch(() => {
      // Existing duplicate test data should not prevent explicit checks above from protecting new writes.
    })

    await pairingCollection.updateOne(
      { pairingCode: pairing.pairingCode },
      {
        $set: {
          status: 'paired',
          nurseId,
          patientId,
          pairedAt: now,
          updatedAt: now,
        },
      },
    )

    response.json({
      success: true,
      patientId: patientId.toHexString(),
      mirrorId: mirrorId.toHexString(),
      mirrorName,
      mirrorPairingStatus: 'paired',
      mirrorPairedAt: now.toISOString(),
    })
  })
}))

function validateBody(body: ConnectMirrorBody | null) {
  if (!body) return 'Request body is required.'
  if (!body.nurseId || !ObjectId.isValid(body.nurseId)) return 'Nurse id is invalid.'
  if (!body.patientId || !ObjectId.isValid(body.patientId)) return 'Patient id is invalid.'
  if (!normalizePairingCode(body.pairingCode)) return 'Pairing code is required.'
  return ''
}

function hasExistingMirrorConnection(patient: Record<string, unknown>) {
  return Boolean(
    patient.mirrorId ||
      patient.deviceAuthToken ||
      patient.mirrorVerified ||
      patient.mirrorPairingStatus === 'paired' ||
      patient.mirrorPairingCode,
  )
}
