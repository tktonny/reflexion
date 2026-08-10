import { Router } from 'express'
import { ObjectId } from 'mongodb'
import { asyncHandler } from '../../../../lib/asyncHandler.js'
import { NURSE_CONFIG_COLLECTION } from '../../../../lib/constants.js'
import { getDb } from '../../../../lib/mongo.js'
import { normalizePairingCode } from '../../../../lib/patients.js'
import { BridgeError, claimV1Pairing, ensureV1Identity } from '../../../../lib/legacyV1Bridge.js'

type ConnectMirrorBody = {
  nurseId?: string
  patientId?: string
  pairingCode?: string
  mirrorName?: string
  timezone?: string
}

// v1-backed: the caregiver app still POSTs the legacy shape { nurseId, patientId, pairingCode, ... },
// but we mirror the nurse/patient into v1 and claim the v1 device_pairing the mirror created (the mirror
// now runs Pairing v2, so its code lives in v1 device_pairings, not the old MirrorPairingSessions).
// See LEGACY_V1_ADAPTER.md §3/§4.
export const mirrorConnectRouter = Router()

mirrorConnectRouter.post('/', asyncHandler(async (request, response) => {
  const body = request.body as ConnectMirrorBody | null
  const validationError = validateBody(body)
  if (validationError) {
    response.status(400).json({ error: validationError })
    return
  }

  const db = await getDb()
  const nurseId = new ObjectId(body!.nurseId!)
  const patientId = new ObjectId(body!.patientId!)

  const config = await db.collection<any>(NURSE_CONFIG_COLLECTION).findOne({ _id: nurseId, 'patients._id': patientId })
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

  // Mirror the legacy nurse/patient into v1 (idempotent even if the batch migration has not run yet).
  const { tenantId, userId, patientId: v1PatientId } = await ensureV1Identity(db, config, patient)
  const mirrorName =
    body!.mirrorName?.trim() ||
    (typeof patient.mirrorName === 'string' && patient.mirrorName.trim()) ||
    `Mirror for ${patient.name || 'patient'}`
  const timezone = body!.timezone?.trim() || patient.timezone || 'Asia/Singapore'
  const pairingCode = normalizePairingCode(body!.pairingCode)!

  try {
    const claim = await claimV1Pairing({
      pairingCode, tenantId, userId, patientId: v1PatientId, patientDisplayName: patient.name || '',
      mirrorName, correlationId: (request as { requestId?: string }).requestId,
    })
    const now = claim.pairedAt

    // Transition double-write: keep the legacy patient's mirror fields in sync so the caregiver app's
    // mirror list / dashboard still show "paired" until those read routes are adapted to v1 (#2).
    await db.collection<any>(NURSE_CONFIG_COLLECTION).updateOne(
      { _id: nurseId, 'patients._id': patientId },
      {
        $set: {
          'patients.$.mirrorId': claim.deviceId,
          'patients.$.mirrorName': claim.mirrorName,
          'patients.$.mirrorVerified': true,
          'patients.$.mirrorPairingStatus': 'paired',
          'patients.$.mirrorPairingCode': pairingCode,
          'patients.$.mirrorPairedAt': now,
          'patients.$.timezone': timezone,
          updatedAt: now,
        },
      },
    )

    response.json({
      success: true,
      patientId: patientId.toHexString(),
      mirrorId: claim.deviceId,
      mirrorName: claim.mirrorName,
      mirrorPairingStatus: 'paired',
      mirrorPairedAt: now.toISOString(),
    })
  } catch (error) {
    const status = error instanceof BridgeError ? error.status : 400
    response.status(status).json({
      error: error instanceof Error ? error.message : 'Pairing code is not valid or has expired.',
    })
  }
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
      patient.mirrorVerified ||
      patient.mirrorPairingStatus === 'paired',
  )
}
