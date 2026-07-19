import type { RequestHandler } from 'expo-router/server'
import { MongoClient, ObjectId, type Document } from 'mongodb'

declare const process: {
  env: Record<string, string | undefined>
}

const DB_NAME = 'ref'
const PAIRING_COLLECTION_NAME = 'MirrorPairingSessions'
const CONFIG_COLLECTION_NAME = 'NursePatientConfig'
const MIRROR_MAP_COLLECTION_NAME = 'MirrorIdToNurseIdMap'

function serializeDocument(document: Document) {
  return JSON.parse(JSON.stringify(document)) as unknown
}

export const POST: RequestHandler = async (request) => {
  const uri = process.env.MONGODB_URI
  if (!uri) {
    return Response.json({ success: false, reason: 'missing_mongodb_uri' }, { status: 500 })
  }

  const body = (await request.json().catch(() => null)) as { deviceId?: string } | null
  const deviceId = body?.deviceId?.trim()

  if (!deviceId || !ObjectId.isValid(deviceId)) {
    return Response.json({ success: false, reason: 'invalid_device_id' }, { status: 400 })
  }

  const client = new MongoClient(uri)
  await client.connect()

  try {
    const db = client.db(DB_NAME)
    const mirrorId = new ObjectId(deviceId)
    const map = await db.collection(MIRROR_MAP_COLLECTION_NAME).findOne({
      mirrorId,
    })
    const mappedPatientId =
      map?.patientId && ObjectId.isValid(String(map.patientId))
        ? new ObjectId(String(map.patientId))
        : null
    const mappedNurseId =
      map?.nurseId && ObjectId.isValid(String(map.nurseId))
        ? new ObjectId(String(map.nurseId))
        : null

    if (!mappedPatientId || !mappedNurseId) {
      return Response.json({ success: true, paired: false })
    }

    const session = await db.collection(PAIRING_COLLECTION_NAME).findOne({
      deviceId,
      status: 'paired',
      nurseId: mappedNurseId,
      patientId: mappedPatientId,
    })
    if (!session) {
      return Response.json({ success: true, paired: false })
    }

    const config = await db.collection(CONFIG_COLLECTION_NAME).findOne({
      _id: mappedNurseId,
      'patients._id': mappedPatientId,
      'patients.mirrorId': mirrorId,
      'patients.mirrorVerified': true,
      'patients.mirrorPairingStatus': 'paired',
    })

    if (!config) {
      return Response.json({ success: true, paired: false })
    }

    const patient = Array.isArray(config.patients)
      ? config.patients.find((candidate: { _id?: ObjectId; mirrorId?: ObjectId }) =>
          candidate._id?.toHexString?.() === mappedPatientId.toHexString() &&
          candidate.mirrorId?.toHexString?.() === deviceId,
        )
      : null

    if (!patient?.mirrorVerified || patient.mirrorPairingStatus !== 'paired') {
      return Response.json({ success: true, paired: false })
    }

    const serializedConfig = serializeDocument(config)

    return Response.json({
      success: true,
      paired: true,
      deviceId,
      authToken: session?.authToken || patient.deviceAuthToken || '',
      nurseId: mappedNurseId.toHexString(),
      patientId: patient._id?.toHexString?.(),
      language: patient.preferredLanguage || 'english',
      timezone: patient.timezone || session?.timezone || 'Asia/Singapore',
      nursePatientConfig: serializedConfig,
    })
  } finally {
    await client.close()
  }
}
