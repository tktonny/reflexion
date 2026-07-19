import type { RequestHandler } from 'expo-router/server'
import { MongoClient, ObjectId, type Document } from 'mongodb'

declare const process: {
  env: Record<string, string | undefined>
}

const DB_NAME = 'ref'
const PAIRING_COLLECTION_NAME = 'MirrorPairingSessions'
const CONFIG_COLLECTION_NAME = 'NursePatientConfig'

function serializeDocument(document: Document) {
  return JSON.parse(JSON.stringify(document)) as unknown
}

export const POST: RequestHandler = async (request) => {
  const uri = process.env.MONGODB_URI
  if (!uri) {
    return Response.json({ success: false, reason: 'missing_mongodb_uri' }, { status: 500 })
  }

  const body = (await request.json().catch(() => null)) as {
    deviceId?: string
    pairingCode?: string
  } | null
  const deviceId = body?.deviceId?.trim()
  const pairingCode = body?.pairingCode?.replace(/\D/g, '')

  if (!deviceId || !ObjectId.isValid(deviceId) || !pairingCode) {
    return Response.json({ success: false, reason: 'invalid_pairing_status_request' }, { status: 400 })
  }

  const client = new MongoClient(uri)
  await client.connect()

  try {
    const db = client.db(DB_NAME)
    const session = await db.collection(PAIRING_COLLECTION_NAME).findOne({
      deviceId,
      pairingCode,
    })

    if (!session || session.status !== 'paired') {
      return Response.json({ success: true, paired: false })
    }

    const patientId =
      session.patientId && ObjectId.isValid(String(session.patientId))
        ? new ObjectId(String(session.patientId))
        : null
    const config = patientId
      ? await db.collection(CONFIG_COLLECTION_NAME).findOne({
          'patients._id': patientId,
        })
      : await db.collection(CONFIG_COLLECTION_NAME).findOne({
          'patients.mirrorId': new ObjectId(deviceId),
        })

    if (!config) {
      return Response.json({ success: true, paired: false })
    }

    const patient = Array.isArray(config.patients)
      ? config.patients.find((candidate: { _id?: ObjectId; mirrorId?: ObjectId }) =>
          patientId
            ? candidate._id?.toHexString?.() === patientId.toHexString()
            : candidate.mirrorId?.toHexString?.() === deviceId,
        )
      : null

    if (!patient) {
      return Response.json({ success: true, paired: false })
    }

    if (patientId && config._id) {
      await db.collection(CONFIG_COLLECTION_NAME).updateOne(
        { _id: config._id },
        {
          $set: {
            'patients.$[oldPatient].mirrorId': null,
            'patients.$[oldPatient].mirrorVerified': false,
            'patients.$[oldPatient].mirrorPairingStatus': 'replaced',
            updatedAt: new Date(),
          },
        },
        {
          arrayFilters: [
            {
              'oldPatient._id': { $ne: patientId },
              'oldPatient.mirrorId': new ObjectId(deviceId),
            },
          ],
        },
      )
    }

    const serializedConfig = serializeDocument({
      ...config,
      patients: Array.isArray(config.patients)
        ? config.patients.map((candidate: { _id?: ObjectId; mirrorId?: ObjectId }) =>
            patientId &&
            candidate._id?.toHexString?.() !== patientId.toHexString() &&
            candidate.mirrorId?.toHexString?.() === deviceId
              ? {
                  ...candidate,
                  mirrorId: null,
                  mirrorVerified: false,
                  mirrorPairingStatus: 'replaced',
                }
              : candidate,
          )
        : config.patients,
    })

    return Response.json({
      success: true,
      paired: true,
      deviceId,
      authToken: session.authToken,
      nurseId: config._id?.toHexString?.(),
      patientId: patientId?.toHexString?.() || patient._id?.toHexString?.(),
      language: patient?.preferredLanguage || 'english',
      timezone: patient?.timezone || session.timezone || 'Asia/Singapore',
      nursePatientConfig: serializedConfig,
    })
  } finally {
    await client.close()
  }
}
