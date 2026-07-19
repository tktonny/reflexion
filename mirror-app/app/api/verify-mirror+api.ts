import type { RequestHandler } from 'expo-router/server'
import { MongoClient, ObjectId, type Document } from 'mongodb'

declare const process: {
  env: Record<string, string | undefined>
}

const DB_NAME = 'ref'
const COLLECTION_NAME = 'NursePatientConfig'

function serializeDocument(document: Document) {
  return JSON.parse(JSON.stringify(document)) as unknown
}

export const POST: RequestHandler = async (request) => {
  const uri = process.env.MONGODB_URI
  if (!uri) {
    return Response.json(
      { success: false, reason: 'missing_mongodb_uri' },
      { status: 500 },
    )
  }

  const body = (await request.json().catch(() => null)) as { mirrorId?: string } | null
  const mirrorId = body?.mirrorId
  if (!mirrorId) {
    return Response.json(
      { success: false, reason: 'missing_mirror_id' },
      { status: 400 },
    )
  }
  if (!ObjectId.isValid(mirrorId)) {
    return Response.json(
      { success: false, reason: 'invalid_mirror_id' },
      { status: 400 },
    )
  }

  const client = new MongoClient(uri)
  await client.connect()

  try {
    const updatedConfig = await client
      .db(DB_NAME)
      .collection(COLLECTION_NAME)
      .findOneAndUpdate(
        { 'patients.mirrorId': new ObjectId(mirrorId) },
        {
          $set: {
            'patients.$.mirrorVerified': true,
            updatedAt: new Date(),
          },
        },
        { returnDocument: 'after' },
      )

    if (!updatedConfig) {
      return Response.json(
        { success: false, reason: 'mirror_not_found' },
        { status: 404 },
      )
    }

    return Response.json({
      success: true,
      nursePatientConfig: serializeDocument(updatedConfig),
    })
  } catch (error) {
    return Response.json(
      {
        success: false,
        reason:
          error instanceof Error ? error.message : 'unknown_verify_mirror_error',
      },
      { status: 500 },
    )
  } finally {
    await client.close()
  }
}
