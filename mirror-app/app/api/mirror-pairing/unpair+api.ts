import type { RequestHandler } from 'expo-router/server'
import { MongoClient } from 'mongodb'

declare const process: {
  env: Record<string, string | undefined>
}

const DB_NAME = 'ref'
const COLLECTION_NAME = 'MirrorPairingSessions'

export const OPTIONS: RequestHandler = async () => {
  return Response.json({ ok: true })
}

// Release the server-side pairing so a factory/support reset does not leave an orphaned
// status:'paired' session (which would 409 future re-pairing and keep device-status reporting
// "paired"). Best-effort: the client calls this before wiping local state.
export const POST: RequestHandler = async (request) => {
  const uri = process.env.MONGODB_URI
  if (!uri) {
    return Response.json({ success: false, reason: 'missing_mongodb_uri' }, { status: 500 })
  }

  const body = (await request.json().catch(() => null)) as {
    deviceId?: string
    authToken?: string
  } | null
  const deviceId = body?.deviceId?.trim()
  const authToken = body?.authToken?.trim()
  if (!deviceId) {
    return Response.json({ success: false, reason: 'missing_device_id' }, { status: 400 })
  }
  // Require the device's own secret so a caller can't unpair arbitrary devices by guessing a deviceId.
  if (!authToken) {
    return Response.json({ success: false, reason: 'missing_auth_token' }, { status: 401 })
  }

  const client = new MongoClient(uri)
  await client.connect()

  try {
    const collection = client.db(DB_NAME).collection(COLLECTION_NAME)
    // Only the owning device (deviceId + its authToken) can release the session.
    const filter: Record<string, unknown> = { deviceId, authToken }
    const result = await collection.updateMany(filter, {
      $set: { status: 'released', updatedAt: new Date() },
    })
    return Response.json({ success: true, released: result.modifiedCount })
  } finally {
    await client.close()
  }
}
