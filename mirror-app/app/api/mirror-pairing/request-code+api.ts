import type { RequestHandler } from 'expo-router/server'
import { MongoClient, ObjectId, type Collection } from 'mongodb'

declare function require(moduleName: string): {
  randomBytes: (size: number) => { toString: (encoding: string) => string }
}

declare const process: {
  env: Record<string, string | undefined>
}

const { randomBytes } = require('crypto')

const DB_NAME = 'ref'
const COLLECTION_NAME = 'MirrorPairingSessions'

export const OPTIONS: RequestHandler = async () => {
  return Response.json({ ok: true })
}

export const POST: RequestHandler = async (request) => {
  const uri = process.env.MONGODB_URI
  if (!uri) {
    return Response.json({ success: false, reason: 'missing_mongodb_uri' }, { status: 500 })
  }

  const body = (await request.json().catch(() => null)) as {
    deviceId?: string
    timezone?: string
  } | null
  const deviceId = body?.deviceId?.trim()

  if (!deviceId || !ObjectId.isValid(deviceId)) {
    return Response.json({ success: false, reason: 'invalid_device_id' }, { status: 400 })
  }

  const client = new MongoClient(uri)
  await client.connect()

  try {
    const db = client.db(DB_NAME)
    const collection = db.collection(COLLECTION_NAME)
    const pairedSession = await collection.findOne({ deviceId, status: 'paired' })
    if (pairedSession) {
      return Response.json({
        success: false,
        reason: 'device_already_paired',
        deviceId,
      }, { status: 409 })
    }

    const now = new Date()
    const expiresAt = new Date(now.getTime() + 15 * 60 * 1000)
    const pairingCode = await createUniquePairingCode(collection)
    const authToken = randomBytes(24).toString('hex')
    const qrPayload = JSON.stringify({
      type: 'reflexion_mirror_pairing',
      deviceId,
      pairingCode,
    })

    await collection.updateOne(
      { deviceId },
      {
        $set: {
          deviceId,
          pairingCode,
          qrPayload,
          authToken,
          status: 'pending',
          timezone: body?.timezone || null,
          expiresAt,
          updatedAt: now,
        },
        $setOnInsert: {
          createdAt: now,
        },
      },
      { upsert: true },
    )

    return Response.json({
      success: true,
      deviceId,
      pairingCode,
      qrPayload,
      expiresAt: expiresAt.toISOString(),
    })
  } finally {
    await client.close()
  }
}

async function createUniquePairingCode(collection: Collection) {
  for (let attempt = 0; attempt < 8; attempt += 1) {
    const pairingCode = String(Math.floor(100000 + Math.random() * 900000))
    const existing = await collection.findOne({ pairingCode, status: 'pending' })
    if (!existing) return pairingCode
  }

  return String(Math.floor(100000 + Math.random() * 900000))
}
