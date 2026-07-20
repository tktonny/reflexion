// Server-side device verification for +api.ts routes (token mint / conversation save).
// A paired mirror holds an authToken (from request-code, confirmed by the caregiver app);
// we verify {deviceId, authToken} against a paired MirrorPairingSessions row in DB `ref`.
// Uses a pooled MongoClient (avoids the per-request connect/close anti-pattern).

import { MongoClient } from 'mongodb'

let clientPromise: Promise<MongoClient> | null = null

function getClient(uri: string): Promise<MongoClient> {
  if (!clientPromise) clientPromise = new MongoClient(uri).connect()
  return clientPromise
}

export type DeviceVerification = { ok: boolean; nurseId?: string; patientId?: string }

export async function verifyDevice(uri: string, deviceId?: string, authToken?: string): Promise<DeviceVerification> {
  if (!deviceId || !authToken) return { ok: false }
  try {
    const db = (await getClient(uri)).db('ref')
    const session = await db.collection('MirrorPairingSessions').findOne({ deviceId, authToken, status: 'paired' })
    if (!session) return { ok: false }
    return {
      ok: true,
      nurseId: session.nurseId?.toString?.(),
      patientId: session.patientId?.toString?.(),
    }
  } catch {
    return { ok: false }
  }
}
