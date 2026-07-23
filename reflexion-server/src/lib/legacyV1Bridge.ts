// Legacy ↔ v1 bridge helpers (see LEGACY_V1_ADAPTER.md).
// Legacy nurse/patient are mirrored into the v1 normalized model, REUSING the legacy ObjectId hex as
// the v1 _id so the legacy API keeps returning the same 24-hex ids the caregiver app validates.
// claimV1Pairing replicates POST /api/v1/device-pairing-claims but is authorized by legacy nurse
// ownership (the app has no v1 token yet).

import type { Db } from 'mongodb'
import { getDb, inTransaction } from './mongo.js'
import { collections } from '../v1/platform/collections.js'
import { hashSecret, hmac, sealSecret, sha256 } from '../v1/platform/crypto.js'
import { newId, randomSecret } from '../v1/platform/ids.js'
import { appendOutbox } from '../v1/platform/outbox.js'

const EXCHANGE_TTL_MS = 5 * 60 * 1000
const RELATIONSHIP_SCOPES = [
  'patient:read', 'patient:write', 'device:assign', 'care_plan:read', 'care_plan:write', 'monitoring:read',
]

/** Error carrying an HTTP status so legacy routes can map it to their {error} shape. */
export class BridgeError extends Error {
  constructor(public status: number, message: string) { super(message) }
}

export function tenantIdForNurse(nurseHex: string): string { return `ten_${nurseHex}` }

function ageBand(age: unknown): string | null {
  const n = Number(age)
  if (!Number.isFinite(n) || n <= 0) return null
  if (n < 65) return 'under_65'
  if (n < 75) return '65_74'
  if (n < 85) return '75_84'
  return '85_plus'
}

type HexId = { toHexString(): string }
export type LegacyNurse = {
  _id: HexId; name?: string; email?: string; passwordHash?: string; phoneNumber?: string
  pushNotificationsEnabled?: boolean; alertSensitivity?: string; preferredDailySummaryTime?: string
}
export type LegacyPatient = {
  _id: HexId; name?: string; preferredLanguage?: string; timezone?: string; age?: number
  relationshipToElderly?: string; mirrorName?: string
}

/** Idempotently upsert the v1 tenant + user for a legacy nurse. Returns { tenantId, userId }. */
export async function ensureV1TenantUser(db: Db, nurse: LegacyNurse): Promise<{ tenantId: string; userId: string }> {
  const userId = nurse._id.toHexString()
  const tenantId = tenantIdForNurse(userId)
  const now = new Date()
  await db.collection<any>(collections.tenants).updateOne(
    { _id: tenantId },
    { $setOnInsert: { _id: tenantId, name: `${nurse.name || 'Caregiver'} tenant`, status: 'active', createdAt: now }, $set: { updatedAt: now } },
    { upsert: true },
  )
  await db.collection<any>(collections.users).updateOne(
    { _id: userId },
    {
      $setOnInsert: { _id: userId, createdAt: now },
      $set: {
        tenantId, name: nurse.name || '', email: (nurse.email || '').trim().toLowerCase(),
        passwordHash: nurse.passwordHash || '', phoneNumber: nurse.phoneNumber || '',
        roles: ['caregiver', 'tenant_admin'], scopes: [], status: 'active',
        notificationPreferences: {
          pushNotificationsEnabled: nurse.pushNotificationsEnabled ?? true,
          alertSensitivity: nurse.alertSensitivity || 'only_important_changes',
          preferredDailySummaryTime: nurse.preferredDailySummaryTime || '19:00',
        },
        updatedAt: now,
      },
    },
    { upsert: true },
  )
  return { tenantId, userId }
}

/** Idempotently upsert the v1 patient + care_relationship for a legacy embedded patient. */
export async function ensureV1Patient(db: Db, tenantId: string, userId: string, patient: LegacyPatient): Promise<string> {
  const patientId = patient._id.toHexString()
  const now = new Date()
  await db.collection<any>(collections.patients).updateOne(
    { _id: patientId },
    {
      $setOnInsert: { _id: patientId, version: 1, createdAt: now },
      $set: {
        tenantId, displayName: patient.name || '', preferredLanguage: patient.preferredLanguage || 'english',
        timezone: patient.timezone || 'Asia/Singapore', ageBand: ageBand(patient.age), status: 'active', updatedAt: now,
      },
    },
    { upsert: true },
  )
  const relId = `rel_${patientId}`
  await db.collection<any>(collections.careRelationships).updateOne(
    { _id: relId },
    {
      $setOnInsert: { _id: relId, createdAt: now, validFrom: now, validTo: null },
      $set: {
        tenantId, patientId, userId, relationshipType: patient.relationshipToElderly || 'caregiver',
        scopes: RELATIONSHIP_SCOPES, status: 'active',
      },
    },
    { upsert: true },
  )
  return patientId
}

/** Combined: ensure tenant+user+patient+relationship. Returns the v1 ids (all = legacy hex). */
export async function ensureV1Identity(db: Db, nurse: LegacyNurse, patient: LegacyPatient) {
  const { tenantId, userId } = await ensureV1TenantUser(db, nurse)
  const patientId = await ensureV1Patient(db, tenantId, userId, patient)
  return { tenantId, userId, patientId }
}

/** Claim a v1 device_pairing by its 6-digit code — mirrors POST /api/v1/device-pairing-claims,
 *  authorized by legacy nurse ownership. Writes the assignment + one-time exchange ticket so the
 *  mirror can redeem device credentials. Throws BridgeError on invalid/expired/already-claimed. */
export async function claimV1Pairing(opts: {
  pairingCode: string; tenantId: string; userId: string; patientId: string; patientDisplayName: string
  mirrorName?: string; correlationId?: string
}): Promise<{ deviceId: string; assignmentId: string; mirrorName: string; pairedAt: Date }> {
  const { pairingCode, tenantId, userId, patientId, patientDisplayName } = opts
  const mirrorName = opts.mirrorName?.trim() || 'Reflexion Mirror'
  const db = await getDb()
  const pairing = await db.collection<any>(collections.pairings).findOne({
    codeHash: hmac(pairingCode), state: 'pending', expiresAt: { $gt: new Date() }, failedAttempts: { $lt: 5 },
  })
  if (!pairing) throw new BridgeError(400, 'Pairing code is not valid or has expired.')

  const assignmentId = newId('asg')
  const exchangeTicket = randomSecret()
  const exchangeTicketExpiresAt = new Date(Date.now() + EXCHANGE_TTL_MS)
  const now = new Date()
  await inTransaction(async (tdb, session) => {
    await tdb.collection<any>(collections.assignments).updateMany(
      { tenantId, status: 'active', $or: [{ deviceId: pairing.deviceId }, { patientId, assignmentType: 'primary' }] },
      { $set: { status: 'replaced', revokedAt: now, revokedBy: userId } }, { session },
    )
    await tdb.collection<any>(collections.assignments).insertOne({
      _id: assignmentId, tenantId, deviceId: pairing.deviceId, patientId, assignmentType: 'primary',
      mirrorName, status: 'active', assignedAt: now, assignedBy: userId, version: 1,
    }, { session })
    const claimed = await tdb.collection<any>(collections.pairings).updateOne(
      { _id: pairing._id, state: 'pending', expiresAt: { $gt: now } },
      { $set: {
        state: 'paired', tenantId, claimedBy: userId, claimedPatientId: patientId, patientDisplayName, pairedAt: now,
        exchangeTicketHash: hashSecret(exchangeTicket), exchangeTicketDigest: sha256(exchangeTicket),
        exchangeTicketCipher: sealSecret(exchangeTicket), exchangeTicketExpiresAt, exchangeConsumedAt: null,
      } }, { session },
    )
    if (!claimed.modifiedCount) throw new BridgeError(409, 'This pairing session was already claimed.')
    await tdb.collection<any>(collections.devices).updateOne(
      { _id: pairing.deviceId }, { $set: { tenantId, status: 'active', displayName: mirrorName, updatedAt: now } }, { session },
    )
    await appendOutbox(tdb, {
      eventType: 'device.paired', tenantId, patientId, aggregateType: 'device', aggregateId: String(pairing.deviceId),
      correlationId: opts.correlationId, payload: { assignmentId },
    } as any, session)
  })
  return { deviceId: String(pairing.deviceId), assignmentId, mirrorName, pairedAt: now }
}
