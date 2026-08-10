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
import { CAREGIVER_RELATIONSHIP_SCOPES } from '../v1/platform/scopes.js'
import { appendOutbox } from '../v1/platform/outbox.js'

const EXCHANGE_TTL_MS = 5 * 60 * 1000
const RELATIONSHIP_SCOPES = [...CAREGIVER_RELATIONSHIP_SCOPES]

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
/**
 * The legacy embedded patient. Every field below exists only in NursePatientConfig — the migration is what
 * gives them a v1 home: display-only ones go to `patients.profile`, and the ones that change how Aria talks
 * go to the care plan, which is what the device is already wired to receive.
 */
export type LegacyPatient = {
  _id: HexId; name?: string; preferredLanguage?: string; timezone?: string; age?: number
  relationshipToElderly?: string; mirrorName?: string
  gender?: string; photoUrl?: string; phoneNumber?: string; speechSpeed?: string
  usualWakeTime?: string; speechOrHearingConditions?: string | null
  keyTopics?: string[]; keyTopicsOtherText?: string | null
}

/** Idempotently upsert the v1 tenant + user for a legacy nurse. Returns { tenantId, userId }. */
/**
 * Document version stamped on a consent this code inferred rather than asked for.
 *
 * A real consent record answers "who agreed, when, to what document". A backfilled one cannot — nobody was
 * asked, because no onboarding flow has ever had a consent step. Marking it keeps the two distinguishable
 * forever: `db.consents.find({ documentVersion: 'legacy-onboarding-backfill-v1' })` is exactly the set that
 * was assumed, so it can be audited, re-confirmed or revoked later without guessing.
 */
export const BACKFILL_CONSENT_DOCUMENT_VERSION = 'legacy-onboarding-backfill-v1'

/** The purpose a daily check-in is gated on. Must match v1/routes/patients.ts DAILY_CHECKIN_CONSENT_PURPOSE. */
const CHECKIN_CONSENT_PURPOSE = 'home_cognitive_monitoring'

/**
 * Ensures the patient has the consent a daily check-in requires, creating a marked backfill if not.
 *
 * Why this is needed at all: consent is a HARD gate, not bookkeeping. POST /sessions returns 403
 * CONSENT_REQUIRED for a `daily_checkin` without a granted consent, and the monitoring pipeline excludes
 * any session lacking a consentRef. No patient-creation path has ever written one, so the daily check-in —
 * the product's core function — could never start, and the mirror surfaced only a generic error.
 *
 * Idempotent: an existing granted consent (backfilled or real) is left exactly as it is.
 */
export async function ensureCheckInConsent(
  db: Db,
  input: { tenantId: string; patientId: string; actorId: string },
): Promise<'created' | 'present'> {
  const existing = await db.collection<any>(collections.consents).findOne({
    tenantId: input.tenantId, patientId: input.patientId,
    purpose: CHECKIN_CONSENT_PURPOSE, status: 'granted', withdrawnAt: null,
  })
  if (existing) return 'present'

  const now = new Date()
  await db.collection<any>(collections.consents).insertOne({
    _id: newId('con'), tenantId: input.tenantId, patientId: input.patientId,
    purpose: CHECKIN_CONSENT_PURPOSE, documentVersion: BACKFILL_CONSENT_DOCUMENT_VERSION,
    status: 'granted', signedAt: now, withdrawnAt: null,
    actorId: input.actorId,
    // Queryable provenance, so a backfilled record is never mistaken for one a caregiver actually gave.
    source: 'legacy_onboarding_backfill',
    createdAt: now,
  })
  return 'created'
}

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
      // passwordHash is seeded ONLY on insert. It used to be re-$set on every legacy sign-in, which copied the
      // legacy hash over the v1 one — reverting a completed password reset. Both stores are now written
      // together by the reset route (routes/auth/password-reset.ts), so they stay in step without this.
      // SEED-ONLY for anything PATCH /me can now change. These used to be re-$set on every legacy sign-in,
      // which was fine while legacy was the only writer — but once a caregiver can edit their name, phone or
      // notification preferences over v1, re-seeding from the legacy document silently reverts the edit on
      // their next sign-in. The legacy settings route writes v1 directly instead (see
      // routes/nurse-patient-config/settings.ts), so both surfaces stay in step without this overwriting.
      $setOnInsert: {
        _id: userId, createdAt: now, passwordHash: nurse.passwordHash || '',
        name: nurse.name || '', phoneNumber: nurse.phoneNumber || '',
        notificationPreferences: {
          pushNotificationsEnabled: nurse.pushNotificationsEnabled ?? true,
          alertSensitivity: nurse.alertSensitivity || 'only_important_changes',
          preferredDailySummaryTime: nurse.preferredDailySummaryTime || '19:00',
        },
      },
      $set: {
        // emailNormalized is the field v1 login (identity.ts POST /auth/sessions) matches on — WITHOUT it
        // a bridged caregiver can never obtain a v1 session (401 despite a correct password), which locks
        // every v1-gated screen behind "Sign in again". Keep it identical to `email` (lowercased).
        // Identity and authorization stay legacy-owned: email is the login key and roles are not
        // user-editable, so keeping these authoritative is correct.
        //
        // `caregiver` ONLY — deliberately not `tenant_admin`. A caregiver is the owner of one family's data,
        // not an operator of the tenant, and `tenant_admin` is read as the latter in three places:
        //   - platform/auth.ts authorizePatient() returns the patient immediately, so care_relationships
        //     scopes are never enforced at all;
        //   - v1/routes/patients.ts GET /patients stops filtering by relationship and lists every patient
        //     in the tenant;
        //   - v1/routes/monitoring.ts admits the holder to the clinical review queue (/review-cases), which
        //     a caregiver must never reach, and to POST .../dispositions.
        // Granting it here made all three true for every caregiver. It went unnoticed because the bridge
        // derives tenantId from the nurse's own _id, so one tenant holds exactly one caregiver and "every
        // patient in the tenant" happened to equal "my own family" — it would become a real cross-family
        // read the moment two caregivers share a tenant. Access now comes from care_relationships alone.
        tenantId, email: (nurse.email || '').trim().toLowerCase(),
        emailNormalized: (nurse.email || '').trim().toLowerCase(),
        roles: ['caregiver'], scopes: [], status: 'active',
        updatedAt: now,
      },
    },
    { upsert: true },
  )
  // Collapse duplicates so one email means one account.
  //
  // The hex-keyed row is canonical: its _id IS the legacy nurse's ObjectId, and every existing patient and
  // care_relationship for this caregiver is keyed to its tenant. An older `usr_`-keyed row for the same email
  // (created by an earlier code path, in its own tenant) is an interloper — but because the users unique index
  // is per tenant, nothing stopped it existing, and every tenant-less email lookup in the codebase could
  // return either one: v1 sign-in 401'd with the correct password, and a password-reset token could be minted
  // against the wrong row entirely.
  //
  // Archived rather than deleted: this is a data repair inferred from shape, so it stays reversible.
  const emailNormalized = (nurse.email || '').trim().toLowerCase()
  if (emailNormalized) {
    // NEVER archive an operator or provider. The bootstrap script creates the console account with
    // roles ['tenant_admin','provider','caregiver'] in its own tenant, and on a small team that is very
    // likely the same email as a caregiver account — so an unguarded sweep would archive the operator the
    // first time that person signed into the app, locking them out of admin-web entirely.
    // Only a caregiver-shaped row is collapsible.
    const collapsed = await db.collection<any>(collections.users).updateMany(
      {
        emailNormalized, status: 'active', _id: { $ne: userId },
        roles: { $nin: ['operator', 'provider'] },
      },
      { $set: { status: 'archived', archivedAt: now, archivedReason: 'duplicate_email_superseded_by_legacy_identity' } },
    )
    if (collapsed.modifiedCount) {
      console.warn(`[bridge] archived ${collapsed.modifiedCount} duplicate v1 user(s) for ${emailNormalized}; canonical is ${userId}`)
    }

    // A privileged duplicate is left alone but reported: sign-in resolves it correctly by verifying the
    // password, so nothing is broken, but two accounts on one email stays a data question for a human.
    const privileged = await db.collection<any>(collections.users).countDocuments({
      emailNormalized, status: 'active', _id: { $ne: userId }, roles: { $in: ['operator', 'provider'] },
    })
    if (privileged) {
      console.warn(`[bridge] ${emailNormalized} also has ${privileged} operator/provider account(s); left untouched`)
    }
  }

  return { tenantId, userId }
}

/** Idempotently upsert the v1 patient + care_relationship for a legacy embedded patient. */
export async function ensureV1Patient(db: Db, tenantId: string, userId: string, patient: LegacyPatient): Promise<string> {
  const patientId = patient._id.toHexString()
  const now = new Date()
  await db.collection<any>(collections.patients).updateOne(
    { _id: patientId },
    {
      $setOnInsert: {
        _id: patientId, version: 1, createdAt: now,
        // Seed-only, for the same reason as the caregiver's own fields: PATCH /patients can edit these, and
        // re-asserting them from the legacy document on every sign-in would revert that edit. The legacy
        // patient-settings route writes v1 directly instead.
        profile: {
          age: Number.isFinite(Number(patient.age)) && Number(patient.age) > 0 ? Number(patient.age) : null,
          gender: patient.gender || null,
          photoUrl: patient.photoUrl || null,
          phoneNumber: patient.phoneNumber || null,
          speechSpeed: normalizeSpeechSpeed(patient.speechSpeed),
        },
      },
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
  // Without this the patient exists but every daily check-in is refused with 403 CONSENT_REQUIRED.
  await ensureCheckInConsent(db, { tenantId, patientId, actorId: userId })
  await ensureCarePlanFromLegacyProfile(db, { tenantId, patientId, userId, patient })

  return patientId
}

function normalizeSpeechSpeed(value: unknown): 'slow' | 'normal' | 'fast' | null {
  const speed = String(value || '').trim().toLowerCase()
  return speed === 'slow' || speed === 'normal' || speed === 'fast' ? speed : null
}

/**
 * Moves the fields that shape how Aria TALKS into the care plan, which is what the device actually receives
 * (GET /devices/:deviceId/configuration ships carePlan.dailyRoutine and carePlan.communicationPreferences).
 * Left in the legacy document they were invisible to the mirror, so a caregiver could set a wake time and
 * favourite topics that nothing ever used.
 *
 * Only seeds a plan when none exists — a caregiver or clinician who has since edited the plan owns it.
 */
async function ensureCarePlanFromLegacyProfile(
  db: Db,
  input: { tenantId: string; patientId: string; userId: string; patient: LegacyPatient & Record<string, any> },
): Promise<'created' | 'present' | 'nothing_to_seed'> {
  const existing = await db.collection<any>(collections.carePlans).findOne({
    tenantId: input.tenantId, patientId: input.patientId, status: 'active',
  })
  if (existing) return 'present'

  const source = input.patient
  const topics = Array.isArray(source.keyTopics) ? source.keyTopics.filter(Boolean) : []
  const dailyRoutine: Record<string, unknown> = {}
  const communicationPreferences: Record<string, unknown> = {}
  if (source.usualWakeTime) dailyRoutine.wakeTime = String(source.usualWakeTime)
  if (topics.length) communicationPreferences.topics = topics
  if (source.keyTopicsOtherText) communicationPreferences.otherTopic = String(source.keyTopicsOtherText)
  const speechSpeed = normalizeSpeechSpeed(source.speechSpeed)
  if (speechSpeed) communicationPreferences.speechSpeed = speechSpeed
  if (source.speechOrHearingConditions) {
    communicationPreferences.speechOrHearingNotes = String(source.speechOrHearingConditions)
  }
  if (!Object.keys(dailyRoutine).length && !Object.keys(communicationPreferences).length) {
    return 'nothing_to_seed'
  }

  const now = new Date()
  await db.collection<any>(collections.carePlans).insertOne({
    _id: newId('plan'), tenantId: input.tenantId, patientId: input.patientId, version: 1, status: 'active',
    effectiveFrom: now, effectiveTo: null, ownerId: input.userId,
    dailyRoutine, communicationPreferences, safetyNotes: null,
    source: 'legacy_profile_migration', createdAt: now, updatedAt: now,
  })
  return 'created'
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
