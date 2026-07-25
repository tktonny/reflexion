import { Router } from 'express'
import { hashPassword, verifyPassword } from '../../lib/password.js'
import { getDb, inTransaction } from '../../lib/mongo.js'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { collections } from '../platform/collections.js'
import { hashSecret, sealSecret, sha256, verifySecret } from '../platform/crypto.js'
import { ApiError, badRequest, unauthorized } from '../platform/errors.js'
import { requireActor, getPrincipal } from '../platform/auth.js'
import { sendData } from '../platform/http.js'
import { newId, randomSecret } from '../platform/ids.js'
import { issueAccessToken } from '../platform/tokens.js'
import { enumValue, objectBody, optionalString, requiredString } from '../platform/validation.js'
import { appendOutbox } from '../platform/outbox.js'

const ACCESS_TTL_SECONDS = 15 * 60
/** Bounded so a pathological duplicate set cannot turn one sign-in into an unbounded scan. */
const MAX_LOGIN_CANDIDATES = 10

const REFRESH_TTL_MS = 30 * 24 * 60 * 60 * 1000

export const identityRouter = Router()

/**
 * Sign in.
 *
 * Resolves the account by trying EVERY user with this email, not the first one Mongo happens to return.
 * That matters because email is only unique per tenant (`{tenantId, emailNormalized}` in platform/indexes.ts)
 * while this lookup is deliberately tenant-less — the caller does not know their tenant yet — and the legacy
 * bridge gives every caregiver a private tenant. So one email CAN legitimately have several user documents,
 * and a plain findOne picked one arbitrarily: for an account with a stale duplicate that meant a 401 with the
 * correct password, and in the worst case signing someone into the wrong account and therefore another
 * family's tenant.
 *
 * Candidates are ordered newest-first so a repeat login is stable, and the first whose password verifies wins.
 */
identityRouter.post('/auth/sessions', asyncHandler(async (request, response) => {
  const body = objectBody(request.body)
  const email = requiredString(body, 'email', 320).toLowerCase()
  const password = requiredString(body, 'password', 500)
  const db = await getDb()
  const candidates = await db.collection<any>(collections.users)
    .find({ emailNormalized: email, status: 'active' })
    .sort({ updatedAt: -1, _id: 1 })
    .limit(MAX_LOGIN_CANDIDATES)
    .toArray()

  const user = candidates.find((candidate) => candidate?.passwordHash && verifyPassword(password, String(candidate.passwordHash)))
  if (!user) {
    throw new ApiError(401, 'INVALID_CREDENTIALS', 'Email or password is incorrect.')
  }
  if (candidates.length > 1) {
    // Not fatal — the right account was found — but duplicate users for one email are a data defect that
    // will keep producing confusing behaviour until they are merged.
    console.warn(`[auth] ${candidates.length} active users share ${email}; signed in as ${user._id} (tenant ${user.tenantId})`)
  }

  const issued = await createHumanSession(user as HumanUser)
  sendData(response, { ...issued, actor: serializeActor(user as HumanUser) }, 201)
}))

const MIN_PASSWORD_LENGTH = 12
const RELATIONSHIP_TYPES = ['parent', 'sibling', 'spouse', 'inlaw', 'grandpa', 'grandma', 'other'] as const

/**
 * Caregiver self-registration — the one thing v1 could not do.
 *
 * Until now nothing in v1 could create a user over HTTP at all: `users` was written only by the CLI
 * bootstrap script, the legacy sign-in bridge and the migration. Caregiver accounts were therefore created
 * by legacy `POST /nurse-patient-config/create`, which is why the app could not leave the legacy API behind.
 *
 * The shape deliberately matches what legacyV1Bridge.ensureV1TenantUser produces, so a caregiver who signs
 * up here is indistinguishable from one who was migrated: one private tenant per caregiver, roles
 * ['caregiver','tenant_admin'] (every /admin route filters on principal.tenantId, so that grants nothing
 * outside their own tenant), and emailNormalized set — without which they could never sign in again.
 *
 * Returns a session directly, because an account you cannot immediately use is not useful. Rate limiting
 * comes from the /api/v1/auth mount in app.ts.
 */
identityRouter.post('/auth/registrations', asyncHandler(async (request, response) => {
  const body = objectBody(request.body)
  const name = requiredString(body, 'name', 120)
  const email = requiredString(body, 'email', 320).trim().toLowerCase()
  const password = requiredString(body, 'password', 500)
  const phoneNumber = optionalString(body, 'phoneNumber', 40)
  const relationshipToElderly = 'relationshipToElderly' in body
    ? enumValue(body.relationshipToElderly, 'relationshipToElderly', RELATIONSHIP_TYPES)
    : null

  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) {
    throw badRequest('EMAIL_INVALID', 'Enter a valid email address.')
  }
  if (password.length < MIN_PASSWORD_LENGTH) {
    throw badRequest('PASSWORD_TOO_SHORT', `Choose a password of at least ${MIN_PASSWORD_LENGTH} characters.`)
  }

  const db = await getDb()
  // Checked before writing AND enforced by the write below, because two simultaneous sign-ups for the same
  // email would both pass this check.
  const existing = await db.collection<any>(collections.users).findOne({ emailNormalized: email, status: 'active' })
  if (existing) {
    throw new ApiError(409, 'EMAIL_IN_USE', 'An account already exists for that email address.')
  }

  const userId = newId('usr')
  const tenantId = newId('ten')
  const now = new Date()
  const user = {
    _id: userId,
    tenantId,
    name,
    email,
    emailNormalized: email,
    passwordHash: hashPassword(password),
    phoneNumber: phoneNumber || '',
    relationshipToElderly,
    roles: ['caregiver', 'tenant_admin'],
    scopes: [],
    status: 'active',
    notificationPreferences: {
      pushNotificationsEnabled: true,
      alertSensitivity: 'only_important_changes',
      preferredDailySummaryTime: '19:00',
    },
    createdAt: now,
    updatedAt: now,
  }

  try {
    await inTransaction(async (transactionDb, session) => {
      await transactionDb.collection<any>(collections.tenants).insertOne({
        _id: tenantId, name: `${name} tenant`, status: 'active', createdAt: now, updatedAt: now,
      }, { session })
      await transactionDb.collection<any>(collections.users).insertOne(user, { session })
      await transactionDb.collection<any>(collections.auditEvents).insertOne({
        _id: newId('audit'), tenantId, actor: { type: 'user', id: userId },
        action: 'caregiver.registered', object: { type: 'user', id: userId }, outcome: 'success',
        correlationId: request.requestId, occurredAt: now,
      }, { session })
    })
  } catch (error) {
    // The unique index is {tenantId, emailNormalized}; a fresh tenant means it cannot catch a cross-tenant
    // race, so the pre-check above is the real guard and this only reports a genuine write failure.
    if ((error as { code?: number })?.code === 11000) {
      throw new ApiError(409, 'EMAIL_IN_USE', 'An account already exists for that email address.')
    }
    throw error
  }

  const issued = await createHumanSession(user as HumanUser)
  sendData(response, { ...issued, actor: serializeActor(user as HumanUser) }, 201)
}))

identityRouter.post('/auth/session-refreshes', asyncHandler(async (request, response) => {
  const body = objectBody(request.body)
  const refreshToken = requiredString(body, 'refreshToken', 500)
  const db = await getDb()
  const session = await db.collection<any>(collections.authSessions).findOne({
    refreshDigest: sha256(refreshToken), status: 'active', refreshExpiresAt: { $gt: new Date() },
  })
  if (!session?.refreshHash || !verifySecret(refreshToken, String(session.refreshHash))) throw unauthorized('The refresh credential is invalid or expired.')
  const user = await db.collection<any>(collections.users).findOne({
    _id: session.userId, tenantId: session.tenantId, status: 'active',
  }) as HumanUser | null
  if (!user) throw unauthorized('The account is no longer active.')

  const nextRefresh = randomSecret()
  const now = new Date()
  const updated = await db.collection<any>(collections.authSessions).updateOne({
    _id: session._id, status: 'active', refreshDigest: sha256(refreshToken),
  }, { $set: {
    refreshDigest: sha256(nextRefresh), refreshHash: hashSecret(nextRefresh), rotatedAt: now, lastUsedAt: now,
  }, $inc: { version: 1 } })
  if (!updated.modifiedCount) throw unauthorized('The refresh credential has already been rotated.')

  const accessToken = issueHumanAccessToken(user, String(session._id))
  sendData(response, {
    accessToken,
    accessTokenExpiresAt: new Date(Date.now() + ACCESS_TTL_SECONDS * 1000).toISOString(),
    refreshToken: nextRefresh,
    refreshTokenExpiresAt: new Date(session.refreshExpiresAt).toISOString(),
  }, 201)
}))

identityRouter.post('/auth/password-reset-requests', asyncHandler(async (request, response) => {
  const body = objectBody(request.body)
  const email = requiredString(body, 'email', 320).toLowerCase()
  const db = await getDb()
  const user = await db.collection<any>(collections.users).findOne({ emailNormalized: email, status: 'active' })
  if (user) {
    const token = randomSecret()
    const tokenId = newId('auth')
    const expiresAt = new Date(Date.now() + 30 * 60 * 1000)
    await db.collection<any>(collections.passwordResetTokens).insertOne({
      _id: tokenId, tenantId: user.tenantId, userId: user._id, tokenDigest: sha256(token), tokenHash: hashSecret(token),
      state: 'active', createdAt: new Date(), expiresAt,
    })
    await appendOutbox(db, { eventType: 'password_reset.requested', tenantId: String(user.tenantId), aggregateType: 'user',
      aggregateId: String(user._id), correlationId: request.requestId,
      payload: { resetTokenId: tokenId, sealedToken: sealSecret(token), email: user.email || email, name: user.name || '' } })
  }
  sendData(response, { state: 'accepted' }, 202)
}))

identityRouter.post('/auth/password-resets', asyncHandler(async (request, response) => {
  const body = objectBody(request.body)
  const token = requiredString(body, 'token', 500)
  const newPassword = requiredString(body, 'newPassword', 500)
  if (newPassword.length < 12) throw badRequest('PASSWORD_TOO_SHORT', 'newPassword must contain at least 12 characters.')
  const db = await getDb()
  const reset = await db.collection<any>(collections.passwordResetTokens).findOne({
    tokenDigest: sha256(token), state: 'active', expiresAt: { $gt: new Date() },
  })
  if (!reset?.tokenHash || !verifySecret(token, String(reset.tokenHash))) throw new ApiError(400, 'PASSWORD_RESET_INVALID', 'The password reset link is invalid or expired.')
  await inTransaction(async (transactionDb, session) => {
    const used = await transactionDb.collection<any>(collections.passwordResetTokens).updateOne({
      _id: reset._id, state: 'active', tokenDigest: sha256(token),
    }, { $set: { state: 'used', usedAt: new Date() } }, { session })
    if (!used.modifiedCount) throw new ApiError(400, 'PASSWORD_RESET_INVALID', 'The password reset link is invalid or already used.')
    await transactionDb.collection<any>(collections.users).updateOne({ _id: reset.userId, tenantId: reset.tenantId }, { $set: {
      passwordHash: hashPassword(newPassword), updatedAt: new Date(),
    } }, { session })
    await transactionDb.collection<any>(collections.authSessions).updateMany({ userId: reset.userId, status: 'active' }, { $set: { status: 'revoked', revokedAt: new Date(), revocationReason: 'password_reset' } }, { session })
  })
  sendData(response, { state: 'completed' })
}))

identityRouter.delete('/auth/sessions/current', requireActor('human'), asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw unauthorized()
  const db = await getDb()
  await db.collection<any>(collections.authSessions).updateOne({ _id: principal.sessionId }, {
    $set: { status: 'revoked', revokedAt: new Date() },
  })
  response.status(204).end()
}))

identityRouter.get('/me', requireActor('human'), asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw unauthorized()
  const db = await getDb()
  const user = await db.collection<any>(collections.users).findOne({
    _id: principal.userId, tenantId: principal.tenantId, status: 'active',
  }) as HumanUser | null
  if (!user) throw unauthorized()
  sendData(response, serializeProfile(user))
}))

/**
 * Update the caregiver's own profile. PARTIAL: only keys present in the body are written, so a client that
 * has not loaded a field can never blank it — the same rule the legacy settings route follows.
 *
 * Email is deliberately NOT editable here. Changing it would move the account's login identity and, with
 * email uniqueness only enforced per tenant, could silently create the duplicate-account situation that
 * broke sign-in. It needs its own verified flow.
 */
identityRouter.patch('/me', requireActor('human'), asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw unauthorized()
  const body = objectBody(request.body)
  const update: Record<string, unknown> = {}

  if ('name' in body) update.name = requiredString(body, 'name', 120)
  if ('phoneNumber' in body) update.phoneNumber = (optionalString(body, 'phoneNumber', 40) || '')
  if ('relationshipToElderly' in body) {
    update.relationshipToElderly = body.relationshipToElderly === null
      ? null
      : enumValue(body.relationshipToElderly, 'relationshipToElderly', RELATIONSHIP_TYPES)
  }
  if ('notificationPreferences' in body) {
    const preferences = objectBody(body.notificationPreferences)
    if ('pushNotificationsEnabled' in preferences) {
      if (typeof preferences.pushNotificationsEnabled !== 'boolean') {
        throw badRequest('VALIDATION_FAILED', 'pushNotificationsEnabled must be a boolean.')
      }
      update['notificationPreferences.pushNotificationsEnabled'] = preferences.pushNotificationsEnabled
    }
    if ('alertSensitivity' in preferences) {
      update['notificationPreferences.alertSensitivity'] = enumValue(preferences.alertSensitivity, 'alertSensitivity', ALERT_SENSITIVITIES)
    }
    if ('preferredDailySummaryTime' in preferences) {
      update['notificationPreferences.preferredDailySummaryTime'] = enumValue(preferences.preferredDailySummaryTime, 'preferredDailySummaryTime', SUMMARY_TIMES)
    }
  }

  if (!Object.keys(update).length) throw badRequest('VALIDATION_FAILED', 'No supported profile field was provided.')
  update.updatedAt = new Date()

  const db = await getDb()
  const updated = await db.collection<any>(collections.users).findOneAndUpdate(
    { _id: principal.userId, tenantId: principal.tenantId, status: 'active' },
    { $set: update },
    { returnDocument: 'after' },
  )
  if (!updated) throw unauthorized()
  sendData(response, serializeProfile(updated as HumanUser))
}))

type HumanUser = {
  _id: string
  tenantId: string
  name?: string
  email?: string
  emailNormalized?: string
  roles?: string[]
  scopes?: string[]
  passwordHash?: string
}

async function createHumanSession(user: HumanUser) {
  if (!user._id || !user.tenantId) throw badRequest('ACCOUNT_INVALID', 'The account is missing tenant identity.')
  const db = await getDb()
  const sessionId = newId('auth')
  const refreshToken = randomSecret()
  const refreshExpiresAt = new Date(Date.now() + REFRESH_TTL_MS)
  await db.collection<any>(collections.authSessions).insertOne({
    _id: sessionId,
    tenantId: user.tenantId,
    userId: user._id,
    refreshDigest: sha256(refreshToken),
    refreshHash: hashSecret(refreshToken),
    status: 'active',
    version: 1,
    createdAt: new Date(),
    refreshExpiresAt,
  })
  return {
    accessToken: issueHumanAccessToken(user, sessionId),
    accessTokenExpiresAt: new Date(Date.now() + ACCESS_TTL_SECONDS * 1000).toISOString(),
    refreshToken,
    refreshTokenExpiresAt: refreshExpiresAt.toISOString(),
  }
}

function issueHumanAccessToken(user: HumanUser, sessionId: string) {
  return issueAccessToken({
    sub: user._id,
    kind: 'human',
    uid: user._id,
    tid: user.tenantId,
    sid: sessionId,
    roles: user.roles || ['caregiver'],
    scopes: user.scopes || [],
  }, ACCESS_TTL_SECONDS)
}

function serializeActor(user: HumanUser) {
  return {
    userId: user._id,
    tenantId: user.tenantId,
    name: user.name || '',
    email: user.email || user.emailNormalized || '',
    roles: user.roles || [],
  }
}

const ALERT_SENSITIVITIES = ['notify_me_about_everything', 'only_important_changes', 'only_urgent_alerts'] as const
const SUMMARY_TIMES = ['09:00', '19:00'] as const

/**
 * The caregiver's own profile, for their settings screen.
 *
 * `phoneNumber` and `notificationPreferences` were already being written into `users` by the legacy bridge,
 * but no v1 route returned them and none could update them — so the settings screen had no choice but to
 * stay on the legacy API. Defaults are applied here so a user document predating a preference still reads
 * as something the UI can render.
 */
function serializeProfile(user: HumanUser & Record<string, any>) {
  const preferences = (user.notificationPreferences || {}) as Record<string, unknown>
  return {
    ...serializeActor(user),
    phoneNumber: user.phoneNumber || '',
    relationshipToElderly: user.relationshipToElderly || null,
    notificationPreferences: {
      pushNotificationsEnabled: preferences.pushNotificationsEnabled !== false,
      alertSensitivity: ALERT_SENSITIVITIES.includes(preferences.alertSensitivity as never)
        ? preferences.alertSensitivity
        : 'only_important_changes',
      preferredDailySummaryTime: SUMMARY_TIMES.includes(preferences.preferredDailySummaryTime as never)
        ? preferences.preferredDailySummaryTime
        : '19:00',
    },
  }
}
