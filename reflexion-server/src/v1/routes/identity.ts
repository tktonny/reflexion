import { Router } from 'express'
import { hashPassword, verifyPassword } from '../../lib/password.js'
import { getDb, inTransaction } from '../../lib/mongo.js'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { collections } from '../platform/collections.js'
import { hashSecret, sealSecret, sha256, verifySecret } from '../platform/crypto.js'
import { ApiError, badRequest, unauthorized } from '../platform/errors.js'
import { requireActor, getPrincipal } from '../platform/auth.js'
import { sendData } from '../platform/http.js'
import { newId, randomPairingCode, randomSecret } from '../platform/ids.js'
import { issueAccessToken } from '../platform/tokens.js'
import { enumValue, objectBody, optionalString, requiredString } from '../platform/validation.js'
import { appendOutbox } from '../platform/outbox.js'
import { SETUP_PROGRESS_CATEGORIES } from './setupProgress.js'
import { authPolicy, emailVerificationRequired } from '../platform/authPolicy.js'
import { emailDeliveryConfigured } from '../notifications/email.js'

const ACCESS_TTL_SECONDS = 15 * 60
/** Bounded so a pathological duplicate set cannot turn one sign-in into an unbounded scan. */
const MAX_LOGIN_CANDIDATES = 10

const REFRESH_TTL_MS = 30 * 24 * 60 * 60 * 1000
const CODE_RESEND_COOLDOWN_MS = 60 * 1000
const CODE_MAX_ATTEMPTS = 5

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
  // v4 exposes an Email/Phone selector. Keep accepting the legacy `email` field for older builds;
  // a +country-code identifier is resolved against the verified phone number instead.
  const identifier = requiredString(body, 'identifier' in body ? 'identifier' : 'email', 320).trim()
  const phone = identifier.startsWith('+') ? normalizePhone(identifier) : null
  const email = phone ? null : identifier.toLowerCase()
  const password = requiredString(body, 'password', 500)
  const db = await getDb()
  const candidates = await db.collection<any>(collections.users)
    .find(phone ? { phoneNumber: phone, status: { $nin: ['archived', 'deleted'] } } : { emailNormalized: email, status: { $nin: ['archived', 'deleted'] } })
    .sort({ updatedAt: -1, _id: 1 })
    .limit(MAX_LOGIN_CANDIDATES)
    .toArray()

  const verificationRequired = emailVerificationRequired()
  const activeCandidates = candidates.filter((candidate) => candidate?.status === 'active')
  const eligibleCandidates = verificationRequired ? activeCandidates : candidates.filter((candidate) => ['active', 'pending_verification'].includes(String(candidate?.status)))
  const user = eligibleCandidates.find((candidate) => candidate?.passwordHash && verifyPassword(password, String(candidate.passwordHash)))
  if (!user) {
    if (verificationRequired && candidates.some((candidate) => candidate?.status === 'pending_verification')) {
      throw new ApiError(403, 'EMAIL_NOT_VERIFIED', 'Your email has not been verified. Resend the verification email before signing in.')
    }
    if (!candidates.length) {
      // Keep the HTTP posture compatible with existing clients while exposing a stable code so the
      // caregiver can give a useful next step without displaying raw server text.
      throw new ApiError(401, 'ACCOUNT_NOT_FOUND', 'We could not find an account with those details.')
    }
    throw new ApiError(401, 'INVALID_CREDENTIALS', 'Email or password is incorrect.')
  }
  if (activeCandidates.length > 1) {
    // Not fatal — the right account was found — but duplicate users for one email are a data defect that
    // will keep producing confusing behaviour until they are merged.
    console.warn(`[auth] ${activeCandidates.length} active users share ${phone || email}; signed in as ${user._id} (tenant ${user.tenantId})`)
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
 * ['caregiver'] and emailNormalized set — without which they could never sign in again.
 *
 * `tenant_admin` is NOT granted. It reads as "operator of this tenant", not "owner of my own family's
 * data": it makes authorizePatient skip the care_relationships check, unfilters GET /patients, and opens
 * the clinical review queue. See the long note in lib/legacyV1Bridge.ts. A caregiver's access comes from
 * their care_relationships rows and nothing else.
 *
 * New accounts remain pending until the six-digit email verification code is used. This keeps setup and all caregiver
 * data behind a verified identity while leaving migrated/legacy active users untouched.
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
  const verificationRequired = emailVerificationRequired()
  // Checked before writing AND enforced by the write below, because two simultaneous sign-ups for the same
  // email would both pass this check.
  const existing = await db.collection<any>(collections.users).findOne({ emailNormalized: email, status: { $nin: ['archived', 'deleted'] } })
  if (existing) {
    throw new ApiError(409, existing.status === 'pending_verification' ? 'EMAIL_VERIFICATION_PENDING' : 'EMAIL_IN_USE', existing.status === 'pending_verification' ? 'A verification email is already pending for that email address.' : 'An account already exists for that email address.')
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
    appLanguage: 'en',
    roles: ['caregiver'],
    scopes: [],
    status: verificationRequired ? 'pending_verification' : 'active',
    emailVerifiedAt: null,
    notificationPreferences: {
      pushNotificationsEnabled: true,
      alertSensitivity: 'only_important_changes',
      preferredDailySummaryTime: '19:00',
      summaryFrequency: 'daily-summary',
      triggers: defaultNotificationTriggers(),
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
      if (verificationRequired) {
        const verificationCode = randomPairingCode()
        const verificationTokenId = newId('auth')
        await transactionDb.collection<any>(collections.emailVerificationTokens).insertOne({
          _id: verificationTokenId, tenantId, userId, verificationCodeDigest: sha256(verificationCode), verificationCodeHash: hashSecret(verificationCode), codeAttempts: 0,
          state: 'active', createdAt: now, expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000),
        }, { session })
        await appendOutbox(transactionDb, { eventType: 'account_verification.requested', tenantId, aggregateType: 'user', aggregateId: userId, correlationId: request.requestId,
          payload: { verificationTokenId, sealedCode: sealSecret(verificationCode), email, name } }, session)
      }
      await transactionDb.collection<any>(collections.setupProgress).insertOne({
        _id: `setp_${userId}`, tenantId, userId,
        categories: Object.fromEntries(SETUP_PROGRESS_CATEGORIES.map((category) => [category, 'not-started'])),
        version: 1, createdAt: now, updatedAt: now,
      }, { session })
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

  if (!verificationRequired) {
    const issued = await createHumanSession(user as HumanUser)
    sendData(response, { ...issued, state: 'authenticated', email, emailVerified: false }, 201)
    return
  }
  sendData(response, { state: 'verification_pending', email }, 202)
}))

identityRouter.get('/auth/policy', (_request, response) => {
  sendData(response, authPolicy())
})

identityRouter.post('/auth/account-verification-requests', asyncHandler(async (request, response) => {
  const body = objectBody(request.body)
  const email = requiredString(body, 'email', 320).trim().toLowerCase()
  const db = await getDb()
  const user = await db.collection<any>(collections.users).findOne({ emailNormalized: email, status: 'pending_verification' })
  if (user) {
    const now = new Date()
    const previous = await db.collection<any>(collections.emailVerificationTokens).findOne({ userId: user._id, tenantId: user.tenantId, state: 'active' }, { sort: { createdAt: -1 } })
    if (previous && now.getTime() - new Date(previous.createdAt).getTime() < CODE_RESEND_COOLDOWN_MS) throw new ApiError(429, 'VERIFICATION_RESEND_TOO_SOON', 'Please wait before requesting another verification code.', true)
    await db.collection<any>(collections.emailVerificationTokens).updateMany({ userId: user._id, tenantId: user.tenantId, state: 'active' }, { $set: { state: 'superseded', supersededAt: now } })
    const code = randomPairingCode()
    const tokenId = newId('auth')
    await db.collection<any>(collections.emailVerificationTokens).insertOne({
      _id: tokenId, tenantId: user.tenantId, userId: user._id, verificationCodeDigest: sha256(code), verificationCodeHash: hashSecret(code), codeAttempts: 0,
      state: 'active', createdAt: now, expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000),
    })
    await appendOutbox(db, { eventType: 'account_verification.requested', tenantId: String(user.tenantId), aggregateType: 'user', aggregateId: String(user._id), correlationId: request.requestId,
      payload: { verificationTokenId: tokenId, sealedCode: sealSecret(code), email: user.email || email, name: user.name || '' } })
  }
  // Always return the same response so a caller cannot discover whether an address has an account.
  sendData(response, { state: 'accepted' }, 202)
}))

identityRouter.post('/auth/account-verifications', asyncHandler(async (request, response) => {
  const body = objectBody(request.body)
  const email = requiredString(body, 'email', 320).trim().toLowerCase()
  const code = requiredString(body, 'code', 6)
  if (!/^\d{6}$/.test(code)) throw badRequest('CODE_INVALID', 'Enter the six-digit verification code.')
  const db = await getDb()
  const userRow = await db.collection<any>(collections.users).findOne({ emailNormalized: email, status: 'pending_verification' })
  const row = userRow && await db.collection<any>(collections.emailVerificationTokens).findOne({ userId: userRow._id, tenantId: userRow.tenantId, state: 'active', expiresAt: { $gt: new Date() } }, { sort: { createdAt: -1 } })
  if (!row?.verificationCodeHash || Number(row.codeAttempts || 0) >= CODE_MAX_ATTEMPTS || !verifySecret(code, String(row.verificationCodeHash))) {
    if (row && Number(row.codeAttempts || 0) < CODE_MAX_ATTEMPTS) await db.collection<any>(collections.emailVerificationTokens).updateOne({ _id: row._id, state: 'active' }, { $inc: { codeAttempts: 1 } })
    throw new ApiError(400, 'ACCOUNT_VERIFICATION_INVALID', 'This verification code is invalid or expired.')
  }
  let user: HumanUser | null = null
  await inTransaction(async (transactionDb, session) => {
    const used = await transactionDb.collection<any>(collections.emailVerificationTokens).updateOne({ _id: row._id, state: 'active', verificationCodeDigest: sha256(code) }, { $set: { state: 'used', usedAt: new Date() } }, { session })
    if (!used.modifiedCount) throw new ApiError(400, 'ACCOUNT_VERIFICATION_INVALID', 'This verification code is invalid or already used.')
    const updated = await transactionDb.collection<any>(collections.users).findOneAndUpdate({ _id: row.userId, tenantId: row.tenantId, status: 'pending_verification' }, { $set: { status: 'active', emailVerifiedAt: new Date(), updatedAt: new Date() } }, { returnDocument: 'after', session })
    if (!updated) throw new ApiError(400, 'ACCOUNT_VERIFICATION_INVALID', 'This account is no longer awaiting verification.')
    user = updated as HumanUser
    await transactionDb.collection<any>(collections.auditEvents).insertOne({
      _id: newId('audit'), tenantId: row.tenantId, actor: { type: 'user', id: row.userId }, action: 'caregiver.verified', object: { type: 'user', id: row.userId }, outcome: 'success', correlationId: request.requestId, occurredAt: new Date(),
    }, { session })
  })
  if (!user) throw new ApiError(500, 'ACCOUNT_VERIFICATION_FAILED', 'The account could not be verified.', true)
  const issued = await createHumanSession(user)
  sendData(response, { ...issued, actor: serializeActor(user), state: 'verified' }, 201)
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
  if (!emailDeliveryConfigured()) throw new ApiError(503, 'EMAIL_NOT_CONFIGURED', 'Password reset email is unavailable during this pilot. Contact support for help resetting your password.', true)
  const body = objectBody(request.body)
  const email = requiredString(body, 'email', 320).toLowerCase()
  const db = await getDb()
  const user = await db.collection<any>(collections.users).findOne({ emailNormalized: email, status: 'active' })
  if (user) {
    const previous = await db.collection<any>(collections.passwordResetTokens).findOne({ userId: user._id, tenantId: user.tenantId, state: 'active' }, { sort: { createdAt: -1 } })
    if (previous && Date.now() - new Date(previous.createdAt).getTime() < CODE_RESEND_COOLDOWN_MS) throw new ApiError(429, 'VERIFICATION_RESEND_TOO_SOON', 'Please wait before requesting another verification code.', true)
    const code = randomPairingCode()
    const tokenId = newId('auth')
    const expiresAt = new Date(Date.now() + 30 * 60 * 1000)
    await db.collection<any>(collections.passwordResetTokens).updateMany({ userId: user._id, tenantId: user.tenantId, state: 'active' }, { $set: { state: 'superseded', supersededAt: new Date() } })
    await db.collection<any>(collections.passwordResetTokens).insertOne({
      _id: tokenId, tenantId: user.tenantId, userId: user._id,
      verificationCodeDigest: sha256(code), verificationCodeHash: hashSecret(code), codeAttempts: 0,
      state: 'active', createdAt: new Date(), expiresAt,
    })
    await appendOutbox(db, { eventType: 'password_reset.requested', tenantId: String(user.tenantId), aggregateType: 'user',
      aggregateId: String(user._id), correlationId: request.requestId,
      payload: { resetTokenId: tokenId, sealedCode: sealSecret(code), email: user.email || email, name: user.name || '' } })
  }
  sendData(response, { state: 'accepted' }, 202)
}))

identityRouter.post('/auth/password-reset-verifications', asyncHandler(async (request, response) => {
  const body = objectBody(request.body)
  const email = requiredString(body, 'email', 320).trim().toLowerCase()
  const code = requiredString(body, 'code', 6)
  if (!/^\d{6}$/.test(code)) throw badRequest('CODE_INVALID', 'code must contain six digits.')
  const db = await getDb()
  const user = await db.collection<any>(collections.users).findOne({ emailNormalized: email, status: 'active' }, { projection: { _id: 1, tenantId: 1 } })
  const reset = user && await db.collection<any>(collections.passwordResetTokens).findOne({ userId: user._id, tenantId: user.tenantId, state: 'active', expiresAt: { $gt: new Date() } }, { sort: { createdAt: -1 } })
  if (!reset || reset.codeVerifiedAt || Number(reset.codeAttempts || 0) >= CODE_MAX_ATTEMPTS || !reset.verificationCodeHash || !verifySecret(code, String(reset.verificationCodeHash))) {
    if (reset && !reset.codeVerifiedAt && Number(reset.codeAttempts || 0) < CODE_MAX_ATTEMPTS) {
      await db.collection<any>(collections.passwordResetTokens).updateOne({ _id: reset._id, state: 'active' }, { $inc: { codeAttempts: 1 } })
    }
    throw new ApiError(400, 'PASSWORD_RESET_CODE_INVALID', 'The verification code is invalid or expired.')
  }
  await db.collection<any>(collections.passwordResetTokens).updateOne({ _id: reset._id, state: 'active', codeVerifiedAt: { $exists: false } }, { $set: { codeVerifiedAt: new Date() } })
  // The raw token is returned only after the correct code, and is used once by /auth/password-resets.
  // It is not recoverable from the stored hash, so retain it in the request flow by minting a fresh one.
  const freshToken = randomSecret()
  await db.collection<any>(collections.passwordResetTokens).updateOne({ _id: reset._id, state: 'active' }, { $set: { tokenDigest: sha256(freshToken), tokenHash: hashSecret(freshToken) } })
  sendData(response, { resetToken: freshToken }, 200)
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
  if ('appLanguage' in body) update.appLanguage = enumValue(body.appLanguage, 'appLanguage', ['en', 'zh'] as const)
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
    if ('summaryFrequency' in preferences) {
      update['notificationPreferences.summaryFrequency'] = enumValue(preferences.summaryFrequency, 'summaryFrequency', SUMMARY_FREQUENCIES)
    }
    if ('triggers' in preferences) {
      const triggers = objectBody(preferences.triggers)
      for (const trigger of NOTIFICATION_TRIGGERS) {
        if (trigger in triggers && typeof triggers[trigger] !== 'boolean') throw badRequest('VALIDATION_FAILED', `notificationPreferences.triggers.${trigger} must be a boolean.`)
        if (trigger in triggers) update[`notificationPreferences.triggers.${trigger}`] = triggers[trigger]
      }
    }
  }

  // A privacy choice, not a notification one: it governs whether a session's summary text is kept at all.
  // The legacy settings route already offered this switch, and dropping it on the way to v1 would have
  // quietly removed a control a caregiver had used.
  if ('storeSessionSummaries' in body) {
    if (typeof body.storeSessionSummaries !== 'boolean') {
      throw badRequest('VALIDATION_FAILED', 'storeSessionSummaries must be a boolean.')
    }
    update.storeSessionSummaries = body.storeSessionSummaries
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

// Login identity changes are separate from profile edits. A new address is not trusted until the person who
// controls it follows the one-time link dispatched by the outbox worker.
identityRouter.post('/me/email-change-requests', requireActor('human'), asyncHandler(async (request, response) => {
  if (!emailDeliveryConfigured()) throw new ApiError(503, 'EMAIL_NOT_CONFIGURED', 'Email address changes are unavailable until email delivery is configured.', true)
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw unauthorized()
  const email = requiredString(objectBody(request.body), 'email', 320).trim().toLowerCase()
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) throw badRequest('EMAIL_INVALID', 'Enter a valid email address.')
  const db = await getDb()
  const existing = await db.collection<any>(collections.users).findOne({ emailNormalized: email, status: 'active', _id: { $ne: principal.userId } })
  if (existing) throw new ApiError(409, 'EMAIL_IN_USE', 'An account already exists for that email address.')
  const code = randomPairingCode(); const now = new Date(); const expiresAt = new Date(Date.now() + 30 * 60 * 1000)
  const previous = await db.collection<any>(collections.emailChangeTokens).findOne({ userId: principal.userId, tenantId: principal.tenantId, state: 'active' }, { sort: { createdAt: -1 } })
  if (previous && now.getTime() - new Date(previous.createdAt).getTime() < CODE_RESEND_COOLDOWN_MS) throw new ApiError(429, 'VERIFICATION_RESEND_TOO_SOON', 'Please wait before requesting another verification code.', true)
  await db.collection<any>(collections.emailChangeTokens).updateMany({ userId: principal.userId, state: 'active' }, { $set: { state: 'superseded', supersededAt: now } })
  await db.collection<any>(collections.emailChangeTokens).insertOne({ _id: newId('auth'), tenantId: principal.tenantId, userId: principal.userId, email, verificationCodeDigest: sha256(code), verificationCodeHash: hashSecret(code), codeAttempts: 0, state: 'active', createdAt: now, expiresAt })
  const user = await db.collection<any>(collections.users).findOne({ _id: principal.userId, tenantId: principal.tenantId })
  await appendOutbox(db, { eventType: 'email_change.requested', tenantId: principal.tenantId, aggregateType: 'user', aggregateId: principal.userId, correlationId: request.requestId, payload: { email, name: user?.name || '', sealedCode: sealSecret(code) } })
  sendData(response, { state: 'accepted' }, 202)
}))

identityRouter.post('/me/email-changes', requireActor('human'), asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw unauthorized()
  const code = requiredString(objectBody(request.body), 'code', 6)
  if (!/^\d{6}$/.test(code)) throw badRequest('CODE_INVALID', 'Enter the six-digit verification code.')
  const db = await getDb(); const row = await db.collection<any>(collections.emailChangeTokens).findOne({ userId: principal.userId, tenantId: principal.tenantId, state: 'active', expiresAt: { $gt: new Date() } }, { sort: { createdAt: -1 } })
  if (!row?.verificationCodeHash || Number(row.codeAttempts || 0) >= CODE_MAX_ATTEMPTS || !verifySecret(code, String(row.verificationCodeHash))) {
    if (row && Number(row.codeAttempts || 0) < CODE_MAX_ATTEMPTS) await db.collection<any>(collections.emailChangeTokens).updateOne({ _id: row._id, state: 'active' }, { $inc: { codeAttempts: 1 } })
    throw new ApiError(400, 'EMAIL_CHANGE_INVALID', 'This email verification code is invalid or expired.')
  }
  const duplicate = await db.collection<any>(collections.users).findOne({ emailNormalized: row.email, status: 'active', _id: { $ne: principal.userId } })
  if (duplicate) throw new ApiError(409, 'EMAIL_IN_USE', 'An account already exists for that email address.')
  const now = new Date()
  const updated = await db.collection<any>(collections.users).findOneAndUpdate({ _id: principal.userId, tenantId: principal.tenantId, status: 'active' }, { $set: { email: row.email, emailNormalized: row.email, updatedAt: now } }, { returnDocument: 'after' })
  await db.collection<any>(collections.emailChangeTokens).updateOne({ _id: row._id, state: 'active', verificationCodeDigest: sha256(code) }, { $set: { state: 'used', usedAt: now } })
  if (!updated) throw unauthorized()
  sendData(response, serializeProfile(updated as HumanUser))
}))

identityRouter.post('/me/phone-change-requests', requireActor('human'), asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw unauthorized()
  const phoneNumber = normalizePhone(requiredString(objectBody(request.body), 'phoneNumber', 40))
  const db = await getDb()
  const now = new Date()
  await db.collection<any>(collections.phoneChangeTokens).updateMany({ userId: principal.userId, tenantId: principal.tenantId, state: 'active' }, { $set: { state: 'superseded', supersededAt: now } })
  const code = randomPairingCode()
  const token = randomSecret()
  const tokenId = newId('auth')
  await db.collection<any>(collections.phoneChangeTokens).insertOne({ _id: tokenId, tenantId: principal.tenantId, userId: principal.userId, phoneNumber, tokenDigest: sha256(token), tokenHash: hashSecret(token), verificationCodeDigest: sha256(code), verificationCodeHash: hashSecret(code), codeAttempts: 0, state: 'active', createdAt: now, expiresAt: new Date(Date.now() + 30 * 60 * 1000) })
  const user = await db.collection<any>(collections.users).findOne({ _id: principal.userId, tenantId: principal.tenantId })
  await appendOutbox(db, { eventType: 'phone_change.requested', tenantId: principal.tenantId, aggregateType: 'user', aggregateId: principal.userId, correlationId: request.requestId, payload: { phoneNumber, sealedCode: sealSecret(code), name: user?.name || '' } })
  sendData(response, { state: 'accepted', phoneNumber }, 202)
}))

identityRouter.post('/me/phone-changes', requireActor('human'), asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw unauthorized()
  const body = objectBody(request.body)
  const phoneNumber = normalizePhone(requiredString(body, 'phoneNumber', 40))
  const code = requiredString(body, 'code', 6)
  if (!/^\d{6}$/.test(code)) throw badRequest('CODE_INVALID', 'code must contain six digits.')
  const db = await getDb()
  const row = await db.collection<any>(collections.phoneChangeTokens).findOne({ userId: principal.userId, tenantId: principal.tenantId, phoneNumber, state: 'active', expiresAt: { $gt: new Date() } }, { sort: { createdAt: -1 } })
  if (!row || Number(row.codeAttempts || 0) >= 5 || !row.verificationCodeHash || !verifySecret(code, String(row.verificationCodeHash))) {
    if (row && Number(row.codeAttempts || 0) < 5) await db.collection<any>(collections.phoneChangeTokens).updateOne({ _id: row._id, state: 'active' }, { $inc: { codeAttempts: 1 } })
    throw new ApiError(400, 'PHONE_CHANGE_CODE_INVALID', 'The phone verification code is invalid or expired.')
  }
  const now = new Date()
  const updated = await db.collection<any>(collections.users).findOneAndUpdate({ _id: principal.userId, tenantId: principal.tenantId, status: 'active' }, { $set: { phoneNumber, updatedAt: now } }, { returnDocument: 'after' })
  await db.collection<any>(collections.phoneChangeTokens).updateOne({ _id: row._id, state: 'active' }, { $set: { state: 'used', usedAt: now } })
  if (!updated) throw unauthorized()
  sendData(response, serializeProfile(updated as HumanUser))
}))

identityRouter.post('/me/password-changes', requireActor('human'), asyncHandler(async (request, response) => {
  const principal = getPrincipal(request)
  if (principal.kind !== 'human') throw unauthorized()
  const body = objectBody(request.body); const currentPassword = requiredString(body, 'currentPassword', 500); const newPassword = requiredString(body, 'newPassword', 500)
  if (newPassword.length < MIN_PASSWORD_LENGTH) throw badRequest('PASSWORD_TOO_SHORT', `Choose a password of at least ${MIN_PASSWORD_LENGTH} characters.`)
  const db = await getDb(); const user = await db.collection<any>(collections.users).findOne({ _id: principal.userId, tenantId: principal.tenantId, status: 'active' })
  if (!user?.passwordHash || !verifyPassword(currentPassword, String(user.passwordHash))) throw new ApiError(401, 'CURRENT_PASSWORD_INVALID', 'Your current password is incorrect.')
  await db.collection<any>(collections.users).updateOne({ _id: principal.userId, tenantId: principal.tenantId }, { $set: { passwordHash: hashPassword(newPassword), updatedAt: new Date() } })
  sendData(response, { state: 'completed' })
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

function normalizePhone(value: string) {
  const normalized = value.replace(/[\s().-]/g, '')
  if (!/^\+\d{7,15}$/.test(normalized)) throw badRequest('PHONE_INVALID', 'Enter a phone number with country code.')
  return normalized
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
const SUMMARY_FREQUENCIES = ['immediately-after-each-session', 'daily-summary', 'weekly-summary', 'off'] as const
const NOTIFICATION_TRIGGERS = [
  'conversation-session-summary', 'no-interaction-yet-today', 'repeated-missed-interactions',
  'recent-interaction-shorter-than-usual', 'device-may-be-offline', 'reminder-not-completed-or-unclear',
  'new-chat-reply', 'weekly-summary',
] as const

function defaultNotificationTriggers() {
  return Object.fromEntries(NOTIFICATION_TRIGGERS.map((trigger) => [trigger, true]))
}

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
  const storedTriggers = preferences.triggers && typeof preferences.triggers === 'object' && !Array.isArray(preferences.triggers)
    ? preferences.triggers as Record<string, unknown>
    : {}
  return {
    ...serializeActor(user),
    phoneNumber: user.phoneNumber || '',
    relationshipToElderly: user.relationshipToElderly || null,
    appLanguage: user.appLanguage === 'zh' ? 'zh' : 'en',
    notificationPreferences: {
      pushNotificationsEnabled: preferences.pushNotificationsEnabled !== false,
      alertSensitivity: ALERT_SENSITIVITIES.includes(preferences.alertSensitivity as never)
        ? preferences.alertSensitivity
        : 'only_important_changes',
      preferredDailySummaryTime: SUMMARY_TIMES.includes(preferences.preferredDailySummaryTime as never)
        ? preferences.preferredDailySummaryTime
        : '19:00',
      summaryFrequency: SUMMARY_FREQUENCIES.includes(preferences.summaryFrequency as never)
        ? preferences.summaryFrequency
        : 'daily-summary',
      triggers: Object.fromEntries(NOTIFICATION_TRIGGERS.map((trigger) => [trigger, storedTriggers[trigger] !== false])),
    },
    // Defaults to on for accounts that predate the field, the same way the legacy reader treated it
    // (`!== false`). Defaulting a privacy switch the other way would silently stop keeping summaries a
    // caregiver had chosen to keep; defaulting it on preserves what they had and leaves it theirs to change.
    storeSessionSummaries: user.storeSessionSummaries !== false,
  }
}
