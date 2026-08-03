// Reserved forgot-password flow (email delivery is DORMANT until Postmark env is configured).
// Backed on v1 `users` + `password_reset_tokens`; the new password is written as legacy pbkdf2
// (lib/password.hashPassword) so the legacy /auth/sign-in can verify it. See LEGACY_V1_ADAPTER.md.
import { Router } from 'express'
import { ObjectId } from 'mongodb'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { NURSE_CONFIG_COLLECTION } from '../../lib/constants.js'
import { getDb } from '../../lib/mongo.js'
import { hashPassword } from '../../lib/password.js'
import { collections } from '../../v1/platform/collections.js'
import { hashSecret, sha256, verifySecret } from '../../v1/platform/crypto.js'
import { newId, randomPairingCode, randomSecret } from '../../v1/platform/ids.js'
import { sendPasswordResetEmail } from '../../v1/notifications/email.js'

const TOKEN_TTL_MS = 30 * 60 * 1000

// POST /auth/password-reset-requests { email } -> 202 always (no account enumeration).
export const passwordResetRequestRouter = Router()
passwordResetRequestRouter.post('/', asyncHandler(async (request, response) => {
  const email = String((request.body as { email?: string } | null)?.email || '').trim().toLowerCase()
  if (email) {
    const db = await getDb()
    const user = await db.collection<any>(collections.users).findOne({ email, status: 'active' })
    if (user?._id) {
      const code = randomPairingCode()
      const now = new Date()
      await db.collection<any>(collections.passwordResetTokens).insertOne({
        _id: newId('auth'), userId: user._id, tenantId: user.tenantId,
        verificationCodeDigest: sha256(code), verificationCodeHash: hashSecret(code), codeAttempts: 0,
        state: 'active', expiresAt: new Date(now.getTime() + TOKEN_TTL_MS), createdAt: now,
      })
      // Dormant until Postmark is configured: EMAIL_NOT_CONFIGURED (and any delivery error) is swallowed
      // so the request still succeeds; wiring the env at launch makes the reset email flow with no code change.
      try {
        await sendPasswordResetEmail({ email, name: user.name, code })
      } catch (error) {
        console.warn('password_reset_email_skipped:', error instanceof Error ? error.message : error)
      }
    }
  }
  response.status(202).json({ state: 'accepted' })
}))

// Code verification for the legacy mount mirrors the v1 flow so older clients do not receive link emails.
export const passwordResetVerificationRouter = Router()
passwordResetVerificationRouter.post('/', asyncHandler(async (request, response) => {
  const body = request.body as { email?: string; code?: string } | null
  const email = String(body?.email || '').trim().toLowerCase()
  const code = String(body?.code || '')
  const db = await getDb()
  const user = await db.collection<any>(collections.users).findOne({ emailNormalized: email, status: 'active' })
  const record = user && await db.collection<any>(collections.passwordResetTokens).findOne({ userId: user._id, state: 'active', expiresAt: { $gt: new Date() } }, { sort: { createdAt: -1 } })
  if (!/^\d{6}$/.test(code) || !record?.verificationCodeHash || Number(record.codeAttempts || 0) >= 5 || !verifySecret(code, String(record.verificationCodeHash))) {
    if (record && Number(record.codeAttempts || 0) < 5) await db.collection<any>(collections.passwordResetTokens).updateOne({ _id: record._id }, { $inc: { codeAttempts: 1 } })
    response.status(400).json({ error: 'The verification code is invalid or expired.' }); return
  }
  const token = randomSecret()
  await db.collection<any>(collections.passwordResetTokens).updateOne({ _id: record._id, state: 'active' }, { $set: { tokenDigest: sha256(token), tokenHash: hashSecret(token), codeVerifiedAt: new Date() } })
  response.json({ resetToken: token })
}))

// POST /auth/password-resets { token, newPassword } -> set the new password (legacy pbkdf2 hash).
export const passwordResetRouter = Router()
passwordResetRouter.post('/', asyncHandler(async (request, response) => {
  const body = request.body as { token?: string; newPassword?: string } | null
  const token = String(body?.token || '')
  const newPassword = String(body?.newPassword || '')
  if (!token || newPassword.length < 8) {
    response.status(400).json({ error: 'A reset token and a password of at least 8 characters are required.' })
    return
  }
  const db = await getDb()
  const record = await db.collection<any>(collections.passwordResetTokens).findOne({
    tokenDigest: sha256(token), state: 'active', expiresAt: { $gt: new Date() },
  })
  if (!record?.tokenHash || !verifySecret(token, String(record.tokenHash))) {
    response.status(400).json({ error: 'This reset link is invalid or has expired.' })
    return
  }
  const now = new Date()
  const passwordHash = hashPassword(newPassword)
  // BOTH stores. Writing only the v1 user meant legacy sign-in still verified the old NursePatientConfig
  // hash — so the caregiver's new password did not work — and the sign-in bridge then copied that stale hash
  // back over the v1 one, silently undoing the reset in both places.
  await db.collection<any>(collections.users).updateOne(
    { _id: record.userId },
    { $set: { passwordHash, updatedAt: now } },
  )
  if (ObjectId.isValid(String(record.userId))) {
    await db.collection<any>(NURSE_CONFIG_COLLECTION).updateOne(
      { _id: new ObjectId(String(record.userId)) },
      { $set: { passwordHash, updatedAt: now } },
    )
  }
  await db.collection<any>(collections.passwordResetTokens).updateOne(
    { _id: record._id },
    { $set: { state: 'used', usedAt: now } },
  )
  response.json({ state: 'completed' })
}))
