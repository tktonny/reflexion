// Reserved forgot-password flow (email delivery is DORMANT until Postmark env is configured).
// Backed on v1 `users` + `password_reset_tokens`; the new password is written as legacy pbkdf2
// (lib/password.hashPassword) so the legacy /auth/sign-in can verify it. See LEGACY_V1_ADAPTER.md.
import { Router } from 'express'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { getDb } from '../../lib/mongo.js'
import { hashPassword } from '../../lib/password.js'
import { collections } from '../../v1/platform/collections.js'
import { hashSecret, sha256, verifySecret } from '../../v1/platform/crypto.js'
import { newId, randomSecret } from '../../v1/platform/ids.js'
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
      const token = randomSecret()
      const now = new Date()
      await db.collection<any>(collections.passwordResetTokens).insertOne({
        _id: newId('auth'), userId: user._id, tenantId: user.tenantId,
        tokenDigest: sha256(token), tokenHash: hashSecret(token),
        state: 'active', expiresAt: new Date(now.getTime() + TOKEN_TTL_MS), createdAt: now,
      })
      // Dormant until Postmark is configured: EMAIL_NOT_CONFIGURED (and any delivery error) is swallowed
      // so the request still succeeds; wiring the env at launch makes the reset email flow with no code change.
      try {
        await sendPasswordResetEmail({ email, name: user.name, token })
      } catch (error) {
        console.warn('password_reset_email_skipped:', error instanceof Error ? error.message : error)
      }
    }
  }
  response.status(202).json({ state: 'accepted' })
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
  await db.collection<any>(collections.users).updateOne(
    { _id: record.userId },
    { $set: { passwordHash: hashPassword(newPassword), updatedAt: now } },
  )
  await db.collection<any>(collections.passwordResetTokens).updateOne(
    { _id: record._id },
    { $set: { state: 'used', usedAt: now } },
  )
  response.json({ state: 'completed' })
}))
