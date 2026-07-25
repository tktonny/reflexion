import { Router } from 'express'
import { ObjectId } from 'mongodb'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { DB_NAME, NURSE_CONFIG_COLLECTION } from '../../lib/constants.js'
import { getDb, withMongo } from '../../lib/mongo.js'
import { verifyPassword } from '../../lib/password.js'
import { ensureV1TenantUser, type LegacyNurse } from '../../lib/legacyV1Bridge.js'

type SignInBody = {
  email?: string
  password?: string
}

type NurseConfig = {
  _id?: ObjectId
  name?: string
  email?: string
  passwordHash?: string
}

export const signInRouter = Router()

signInRouter.post('/', asyncHandler(async (request, response) => {
  const body = request.body as SignInBody
  const email = body.email?.trim().toLowerCase()
  const password = body.password || ''

  if (!email || !password) {
    response.status(400).json({ error: 'Email and password are required.' })
    return
  }

  await withMongo(async (client) => {
    const user = await client.db(DB_NAME).collection<NurseConfig>(NURSE_CONFIG_COLLECTION).findOne({ email })
    if (!user?.passwordHash || !verifyPassword(password, user.passwordHash)) {
      response.status(401).json({ error: 'Invalid email or password.' })
      return
    }

    // Mirror the caregiver into the v1 users model (with emailNormalized) so the app's best-effort v1
    // login — POST /api/v1/auth/sessions, which matches users.emailNormalized — can obtain a session and
    // unlock the v1 status/alerts/loved-ones screens. Best-effort: a bridge failure never blocks sign-in.
    try {
      await ensureV1TenantUser(await getDb(), user as unknown as LegacyNurse)
    } catch (bridgeError) {
      console.warn('[sign-in] v1 user bridge failed; caregiver v1 features may be delayed', bridgeError)
    }

    response.json({
      nurseId: user._id?.toHexString() || '',
      name: user.name || '',
      email: user.email || email,
    })
  })
}))
