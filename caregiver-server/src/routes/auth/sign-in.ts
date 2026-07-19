import { Router } from 'express'
import { ObjectId } from 'mongodb'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { DB_NAME, NURSE_CONFIG_COLLECTION } from '../../lib/constants.js'
import { withMongo } from '../../lib/mongo.js'
import { verifyPassword } from '../../lib/password.js'

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

    response.json({
      nurseId: user._id?.toHexString() || '',
      name: user.name || '',
      email: user.email || email,
    })
  })
}))
