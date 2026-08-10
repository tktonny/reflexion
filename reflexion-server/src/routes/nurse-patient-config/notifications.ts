import { Router } from 'express'
import { ObjectId } from 'mongodb'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { DB_NAME, NURSE_CONFIG_COLLECTION } from '../../lib/constants.js'
import { withMongo } from '../../lib/mongo.js'
import { ALERT_SENSITIVITIES, isOneOf, SUMMARY_TIMES } from '../../lib/validation.js'

type UpdateNotificationsBody = {
  nurseId?: string
  pushNotificationsEnabled?: boolean
  alertSensitivity?: string
  preferredDailySummaryTime?: string
}

export const notificationsRouter = Router()

notificationsRouter.patch('/', asyncHandler(async (request, response) => {
  const body = request.body as UpdateNotificationsBody
  const validationError = validateBody(body)
  if (validationError) {
    response.status(400).json({ error: validationError })
    return
  }

  await withMongo(async (client) => {
    const collection = client.db(DB_NAME).collection(NURSE_CONFIG_COLLECTION)
    let filter = body.nurseId ? { _id: new ObjectId(body.nurseId) } : {}

    if (!body.nurseId) {
      const latest = await collection.findOne({}, { sort: { createdAt: -1 } })
      if (!latest?._id) {
        response.status(404).json({ error: 'Nurse config not found' })
        return
      }
      filter = { _id: latest._id }
    }

    const updateResult = await collection.updateOne(filter, {
      $set: {
        pushNotificationsEnabled: body.pushNotificationsEnabled,
        alertSensitivity: body.alertSensitivity,
        preferredDailySummaryTime: body.preferredDailySummaryTime,
        updatedAt: new Date(),
      },
    })

    if (updateResult.matchedCount === 0) {
      response.status(404).json({ error: 'Nurse config not found' })
      return
    }

    const result = await collection.findOne(filter)

    response.json({
      nurseId: result?._id?.toHexString?.() || '',
      pushNotificationsEnabled: result?.pushNotificationsEnabled,
      alertSensitivity: result?.alertSensitivity,
      preferredDailySummaryTime: result?.preferredDailySummaryTime,
    })
  })
}))

function validateBody(body: UpdateNotificationsBody) {
  if (typeof body.pushNotificationsEnabled !== 'boolean') {
    return 'Push notification setting is required.'
  }
  if (!isOneOf(body.alertSensitivity, ALERT_SENSITIVITIES)) {
    return 'Alert sensitivity is invalid.'
  }
  if (!isOneOf(body.preferredDailySummaryTime, SUMMARY_TIMES)) {
    return 'Preferred daily summary time is invalid.'
  }
  if (body.nurseId && !ObjectId.isValid(body.nurseId)) {
    return 'Nurse id is invalid.'
  }

  return ''
}
