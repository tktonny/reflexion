import { Router } from 'express'
import { ObjectId } from 'mongodb'
import { asyncHandler } from '../lib/asyncHandler.js'
import { DB_NAME, NURSE_CONFIG_COLLECTION } from '../lib/constants.js'
import { withMongo } from '../lib/mongo.js'
import { collections } from '../v1/platform/collections.js'
import { newId } from '../v1/platform/ids.js'

// Free-text caregiver feedback (legacy tokenless surface, keyed by nurseId like the other caregiver
// routes). Ports the capability from the original caregiver-app-server that was never carried into
// reflexion-server. Stored in the `feedback` collection for the operator/admin to review.

type FeedbackBody = { nurseId?: string; message?: string; category?: string }

export const feedbackRouter = Router()

feedbackRouter.post('/', asyncHandler(async (request, response) => {
  const body = request.body as FeedbackBody
  const nurseId = (body.nurseId || '').trim()
  const message = (body.message || '').trim()
  if (!nurseId || !ObjectId.isValid(nurseId)) {
    response.status(400).json({ error: 'Valid nurseId is required.' })
    return
  }
  if (!message) {
    response.status(400).json({ error: 'Message is required.' })
    return
  }
  if (message.length > 5000) {
    response.status(400).json({ error: 'Message is too long (max 5000 characters).' })
    return
  }

  await withMongo(async (client) => {
    const db = client.db(DB_NAME)
    const nurse = await db.collection(NURSE_CONFIG_COLLECTION).findOne({ _id: new ObjectId(nurseId) })
    if (!nurse) {
      response.status(404).json({ error: 'Caregiver not found.' })
      return
    }
    const now = new Date()
    const feedbackId = newId('fbk')
    await db.collection<any>(collections.feedback).insertOne({
      _id: feedbackId,
      nurseId,
      message,
      category: (body.category || '').trim() || null,
      createdAt: now,
      updatedAt: now,
    })
    response.status(201).json({ feedbackId })
  })
}))
