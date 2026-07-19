import { Router } from 'express'
import { ObjectId } from 'mongodb'
import { asyncHandler } from '../lib/asyncHandler.js'
import { DB_NAME } from '../lib/constants.js'
import { getSingaporeDateKey, getSingaporeDayBoundsFromKey } from '../lib/dates.js'
import { findPatient, getConversationsByMaps, getMapsForPatientRange, serializeConversation } from '../lib/conversations.js'
import { withMongo } from '../lib/mongo.js'

export const conversationSessionsByDayRouter = Router()

conversationSessionsByDayRouter.get('/', asyncHandler(async (request, response) => {
  const id = typeof request.query.id === 'string' ? request.query.id : ''
  const date = typeof request.query.date === 'string' ? request.query.date : getSingaporeDateKey(new Date())

  if (!id || !ObjectId.isValid(id)) {
    response.status(400).json({ error: 'Valid patient id is required' })
    return
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    response.status(400).json({ error: 'Date must be YYYY-MM-DD.' })
    return
  }

  const patientId = new ObjectId(id)

  await withMongo(async (client) => {
    const db = client.db(DB_NAME)
    const { start, end } = getSingaporeDayBoundsFromKey(date)
    const maps = await getMapsForPatientRange(db, patientId, start, end)
    const conversationById = await getConversationsByMaps(db, maps)
    const patient = await findPatient(db, patientId)
    const sessions = maps
      .map((map) => {
        const conversationId = map.conversationId?.toHexString?.() || ''
        const conversation = conversationById.get(conversationId)
        if (!conversation) return null
        return serializeConversation(conversationId, patientId, patient?.name || 'Patient', conversation, map)
      })
      .filter((session): session is ReturnType<typeof serializeConversation> => Boolean(session))

    response.json({
      date,
      patientName: patient?.name || 'Patient',
      patientId: patientId.toHexString(),
      sessions,
    })
  })
}))
