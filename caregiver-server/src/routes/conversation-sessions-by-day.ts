import { Router } from 'express'
import { ObjectId } from 'mongodb'
import { asyncHandler } from '../lib/asyncHandler.js'
import { DB_NAME } from '../lib/constants.js'
import { getSingaporeDateKey, getSingaporeDayBoundsFromKey } from '../lib/dates.js'
import { findPatient } from '../lib/conversations.js'
import { getV1SessionsForPatientRange, getV1TurnsBySession, serializeV1Session } from '../lib/v1Conversations.js'
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
  const patientHex = patientId.toHexString()

  await withMongo(async (client) => {
    const db = client.db(DB_NAME)
    const { start, end } = getSingaporeDayBoundsFromKey(date)
    const v1Sessions = await getV1SessionsForPatientRange(db, patientHex, start, end)
    const patient = await findPatient(db, patientId)
    const patientName = patient?.name || 'Patient'
    const turnsBySession = await getV1TurnsBySession(db, v1Sessions.map((session) => session._id))
    const sessions = v1Sessions.map((session) =>
      serializeV1Session(session, turnsBySession.get(session._id) || [], patientHex, patientName, patient?.preferredLanguage),
    )

    response.json({
      date,
      patientName,
      patientId: patientHex,
      sessions,
    })
  })
}))
