import { Router } from 'express'
import { ObjectId } from 'mongodb'
import { asyncHandler } from '../lib/asyncHandler.js'
import { DB_NAME } from '../lib/constants.js'
import { getSingaporeDateKey, getSingaporeDayBoundsFromKey } from '../lib/dates.js'
import { findPatient } from '../lib/conversations.js'
import { getV1SessionsForPatientRange, getV1TurnsBySession } from '../lib/v1Conversations.js'
import { withMongo } from '../lib/mongo.js'
import { collections } from '../v1/platform/collections.js'
import { newId } from '../v1/platform/ids.js'
import { qwenChatCompletion } from '../v1/platform/qwen.js'
import type { StoredPatient } from '../lib/types.js'

// Bump to invalidate every stored summary at once (e.g. after a prompt change).
const AI_SUMMARY_VERSION = 1
const SUMMARY_SYSTEM_PROMPT =
  'You summarize elderly-care voice companion conversations for a caregiver. Be factual, concise, and avoid diagnosis. Be cognizant of the patient details, especially name, age, gender, preferred language, usual wake time, speech or hearing conditions, speech speed, and key topics. Use those details only to interpret context and personalize wording; do not invent facts or overemphasize profile details that are unrelated to the transcript. Mention mood, notable topics, and whether anything may need follow-up.'

type SummaryBody = {
  patientId?: string
  date?: string
  refresh?: boolean
}

export const patientSummaryRouter = Router()

patientSummaryRouter.post('/', asyncHandler(async (request, response) => {
  const body = request.body as SummaryBody
  if (!body.patientId || !ObjectId.isValid(body.patientId)) {
    response.status(400).json({ error: 'Valid patient id is required.' })
    return
  }
  if (body.date && !/^\d{4}-\d{2}-\d{2}$/.test(body.date)) {
    response.status(400).json({ error: 'Date must be YYYY-MM-DD.' })
    return
  }

  const patientId = new ObjectId(body.patientId)
  const patientHex = patientId.toHexString()
  const summaryDate = body.date || getSingaporeDateKey(new Date())
  const force = body.refresh === true

  await withMongo(async (client) => {
    const db = client.db(DB_NAME)
    const cache = db.collection(collections.dailySummaries)
    const patient = await findPatient(db, patientId)
    const { start, end } = getSingaporeDayBoundsFromKey(summaryDate)
    // v1 sessions come back newest-first; reverse to chronological, then read turns in sequence order.
    const sessions = (await getV1SessionsForPatientRange(db, patientHex, start, end)).reverse()
    const turnsBySession = await getV1TurnsBySession(db, sessions.map((session) => session._id))

    const transcript = sessions
      .flatMap((session) => turnsBySession.get(session._id) || [])
      .map((turn) => `${normalizeRole(turn.role)}: ${turn.text?.trim() || ''}`)
      .filter((line) => !line.endsWith(': '))
      .join('\n')

    if (!transcript.trim()) {
      response.json({ summary: `No conversation transcript is available for ${summaryDate} yet.`, cached: false })
      return
    }

    // Serve the cached summary when the version matches AND the day's transcript hasn't grown since it was
    // written — so a re-open costs no Qwen call, but a later session that day still triggers a refresh.
    const cached = await cache.findOne({ patientId: patientHex, dateKey: summaryDate })
    if (!force && cached && cached.version === AI_SUMMARY_VERSION && cached.transcriptLength === transcript.length) {
      response.json({ summary: cached.summary, cached: true, model: cached.model, generatedAt: cached.generatedAt })
      return
    }

    const { content: summary, model } = await summarizeTranscript(patient, transcript, summaryDate)
    const now = new Date()
    await cache.updateOne(
      { patientId: patientHex, dateKey: summaryDate },
      {
        $set: { summary, version: AI_SUMMARY_VERSION, model, transcriptLength: transcript.length, generatedAt: now, updatedAt: now },
        $setOnInsert: { _id: newId('dsum'), patientId: patientHex, dateKey: summaryDate, createdAt: now },
      },
      { upsert: true },
    )
    response.json({ summary, cached: false, model, generatedAt: now })
  })
}))

async function summarizeTranscript(patient: StoredPatient | null, transcript: string, dateKey: string) {
  const patientDetails = formatPatientDetails(patient)
  // Qwen (OpenAI-compatible) — the key stays server-side; region/model resolved by qwenChatCompletion.
  return qwenChatCompletion({
    messages: [
      { role: 'system', content: SUMMARY_SYSTEM_PROMPT },
      { role: 'user', content: `Date: ${dateKey}\n\nPatient details:\n${patientDetails}\n\nTranscript:\n${transcript}\n\nWrite a 2-4 sentence caregiver summary.` },
    ],
    temperature: 0.2,
    maxTokens: 400,
  })
}

function formatPatientDetails(patient: StoredPatient | null) {
  if (!patient) {
    return 'Name: the patient'
  }

  const keyTopics = [
    ...(patient.keyTopics || []),
    patient.keyTopicsOtherText?.trim() ? patient.keyTopicsOtherText.trim() : '',
  ].filter(Boolean)
  const details = [
    ['Name', patient.name],
    ['Age', patient.age],
    ['Gender', patient.gender],
    ['Preferred language', patient.preferredLanguage],
    ['Usual wake time', patient.usualWakeTime],
    ['Speech or hearing conditions', patient.speechOrHearingConditions],
    ['Speech speed', patient.speechSpeed],
    ['Key topics', keyTopics.length ? keyTopics.join(', ') : undefined],
  ]

  return details
    .filter(([, value]) => value !== undefined && value !== null && String(value).trim())
    .map(([label, value]) => `${label}: ${String(value).trim()}`)
    .join('\n')
}

function normalizeRole(role?: string) {
  const value = role?.toLowerCase()
  return value === 'ai' || value === 'assistant' ? 'Aria' : 'Patient'
}
