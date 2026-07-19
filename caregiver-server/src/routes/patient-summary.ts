import { Router } from 'express'
import { ObjectId } from 'mongodb'
import { asyncHandler } from '../lib/asyncHandler.js'
import { DB_NAME } from '../lib/constants.js'
import { getSingaporeDateKey, getSingaporeDayBoundsFromKey } from '../lib/dates.js'
import { getOpenAIApiKey } from '../lib/env.js'
import { findPatient, getLogsForMaps, getMapsForPatientRange } from '../lib/conversations.js'
import { withMongo } from '../lib/mongo.js'
import type { StoredPatient } from '../lib/types.js'

const OPENAI_SUMMARY_MODEL = 'gpt-4o-mini'

type SummaryBody = {
  patientId?: string
  date?: string
}

export const patientSummaryRouter = Router()

patientSummaryRouter.post('/', asyncHandler(async (request, response) => {
  const apiKey = getOpenAIApiKey()
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
  const summaryDate = body.date || getSingaporeDateKey(new Date())

  await withMongo(async (client) => {
    const db = client.db(DB_NAME)
    const patient = await findPatient(db, patientId)
    const { start, end } = getSingaporeDayBoundsFromKey(summaryDate)
    const maps = await getMapsForPatientRange(db, patientId, start, end)
    maps.reverse()
    const logs = await getLogsForMaps(db, maps)

    if (!logs.length) {
      response.json({ summary: `No conversation transcript is available for ${summaryDate} yet.` })
      return
    }

    const transcript = logs
      .map((log) => `${normalizeRole(log.role)}: ${log.sentence?.trim() || ''}`)
      .filter((line) => !line.endsWith(': '))
      .join('\n')

    if (!transcript.trim()) {
      response.json({ summary: `No conversation transcript is available for ${summaryDate} yet.` })
      return
    }

    const summary = await summarizeTranscript(apiKey, patient, transcript, summaryDate)
    response.json({ summary })
  })
}))

async function summarizeTranscript(apiKey: string, patient: StoredPatient | null, transcript: string, dateKey: string) {
  const patientDetails = formatPatientDetails(patient)
  const openAiResponse = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      authorization: `Bearer ${apiKey}`,
      'content-type': 'application/json',
    },
    body: JSON.stringify({
      model: OPENAI_SUMMARY_MODEL,
      messages: [
        {
          role: 'system',
          content:
            'You summarize elderly-care voice companion conversations for a caregiver. Be factual, concise, and avoid diagnosis. Be cognizant of the patient details, especially name, age, gender, preferred language, usual wake time, speech or hearing conditions, speech speed, and key topics. Use those details only to interpret context and personalize wording; do not invent facts or overemphasize profile details that are unrelated to the transcript. Mention mood, notable topics, and whether anything may need follow-up.',
        },
        {
          role: 'user',
          content: `Date: ${dateKey}\n\nPatient details:\n${patientDetails}\n\nTranscript:\n${transcript}\n\nWrite a 2-4 sentence caregiver summary.`,
        },
      ],
      temperature: 0.2,
    }),
  })
  const body = await openAiResponse.json()

  if (!openAiResponse.ok) {
    throw new Error(body?.error?.message || 'Unable to generate summary.')
  }

  return body?.choices?.[0]?.message?.content?.trim() || 'No summary generated.'
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
  return role?.toLowerCase() === 'ai' ? 'Aria' : 'Patient'
}
