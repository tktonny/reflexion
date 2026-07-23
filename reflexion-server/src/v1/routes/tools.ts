import { Router } from 'express'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { getDb } from '../../lib/mongo.js'
import { getPrincipal, requireActor } from '../platform/auth.js'
import { collections } from '../platform/collections.js'
import { badRequest, conflict, forbidden, notFound } from '../platform/errors.js'
import { sendData } from '../platform/http.js'
import { newId } from '../platform/ids.js'
import { executeIdempotent } from '../platform/idempotency.js'
import { enumValue, objectBody, optionalString, requiredString } from '../platform/validation.js'
import { authorizedSession } from './sessions.js'
import { getWeather, webSearch } from '../tools/providers.js'

// Read tools + two controlled WRITE tools (baseline §2.6): the agent may confirm a medication/reminder
// occurrence conversationally (top-3 WTP feature) and log a caregiver task. It may NOT create or edit
// medication schedules/dosages — those stay caregiver/provider-only via the REST care-plan routes.
const TOOL_NAMES = ['weather.get', 'web.search', 'medication.list', 'reminders.upcoming', 'reminder.respond', 'caregiver_task.create'] as const
const WRITE_TOOL_SCOPES: Partial<Record<typeof TOOL_NAMES[number], string>> = {
  'reminder.respond': 'reminder:respond',
  'caregiver_task.create': 'session:write',
}

export const toolsRouter = Router()
const requireDevice = requireActor('device')

toolsRouter.post('/sessions/:sessionId/tool-invocations', requireDevice, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/sessions/:sessionId/tool-invocations', async () => {
    const session = await authorizedSession(request, request.params.sessionId, 'session:write')
    if (!['created', 'active'].includes(String(session.state))) throw conflict('SESSION_NOT_ACTIVE', 'Tools are only available during an active session.')
    const body = objectBody(request.body)
    const tool = enumValue(body.tool, 'tool', TOOL_NAMES)
    const args = body.arguments && typeof body.arguments === 'object' && !Array.isArray(body.arguments) ? body.arguments as Record<string, unknown> : {}
    const principal = getPrincipal(request)
    if (principal.kind !== 'device') throw forbidden()
    // Controlled write tools require the matching device scope beyond the session:write already proven.
    const requiredScope = WRITE_TOOL_SCOPES[tool]
    if (requiredScope && !principal.scopes.includes(requiredScope)) {
      throw forbidden(`This mirror is not permitted to use ${tool}.`)
    }
    const db = await getDb()
    const invocationId = newId('op')
    const startedAt = new Date()
    await db.collection<any>(collections.toolInvocations).insertOne({
      _id: invocationId, tenantId: principal.tenantId, patientId: principal.patientId, sessionId: session._id,
      tool, arguments: redactArguments(tool, args), state: 'running', createdAt: startedAt,
    })
    try {
      const output = await invokeTool(db, principal.tenantId, principal.patientId, tool, args, session)
      await db.collection<any>(collections.toolInvocations).updateOne({ _id: invocationId }, { $set: { state: 'completed', outputSummary: summarizeOutput(tool, output), completedAt: new Date() } })
      return { status: 200, data: { invocationId, tool, state: 'completed', output, fetchedAt: new Date().toISOString() } }
    } catch (error) {
      await db.collection<any>(collections.toolInvocations).updateOne({ _id: invocationId }, { $set: { state: 'failed', errorCode: error instanceof Error ? error.name : 'ERROR', completedAt: new Date() } })
      throw error
    }
  })
  sendData(response, result.data, result.status)
}))

async function invokeTool(db: Awaited<ReturnType<typeof getDb>>, tenantId: string, patientId: string, tool: typeof TOOL_NAMES[number], args: Record<string, unknown>, session: Record<string, any>) {
  if (tool === 'weather.get') {
    const carePlan = await db.collection<any>(collections.carePlans).findOne({ tenantId, patientId, status: 'active' }, { sort: { version: -1 } })
    const configured = carePlan?.communicationPreferences?.location || {}
    const city = optionalString(args, 'city', 160) || (typeof configured.city === 'string' ? configured.city : undefined)
    const latitude = numberArgument(args.latitude, 'latitude') ?? numberArgument(configured.latitude, 'configured latitude')
    const longitude = numberArgument(args.longitude, 'longitude') ?? numberArgument(configured.longitude, 'configured longitude')
    return getWeather({ city, latitude, longitude, language: String(session.acquisition?.language || 'en') })
  }
  if (tool === 'web.search') {
    const query = requiredString(args, 'query', 300)
    const freshness = args.freshness === undefined ? undefined : enumValue(args.freshness, 'freshness', ['pd', 'pw', 'pm', 'py'] as const)
    return webSearch({ query, freshness, language: String(session.acquisition?.language || 'en') })
  }
  if (tool === 'medication.list') {
    return (await db.collection<any>(collections.medicationPlans).find({ tenantId, patientId, status: 'active' }).project({ _id: 1, displayName: 1, instructions: 1, schedule: 1, source: 1, version: 1 }).toArray())
      .map(({ _id, ...plan }) => ({ planId: _id, ...plan }))
  }
  if (tool === 'reminder.respond') {
    // Confirm a medication/reminder occurrence from the conversation. The occurrence MUST belong to
    // this device's patient; the agent can only mark an existing scheduled occurrence, never invent one.
    const occurrenceId = requiredString(args, 'occurrenceId', 100)
    const status = enumValue(args.status, 'status', ['taken', 'skipped', 'snoozed', 'unknown'] as const)
    const note = optionalString(args, 'note', 500)
    const occurrence = await db.collection<any>(collections.reminderOccurrences).findOne({ _id: occurrenceId, tenantId, patientId })
    if (!occurrence) throw notFound('Reminder occurrence')
    const changed = await db.collection<any>(collections.reminderOccurrences).findOneAndUpdate(
      { _id: occurrence._id, status: { $nin: ['cancelled'] } },
      { $set: { status, respondedAt: new Date(), response: { note, actorType: 'device', channel: 'conversational' }, updatedAt: new Date() } },
      { returnDocument: 'after' },
    )
    if (!changed) throw conflict('REMINDER_NOT_RESPONDABLE', 'This reminder cannot be updated.')
    return { occurrenceId: changed._id, status: changed.status, respondedAt: changed.respondedAt }
  }
  if (tool === 'caregiver_task.create') {
    const task = {
      _id: newId('task'), tenantId, patientId,
      category: enumValue(args.category, 'category', ['follow_up', 'appointment', 'medication_review', 'technical', 'custom'] as const),
      priority: enumValue(args.priority, 'priority', ['routine', 'elevated', 'urgent'] as const),
      title: requiredString(args, 'title', 160), details: optionalString(args, 'details', 1000),
      sourceRef: `session:${session._id}`, dueAt: null,
      status: 'open', createdBy: { type: 'device', id: session.deviceId || null }, createdAt: new Date(),
    }
    await db.collection<any>(collections.caregiverTasks).insertOne(task)
    return { taskId: task._id, category: task.category, priority: task.priority, title: task.title, status: task.status }
  }
  const now = new Date(); const to = new Date(now.getTime() + 24 * 60 * 60 * 1000)
  return (await db.collection<any>(collections.reminderOccurrences).find({ tenantId, patientId, scheduledAt: { $gte: now, $lt: to }, status: { $in: ['scheduled', 'delivered', 'snoozed'] } }).sort({ scheduledAt: 1 }).toArray())
    .map(({ _id, ...occurrence }) => ({ occurrenceId: _id, ...occurrence }))
}

function numberArgument(value: unknown, field: string) {
  if (value === undefined || value === null) return undefined
  if (typeof value !== 'number' || !Number.isFinite(value)) throw badRequest('VALIDATION_FAILED', `${field} must be a finite number.`)
  return value
}
function redactArguments(tool: string, args: Record<string, unknown>) {
  if (tool === 'web.search') return { queryHashOnly: true, freshness: args.freshness }
  // Free-text notes/titles may contain personal content — persist structure, not the words.
  if (tool === 'reminder.respond') return { occurrenceId: args.occurrenceId, status: args.status, noteProvided: Boolean(args.note) }
  if (tool === 'caregiver_task.create') return { category: args.category, priority: args.priority, titleProvided: Boolean(args.title) }
  return args
}
function summarizeOutput(tool: string, output: unknown) { return { tool, itemCount: Array.isArray(output) ? output.length : 1 } }
