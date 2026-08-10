import { Router } from 'express'

import { asyncHandler } from '../../lib/asyncHandler.js'
import { getDb } from '../../lib/mongo.js'
import { authorizePatient, getPrincipal, requireActor } from '../platform/auth.js'
import { collections } from '../platform/collections.js'
import { badRequest, conflict, notFound } from '../platform/errors.js'
import { sendData } from '../platform/http.js'
import { newId } from '../platform/ids.js'
import { executeIdempotent } from '../platform/idempotency.js'
import { getObjectStore } from '../platform/objectStore.js'
import { enumValue, objectBody, optionalString, positiveInteger, requiredString } from '../platform/validation.js'
import { authorizedDevice } from './devices.js'

const MESSAGE_KINDS = ['text', 'photo', 'voice'] as const
const INTERACTIONS = ['viewed', 'replayed', 'played'] as const
const VOICE_REPLY_CONTENT_TYPES = ['audio/webm', 'audio/mp4', 'audio/m4a', 'audio/x-m4a', 'audio/wav'] as const
const MAX_VOICE_REPLY_BYTES = 8 * 1024 * 1024
const MAX_VOICE_REPLY_DURATION_MS = 5 * 60 * 1000

type SerializedVoiceReply = {
  replyId: string
  originalMessageId: string
  threadId: string
  senderName: string
  patientId: string
  deviceId: string
  kind: 'voice'
  state: 'queued' | 'sent' | 'failed'
  deliveryState: 'queued' | 'delivered' | 'failed'
  audioUrl: string | null
  contentType: string | null
  durationMs: number
  createdAt: Date | string
  sentAt: Date | string | null
  deliveredAt: Date | string | null
}

type VoiceReplyUploadPlanResponse = {
  replyId: string
  state: 'queued' | 'sent'
  uploadUrl?: string
  expiresAt?: string
  requiredHeaders?: Record<string, string>
  alreadySent?: boolean
  reply?: SerializedVoiceReply
}

export const familyMessagesRouter = Router()

familyMessagesRouter.get('/patients/:patientId/family-messages', requireActor('human'), asyncHandler(async (request, response) => {
  const patientId = request.params.patientId
  await authorizePatient(request, patientId, 'patient:read')
  const principal = getPrincipal(request)
  const db = await getDb()
  const messages = await db.collection<any>(collections.familyMessages).find({ tenantId: principal.tenantId, patientId }).sort({ createdAt: -1 }).limit(100).toArray()
  sendData(response, await Promise.all(messages.map(serializeMessage)))
}))

familyMessagesRouter.post('/patients/:patientId/family-messages', requireActor('human'), asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/patients/:patientId/family-messages', async () => {
    const patientId = request.params.patientId
    await authorizePatient(request, patientId, 'patient:write')
    const principal = getPrincipal(request)
    const body = objectBody(request.body)
    // Accept the current { kind, text/caption, mediaUrl } contract and the short-lived legacy
    // { body, scheduledFor } caregiver contract while the canonical caregiver UI is being migrated.
    const kind = body.kind === undefined ? 'text' : enumValue(body.kind, 'kind', MESSAGE_KINDS)
    const text = optionalString(body, 'text', 5000) || optionalString(body, 'body', 5000)
    const caption = optionalString(body, 'caption', 2000)
    const mediaUrl = optionalString(body, 'mediaUrl', 4000)
    if (kind === 'text' && !text) throw badRequest('MESSAGE_CONTENT_REQUIRED', 'Text message content is required.')
    if (kind === 'photo' && !mediaUrl) throw badRequest('MESSAGE_MEDIA_REQUIRED', 'Photo message media is required.')
    if (kind === 'voice' && !mediaUrl) throw badRequest('MESSAGE_MEDIA_REQUIRED', 'Voice message media is required.')
    const scheduledForRaw = optionalString(body, 'scheduledFor', 80)
    const scheduledFor = scheduledForRaw ? new Date(scheduledForRaw) : new Date()
    if (Number.isNaN(scheduledFor.getTime())) throw badRequest('VALIDATION_FAILED', 'scheduledFor must be an ISO-8601 date.')
    const now = new Date()
    const deliveryState = scheduledFor.getTime() > now.getTime() ? 'scheduled' : 'queued'
    const message = {
      _id: newId('msg'), tenantId: principal.tenantId, patientId, kind,
      senderName: optionalString(body, 'senderName', 160) || 'Your family',
      text: text || null, caption: caption || null, mediaUrl: mediaUrl || null,
      deliveryState, state: deliveryState, scheduledFor, interactionState: null, interactionEvents: [],
      createdAt: now, updatedAt: now, deliveredAt: null, openedAt: null,
    }
    const db = await getDb()
    await db.collection<any>(collections.familyMessages).insertOne(message)
    return { status: 201, data: await serializeMessage(message) }
  })
  sendData(response, result.data, result.status)
}))

familyMessagesRouter.get('/devices/:deviceId/family-messages', requireActor('device'), asyncHandler(async (request, response) => {
  const { assignment } = await authorizedDevice(request, request.params.deviceId)
  const db = await getDb()
  const messages = await db.collection<any>(collections.familyMessages).find({
    tenantId: assignment.tenantId, patientId: assignment.patientId,
    deliveryState: { $in: ['queued', 'delivered'] }, dismissedAt: { $exists: false },
  }).sort({ createdAt: 1 }).limit(20).toArray()
  if (messages.length) {
    await db.collection<any>(collections.familyMessages).updateMany(
      { _id: { $in: messages.map((message) => message._id) }, deliveryState: 'queued' },
      { $set: { deliveryState: 'delivered', deliveredAt: new Date(), updatedAt: new Date() } },
    )
  }
  sendData(response, await Promise.all(messages.map((message) => serializeMessage({ ...message, deliveryState: message.deliveryState === 'queued' ? 'delivered' : message.deliveryState }))))
}))

familyMessagesRouter.post('/devices/:deviceId/family-messages/:messageId/interactions', requireActor('device'), asyncHandler(async (request, response) => {
  const { assignment } = await authorizedDevice(request, request.params.deviceId)
  const body = objectBody(request.body)
  const interaction = enumValue(body.interaction, 'interaction', INTERACTIONS)
  const db = await getDb()
  const message = await db.collection<any>(collections.familyMessages).findOne({ _id: request.params.messageId, tenantId: assignment.tenantId, patientId: assignment.patientId })
  if (!message) throw notFound('Family message')
  const nextInteraction = interaction === 'replayed'
    ? 'replayed'
    : message.interactionState === 'replayed'
      ? 'replayed'
      : interaction
  await db.collection<any>(collections.familyMessages).updateOne({ _id: message._id }, {
    $set: {
      interactionState: nextInteraction,
      interactionEvents: [...(Array.isArray(message.interactionEvents) ? message.interactionEvents : []), { interaction, occurredAt: new Date() }],
      updatedAt: new Date(),
    },
  })
  sendData(response, { messageId: message._id, interactionState: nextInteraction })
}))

/**
 * The Mirror creates the upload intent only after the loved one explicitly taps Send. The bytes are
 * uploaded directly to the configured object store; this API never accepts audio in JSON or logs it.
 */
familyMessagesRouter.post('/devices/:deviceId/family-messages/:messageId/voice-replies/upload-plans', requireActor('device'), asyncHandler(async (request, response) => {
  const result = await executeIdempotent<VoiceReplyUploadPlanResponse>(request, 'POST:/api/v1/devices/:deviceId/family-messages/:messageId/voice-replies/upload-plans', async () => {
    const { assignment } = await authorizedDevice(request, request.params.deviceId)
    const body = objectBody(request.body)
    const clientReplyId = requiredString(body, 'clientReplyId', 120)
    const contentType = enumValue(body.contentType, 'contentType', VOICE_REPLY_CONTENT_TYPES)
    const hash = requiredString(body, 'hash', 128).toLowerCase()
    if (!/^[a-f0-9]{64}$/.test(hash)) throw badRequest('VALIDATION_FAILED', 'hash must be a SHA-256 hex digest.', [{ field: 'hash' }])
    const sizeBytes = positiveInteger(body.sizeBytes, 'sizeBytes')
    const durationMs = positiveInteger(body.durationMs, 'durationMs')
    if (sizeBytes > MAX_VOICE_REPLY_BYTES) throw badRequest('VOICE_REPLY_TOO_LARGE', 'Voice replies must be 8 MB or smaller.')
    if (durationMs > MAX_VOICE_REPLY_DURATION_MS) throw badRequest('VOICE_REPLY_TOO_LONG', 'Voice replies must be five minutes or shorter.')
    const sessionId = optionalString(body, 'sessionId', 160) || null
    const db = await getDb()
    const original = await db.collection<any>(collections.familyMessages).findOne({
      _id: request.params.messageId, tenantId: assignment.tenantId, patientId: assignment.patientId,
    })
    if (!original) throw notFound('Family message')

    const replies = db.collection<any>(collections.familyMessageReplies)
    const existing = await replies.findOne({ tenantId: assignment.tenantId, patientId: assignment.patientId, clientReplyId })
    if (existing) {
      if (String(existing.originalMessageId) !== String(original._id) || existing.audio?.hash !== hash || Number(existing.audio?.sizeBytes) !== sizeBytes) {
        throw conflict('CLIENT_REPLY_ID_REUSED', 'This voice reply id was already used with different content.')
      }
      if (existing.state === 'sent') {
        return { status: 200, data: { replyId: String(existing._id), state: 'sent' as const, alreadySent: true, reply: await serializeVoiceReply(existing) } }
      }
    }

    const replyId = String(existing?._id || newId('vrep'))
    const objectKey = String(existing?.audio?.objectKey || `${assignment.tenantId}/${assignment.patientId}/family-replies/${original._id}/${replyId}`)
    const plan = await getObjectStore().prepareUpload({ objectKey, contentType, hash })
    const now = new Date()
    const reply = {
      _id: replyId,
      tenantId: assignment.tenantId,
      patientId: assignment.patientId,
      deviceId: request.params.deviceId,
      sessionId,
      originalMessageId: original._id,
      threadId: original.threadId || original._id,
      clientReplyId,
      sender: { type: 'loved_one', patientId: assignment.patientId },
      kind: 'voice',
      audio: { objectKey, contentType, hash, sizeBytes, durationMs },
      state: 'queued',
      deliveryState: 'queued',
      createdAt: existing?.createdAt || now,
      updatedAt: now,
      uploadExpiresAt: plan.expiresAt,
    }
    await replies.updateOne({ _id: replyId }, { $set: reply }, { upsert: true })
    return {
      status: 200,
      data: {
        replyId,
        state: 'queued' as const,
        uploadUrl: plan.uploadUrl,
        expiresAt: plan.expiresAt.toISOString(),
        requiredHeaders: plan.requiredHeaders,
      },
    }
  })
  response.setHeader('Cache-Control', 'no-store')
  sendData(response, result.data, result.status)
}))

familyMessagesRouter.post('/devices/:deviceId/family-messages/:messageId/voice-replies/:replyId/commit', requireActor('device'), asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/devices/:deviceId/family-messages/:messageId/voice-replies/:replyId/commit', async () => {
    const { assignment } = await authorizedDevice(request, request.params.deviceId)
    const body = objectBody(request.body)
    const hash = requiredString(body, 'hash', 128).toLowerCase()
    const sizeBytes = positiveInteger(body.sizeBytes, 'sizeBytes')
    const db = await getDb()
    const replies = db.collection<any>(collections.familyMessageReplies)
    const reply = await replies.findOne({
      _id: request.params.replyId, tenantId: assignment.tenantId, patientId: assignment.patientId,
      deviceId: request.params.deviceId, originalMessageId: request.params.messageId,
    })
    if (!reply) throw notFound('Voice reply')
    if (reply.state === 'sent') return { status: 200, data: await serializeVoiceReply(reply) }
    if (reply.audio?.hash !== hash || Number(reply.audio?.sizeBytes) !== sizeBytes) {
      throw conflict('VOICE_REPLY_MISMATCH', 'The voice reply does not match its upload intent.')
    }
    const verified = await getObjectStore().verify({ objectKey: String(reply.audio.objectKey), hash, sizeBytes })
    if (!verified) throw conflict('VOICE_REPLY_UPLOAD_NOT_VERIFIED', 'The voice reply upload is not available yet.', true)
    const now = new Date()
    await replies.updateOne({ _id: reply._id, state: { $ne: 'sent' } }, { $set: {
      state: 'sent', deliveryState: 'delivered', sentAt: now, deliveredAt: now, updatedAt: now,
    } })
    const committed = await replies.findOne({ _id: reply._id })
    return { status: 200, data: await serializeVoiceReply(committed || { ...reply, state: 'sent', deliveryState: 'delivered', sentAt: now, deliveredAt: now }) }
  })
  response.setHeader('Cache-Control', 'no-store')
  sendData(response, result.data, result.status)
}))

async function serializeMessage(message: Record<string, any>) {
  const deliveryState = String(message.deliveryState || message.state || 'queued')
  const interactionState = message.interactionState || (message.openedAt ? 'viewed' : null)
  const text = message.text ?? message.body ?? null
  const kind = message.kind || (message.type === 'text' ? 'text' : message.type) || 'text'
  return {
    messageId: String(message._id), kind, senderName: message.senderName || 'Your family',
    text, caption: message.caption || null, mediaUrl: message.mediaUrl || null,
    deliveryState, interactionState,
    createdAt: message.createdAt, deliveredAt: message.deliveredAt || null,
    voiceReplies: await serializeVoiceReplies(message),
    // Compatibility aliases for the previous caregiver data layer. They can be removed once the
    // canonical caregiver Chat thread consumes the new fields everywhere.
    body: text, type: kind, state: interactionState === 'viewed' ? 'opened' : deliveryState,
    scheduledFor: message.scheduledFor || message.createdAt, openedAt: message.openedAt || null,
  }
}

async function serializeVoiceReplies(message: Record<string, any>) {
  const db = await getDb()
  const replies = await db.collection<any>(collections.familyMessageReplies).find({
    tenantId: message.tenantId, patientId: message.patientId, originalMessageId: message._id,
  }).sort({ createdAt: 1 }).toArray()
  return Promise.all(replies.map(serializeVoiceReply))
}

async function serializeVoiceReply(reply: Record<string, any>): Promise<SerializedVoiceReply> {
  let audioUrl: string | null = null
  if (reply.state === 'sent' && reply.audio?.objectKey) {
    try {
      audioUrl = (await getObjectStore().prepareDownload({
        objectKey: String(reply.audio.objectKey),
        contentType: String(reply.audio.contentType || 'audio/webm'),
      })).downloadUrl
    } catch {
      // A missing object-store configuration should not prevent caregivers from seeing the reply state.
    }
  }
  return {
    replyId: String(reply._id), originalMessageId: String(reply.originalMessageId), threadId: String(reply.threadId || reply.originalMessageId),
    senderName: reply.sender?.displayName || 'Your loved one', patientId: String(reply.patientId), deviceId: String(reply.deviceId),
    kind: 'voice', state: reply.state as SerializedVoiceReply['state'], deliveryState: reply.deliveryState as SerializedVoiceReply['deliveryState'],
    audioUrl, contentType: reply.audio?.contentType || null, durationMs: Number(reply.audio?.durationMs || 0),
    createdAt: reply.createdAt, sentAt: reply.sentAt || null, deliveredAt: reply.deliveredAt || null,
  }
}
