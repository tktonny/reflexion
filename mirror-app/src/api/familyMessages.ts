import { dataOrThrow } from './devicePairing'
import { deviceFetch } from '../storage/deviceCredentials'

export type MirrorFamilyMessage = {
  messageId: string
  kind: 'text' | 'photo' | 'voice'
  senderName: string
  text?: string | null
  caption?: string | null
  mediaUrl?: string | null
  deliveryState: 'queued' | 'delivered' | 'expired' | 'failed'
  interactionState: 'viewed' | 'played' | 'replayed' | null
  createdAt: string
  deliveredAt?: string | null
  voiceReplies?: MirrorVoiceReply[]
}

export type MirrorVoiceReply = {
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
  createdAt: string
  sentAt: string | null
  deliveredAt: string | null
}

export async function getDeviceFamilyMessages(deviceId: string) {
  const response = await deviceFetch(`/api/v1/devices/${encodeURIComponent(deviceId)}/family-messages`)
  return dataOrThrow<MirrorFamilyMessage[]>(response)
}

export async function recordFamilyMessageInteraction(deviceId: string, messageId: string, interaction: 'viewed' | 'played' | 'replayed') {
  const response = await deviceFetch(`/api/v1/devices/${encodeURIComponent(deviceId)}/family-messages/${encodeURIComponent(messageId)}/interactions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Idempotency-Key': `family_message_${messageId}_${interaction}` },
    body: JSON.stringify({ interaction }),
  })
  return dataOrThrow<{ messageId: string; interactionState: string }>(response)
}

export type VoiceReplyUploadPlan = {
  replyId: string
  state: 'queued' | 'sent'
  uploadUrl?: string
  expiresAt?: string
  requiredHeaders?: Record<string, string>
  alreadySent?: boolean
  reply?: MirrorVoiceReply
}

export async function prepareFamilyVoiceReplyUpload(input: {
  deviceId: string
  messageId: string
  clientReplyId: string
  contentType: string
  hash: string
  sizeBytes: number
  durationMs: number
  sessionId?: string | null
}) {
  const response = await deviceFetch(`/api/v1/devices/${encodeURIComponent(input.deviceId)}/family-messages/${encodeURIComponent(input.messageId)}/voice-replies/upload-plans`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Idempotency-Key': `voice_reply_plan_${input.clientReplyId}` },
    body: JSON.stringify({
      clientReplyId: input.clientReplyId, contentType: input.contentType, hash: input.hash,
      sizeBytes: input.sizeBytes, durationMs: input.durationMs, sessionId: input.sessionId || undefined,
    }),
  })
  return dataOrThrow<VoiceReplyUploadPlan>(response)
}

export async function commitFamilyVoiceReply(input: {
  deviceId: string
  messageId: string
  replyId: string
  hash: string
  sizeBytes: number
  clientReplyId: string
}) {
  const response = await deviceFetch(`/api/v1/devices/${encodeURIComponent(input.deviceId)}/family-messages/${encodeURIComponent(input.messageId)}/voice-replies/${encodeURIComponent(input.replyId)}/commit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Idempotency-Key': `voice_reply_commit_${input.clientReplyId}` },
    body: JSON.stringify({ hash: input.hash, sizeBytes: input.sizeBytes }),
  })
  return dataOrThrow<MirrorVoiceReply>(response)
}

export async function uploadFamilyVoiceReply(plan: VoiceReplyUploadPlan, bytes: Uint8Array) {
  if (!plan.uploadUrl) return
  if (!isTrustedUploadUrl(plan.uploadUrl)) throw new Error('voice_reply_upload_url_invalid')
  const response = await fetch(plan.uploadUrl, {
    method: 'PUT',
    headers: plan.requiredHeaders,
    body: bytes.buffer as ArrayBuffer,
  })
  if (!response.ok) throw new Error(`voice_reply_upload_failed_${response.status}`)
}

function isTrustedUploadUrl(value: string) {
  try {
    const parsed = new URL(value)
    return parsed.protocol === 'https:' || (__DEV__ && parsed.protocol === 'http:')
  } catch {
    return false
  }
}
