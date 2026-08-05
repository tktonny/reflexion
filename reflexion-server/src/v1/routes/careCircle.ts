import { Router } from 'express'
import { asyncHandler } from '../../lib/asyncHandler.js'
import { getDb } from '../../lib/mongo.js'
import { authorizePatient, getPrincipal, requireActor } from '../platform/auth.js'
import { CAREGIVER_RELATIONSHIP_SCOPES } from '../platform/scopes.js'
import { collections } from '../platform/collections.js'
import { badRequest, conflict, notFound } from '../platform/errors.js'
import { sendData } from '../platform/http.js'
import { newId } from '../platform/ids.js'
import { hashSecret, sealSecret, sha256 } from '../platform/crypto.js'
import { randomSecret } from '../platform/ids.js'
import { executeIdempotent } from '../platform/idempotency.js'
import { appendOutbox } from '../platform/outbox.js'
import { enumValue, objectBody, optionalString, stringArray } from '../platform/validation.js'

const ROLES = ['full-access', 'standard-access', 'view-only', 'custom-access'] as const
const PERMISSIONS = ['view-loved-ones', 'receive-notifications', 'manage-routines', 'manage-devices', 'invite-or-remove-caregivers'] as const
const ALL_SCOPES = [...CAREGIVER_RELATIONSHIP_SCOPES, 'care_circle:manage']

export const careCircleRouter = Router()
const requireHuman = requireActor('human')

careCircleRouter.get('/patients/:patientId/care-circle', requireHuman, asyncHandler(async (request, response) => {
  const patientId = request.params.patientId
  await authorizePatient(request, patientId, 'patient:read')
  const principal = getPrincipal(request)
  const db = await getDb()
  const [relationships, invitations] = await Promise.all([
    db.collection<any>(collections.careRelationships).find({ tenantId: principal.tenantId, patientId, status: 'active' }).sort({ createdAt: 1 }).toArray(),
    db.collection<any>(collections.careCircleInvitations).find({ tenantId: principal.tenantId, patientId, state: 'pending' }).sort({ createdAt: -1 }).toArray(),
  ])
  const userIds = relationships.map((row) => String(row.userId)).filter(Boolean)
  const users = await db.collection<any>(collections.users).find({ tenantId: principal.tenantId, _id: { $in: userIds } }).project({ name: 1, email: 1, phoneNumber: 1 }).toArray()
  const byId = new Map(users.map((user) => [String(user._id), user]))
  sendData(response, {
    patientId,
    members: relationships.map((row) => serializeMember(row, byId.get(String(row.userId)))),
    invitations: invitations.map(serializeInvitation),
  })
}))

careCircleRouter.post('/patients/:patientId/care-circle/invitations', requireHuman, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'POST:/api/v1/patients/:patientId/care-circle/invitations', async () => {
    const patientId = request.params.patientId
    await authorizePatient(request, patientId, 'patient:write')
    const body = objectBody(request.body)
    const invitee = optionalString(body, 'emailOrPhone', 320)
    if (!invitee) throw badRequest('VALIDATION_FAILED', 'emailOrPhone is required.')
    const role = enumValue(body.role, 'role', ROLES)
    const permissions = requestedPermissions(body, role)
    const principal = getPrincipal(request)
    if (principal.kind !== 'human') throw badRequest('HUMAN_REQUIRED', 'A caregiver account is required.')
    const inviteeNormalized = normalizeInvitee(invitee)
    const db = await getDb()
    const existing = await db.collection<any>(collections.careCircleInvitations).findOne({ tenantId: principal.tenantId, patientId, inviteeNormalized, state: 'pending' })
    if (existing) return { status: 200, data: serializeInvitation(existing) }
    const now = new Date()
    const inviteToken = randomSecret()
    const invitation = {
      _id: newId('inv'), tenantId: principal.tenantId, patientId, invitedBy: principal.userId,
      invitee, inviteeNormalized, role, permissions, scopes: scopesFor(role, permissions), state: 'pending',
      inviteTokenDigest: sha256(inviteToken), inviteTokenHash: hashSecret(inviteToken),
      createdAt: now, updatedAt: now,
    }
    await db.collection<any>(collections.careCircleInvitations).insertOne(invitation)
    await appendOutbox(db, {
      eventType: 'care_circle.invitation.requested', tenantId: principal.tenantId, patientId,
      aggregateType: 'care_circle_invitation', aggregateId: invitation._id, correlationId: request.requestId,
      payload: { invitee, role, permissions, sealedToken: sealSecret(inviteToken) },
    })
    return { status: 202, data: serializeInvitation(invitation) }
  })
  sendData(response, result.data, result.status)
}))

careCircleRouter.patch('/patients/:patientId/care-circle/:memberId', requireHuman, asyncHandler(async (request, response) => {
  const patientId = request.params.patientId
  await authorizePatient(request, patientId, 'patient:write')
  const body = objectBody(request.body)
  const role = enumValue(body.role, 'role', ROLES)
  const permissions = requestedPermissions(body, role)
  const principal = getPrincipal(request)
  const db = await getDb()
  const update = { role, permissions, scopes: scopesFor(role, permissions), updatedAt: new Date() }
  const relationship = await db.collection<any>(collections.careRelationships).findOne({ _id: request.params.memberId, tenantId: principal.tenantId, patientId, status: 'active' })
  if (relationship) {
    const version = Number(relationship.version || 1)
    const expected = request.header('If-Match') ? parseIfMatch(request.header('If-Match')) : version
    if (expected !== version) throw conflict('VERSION_CONFLICT', 'This caregiver changed. Refresh and retry.')
    const updated = await db.collection<any>(collections.careRelationships).findOneAndUpdate({ _id: relationship._id, version }, { $set: update, $inc: { version: 1 } }, { returnDocument: 'after' })
    sendData(response, serializeMember(updated || { ...relationship, ...update, version: version + 1 }))
    return
  }
  const invitation = await db.collection<any>(collections.careCircleInvitations).findOne({ _id: request.params.memberId, tenantId: principal.tenantId, patientId, state: 'pending' })
  if (!invitation) throw notFound('Care Circle member')
  const updated = await db.collection<any>(collections.careCircleInvitations).findOneAndUpdate({ _id: invitation._id, state: 'pending' }, { $set: update }, { returnDocument: 'after' })
  sendData(response, serializeInvitation(updated || { ...invitation, ...update }))
}))

careCircleRouter.delete('/patients/:patientId/care-circle/:memberId', requireHuman, asyncHandler(async (request, response) => {
  const result = await executeIdempotent(request, 'DELETE:/api/v1/patients/:patientId/care-circle/:memberId', async () => {
    const patientId = request.params.patientId
    await authorizePatient(request, patientId, 'patient:write')
    const principal = getPrincipal(request)
    const db = await getDb()
    const now = new Date()
    const relationship = await db.collection<any>(collections.careRelationships).findOneAndUpdate(
      { _id: request.params.memberId, tenantId: principal.tenantId, patientId, status: 'active' },
      { $set: { status: 'revoked', validTo: now, revokedAt: now, updatedAt: now }, $inc: { version: 1 } },
      { returnDocument: 'after' },
    )
    if (relationship) return { status: 202, data: { memberId: request.params.memberId, state: 'revoked' } }
    const invitation = await db.collection<any>(collections.careCircleInvitations).findOneAndUpdate(
      { _id: request.params.memberId, tenantId: principal.tenantId, patientId, state: 'pending' },
      { $set: { state: 'revoked', revokedAt: now, updatedAt: now } }, { returnDocument: 'after' },
    )
    if (!invitation) throw notFound('Care Circle member')
    return { status: 202, data: { memberId: request.params.memberId, state: 'revoked' } }
  })
  sendData(response, result.data, result.status)
}))

function requestedPermissions(body: Record<string, unknown>, role: typeof ROLES[number]) {
  if (role !== 'custom-access') return defaultPermissions(role)
  const permissions = stringArray(body.permissions, 'permissions', PERMISSIONS.length)
  const unique = [...new Set(permissions)]
  if (!unique.length || unique.some((item) => !(PERMISSIONS as readonly string[]).includes(item))) throw badRequest('VALIDATION_FAILED', 'Custom access requires valid permissions.')
  return unique
}

function defaultPermissions(role: typeof ROLES[number]) {
  if (role === 'full-access') return [...PERMISSIONS]
  if (role === 'standard-access') return ['view-loved-ones', 'receive-notifications', 'manage-routines']
  if (role === 'view-only') return ['view-loved-ones']
  return []
}

function scopesFor(role: typeof ROLES[number], permissions: string[]) {
  if (role === 'full-access') return [...ALL_SCOPES]
  const scopes = new Set<string>(['patient:read', 'session:read'])
  if (permissions.includes('receive-notifications')) scopes.add('monitoring:read')
  if (permissions.includes('manage-routines')) scopes.add('care_plan:read').add('care_plan:write')
  if (permissions.includes('manage-devices')) scopes.add('device:assign')
  if (permissions.includes('invite-or-remove-caregivers')) scopes.add('care_circle:manage')
  return [...scopes]
}

function serializeMember(row: Record<string, any>, user?: Record<string, any>) {
  return {
    memberId: row._id, kind: 'member', userId: row.userId, name: user?.name || 'Caregiver', email: user?.email || null,
    phoneNumber: user?.phoneNumber || null, role: row.role || roleFromScopes(row.scopes), permissions: row.permissions || permissionsFromScopes(row.scopes),
    state: row.status, version: Number(row.version || 1),
  }
}

function serializeInvitation(row: Record<string, any>) {
  return { memberId: row._id, kind: 'invitation', invitee: row.invitee, role: row.role, permissions: row.permissions || [], state: row.state, createdAt: row.createdAt, version: Number(row.version || 1) }
}

function normalizeInvitee(value: string) {
  const trimmed = value.trim().toLowerCase()
  return trimmed.includes('@') ? trimmed : trimmed.replace(/[^+\d]/g, '')
}

function roleFromScopes(scopes: unknown) {
  const values = new Set(Array.isArray(scopes) ? scopes.map(String) : [])
  if (values.has('care_circle:manage') && values.has('device:assign')) return 'full-access'
  if (values.has('care_plan:write')) return 'standard-access'
  return 'view-only'
}

function permissionsFromScopes(scopes: unknown) {
  const values = new Set(Array.isArray(scopes) ? scopes.map(String) : [])
  return [
    values.has('patient:read') ? 'view-loved-ones' : null,
    values.has('monitoring:read') ? 'receive-notifications' : null,
    values.has('care_plan:write') ? 'manage-routines' : null,
    values.has('device:assign') ? 'manage-devices' : null,
    values.has('care_circle:manage') ? 'invite-or-remove-caregivers' : null,
  ].filter(Boolean)
}

function parseIfMatch(value?: string) {
  const version = Number(value?.replace(/^W\//, '').replaceAll('"', ''))
  if (!Number.isInteger(version) || version < 1) throw badRequest('IF_MATCH_REQUIRED', 'If-Match must contain the current integer version.')
  return version
}
