import type { NextFunction, Request, Response } from 'express'
import { getDb } from '../../lib/mongo.js'
import { collections } from './collections.js'
import { forbidden, unauthorized } from './errors.js'
import type { DevicePrincipal, HumanPrincipal, Principal } from './principal.js'
import { verifyAccessToken, type TokenKind } from './tokens.js'

export function requireActor(...kinds: TokenKind[]) {
  return async (request: Request, _response: Response, next: NextFunction) => {
    try {
      const authorization = request.header('Authorization') || ''
      if (!authorization.startsWith('Bearer ')) throw unauthorized()
      const claims = verifyAccessToken(authorization.slice(7), kinds.length ? kinds : ['human', 'device'])
      const db = await getDb()
      if (claims.kind === 'human') {
        if (!claims.uid || !claims.tid || !claims.sid) throw unauthorized()
        const activeSession = await db.collection<any>(collections.authSessions).findOne({
          _id: claims.sid,
          tenantId: claims.tid,
          userId: claims.uid,
          status: 'active',
          refreshExpiresAt: { $gt: new Date() },
        }, { projection: { _id: 1 } })
        if (!activeSession) throw unauthorized('The login session has ended.')
        request.principal = {
          kind: 'human', subjectId: claims.sub, userId: claims.uid, tenantId: claims.tid,
          sessionId: claims.sid, roles: claims.roles || [], scopes: claims.scopes || [],
        } satisfies HumanPrincipal
      } else if (claims.kind === 'device') {
        if (!claims.did || !claims.pid || !claims.tid || !claims.cid) throw unauthorized()
        const [credential, assignment] = await Promise.all([
          db.collection<any>(collections.credentials).findOne({
            _id: claims.cid, deviceId: claims.did, status: 'active', refreshExpiresAt: { $gt: new Date() },
          }, { projection: { _id: 1 } }),
          db.collection<any>(collections.assignments).findOne({
            tenantId: claims.tid, deviceId: claims.did, patientId: claims.pid, status: 'active',
          }, { projection: { _id: 1 } }),
        ])
        if (!credential || !assignment) throw unauthorized('The device credential or assignment is no longer active.')
        request.principal = {
          kind: 'device', subjectId: claims.sub, deviceId: claims.did, patientId: claims.pid,
          credentialId: claims.cid, tenantId: claims.tid, roles: ['device'], scopes: claims.scopes || [],
        } satisfies DevicePrincipal
      } else {
        throw unauthorized()
      }
      next()
    } catch (error) {
      next(error)
    }
  }
}

export function getPrincipal(request: Request): Principal {
  if (!request.principal) throw unauthorized()
  return request.principal
}

export function requireHumanRole(...roles: string[]) {
  return (request: Request, _response: Response, next: NextFunction) => {
    try {
      const principal = getPrincipal(request)
      if (principal.kind !== 'human' || !roles.some((role) => principal.roles.includes(role))) throw forbidden()
      next()
    } catch (error) {
      next(error)
    }
  }
}

export async function authorizePatient(request: Request, patientId: string, scope: string) {
  const principal = getPrincipal(request)
  const db = await getDb()
  const patient = await db.collection<any>(collections.patients).findOne({
    _id: patientId,
    tenantId: principal.tenantId,
    status: { $ne: 'archived' },
  })
  if (!patient) throw forbidden()
  if (principal.kind === 'device') {
    if (principal.patientId !== patientId || !principal.scopes.includes(scope)) throw forbidden()
    return patient
  }
  if (principal.roles.includes('tenant_admin')) return patient
  const relationship = await db.collection<any>(collections.careRelationships).findOne({
    tenantId: principal.tenantId,
    patientId,
    userId: principal.userId,
    status: 'active',
    scopes: scope,
    $or: [{ validTo: null }, { validTo: { $gt: new Date() } }, { validTo: { $exists: false } }],
  })
  if (!relationship) throw forbidden()
  return patient
}
