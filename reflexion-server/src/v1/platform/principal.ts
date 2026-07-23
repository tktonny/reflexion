export type HumanPrincipal = {
  kind: 'human'
  subjectId: string
  userId: string
  tenantId: string
  sessionId: string
  roles: string[]
  scopes: string[]
}

export type DevicePrincipal = {
  kind: 'device'
  subjectId: string
  deviceId: string
  credentialId: string
  tenantId: string
  patientId: string
  roles: ['device']
  scopes: string[]
}

export type Principal = HumanPrincipal | DevicePrincipal

declare global {
  namespace Express {
    interface Request {
      requestId: string
      principal?: Principal
    }
  }
}

export {}
