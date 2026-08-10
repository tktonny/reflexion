import AsyncStorage from '@react-native-async-storage/async-storage'

/**
 * These are the states the setup flow can actually prove. A caller may only advance to a later state
 * after the preceding check has succeeded; the UI deliberately does not infer internet or pairing from
 * a Wi-Fi icon alone.
 */
export const PAIRING_STAGES = [
  'WIFI_DISCONNECTED',
  'WIFI_CONNECTED',
  'INTERNET_UNAVAILABLE',
  'INTERNET_AVAILABLE',
  'SERVICE_UNREACHABLE',
  'SERVICE_REACHABLE',
  'AUTHENTICATION_READY',
  'DEVICE_IDENTITY_FAILED',
  'PAIRING_SESSION_CREATING',
  'PAIRING_SESSION_FAILED',
  'READY_TO_PAIR',
  'PAIRING_CODE_EXPIRED',
  'PAIRING_CODE_INVALID',
  'PAIRING_CODE_USED',
  'DEVICE_ALREADY_CLAIMED',
  'CLAIM_IN_PROGRESS',
  'ASSIGNMENT_FAILED',
  'CREDENTIAL_ISSUANCE_FAILED',
  'CREDENTIAL_STORAGE_FAILED',
  'AUTHENTICATION_FAILED',
  'PAIRED',
] as const

export type PairingDiagnosticStage = typeof PAIRING_STAGES[number]

export type PairingDiagnosticReasonCode =
  | 'NETWORK_UNAVAILABLE'
  | 'SERVICE_UNREACHABLE'
  | 'API_BASE_NOT_CONFIGURED'
  | 'TIMEOUT'
  | 'DNS_FAILURE'
  | 'TLS_FAILURE'
  | 'HEALTH_CHECK_FAILED'
  | 'HTTP_ERROR'
  | 'ROUTE_NOT_FOUND'
  | 'UNAUTHORIZED'
  | 'DEVICE_RECOVERY_REQUIRED'
  | 'PAIRING_CODE_INVALID'
  | 'PAIRING_CODE_EXPIRED'
  | 'PAIRING_CODE_USED'
  | 'PAIRING_ATTEMPTS_EXCEEDED'
  | 'DEVICE_ALREADY_CLAIMED'
  | 'CREDENTIAL_MISSING'
  | 'CREDENTIAL_REJECTED'
  | 'EXCHANGE_TICKET_INVALID'
  | 'ASSIGNMENT_NOT_ACTIVE'
  | 'ASSIGNMENT_FAILED'
  | 'SECURE_STORAGE_FAILURE'
  | 'PAIRING_CODE_UNAVAILABLE'
  | 'PAIRING_FAILED'
  | 'UNKNOWN'

export type PairingDiagnosticEvent = {
  stage: PairingDiagnosticStage
  ok: boolean
  at: string
  retryable: boolean
  userMessage: string
  supportCode: string
  reasonCode?: PairingDiagnosticReasonCode
  httpStatus?: number
  requestId?: string
}

export type PairingDiagnosticOptions = {
  retryable?: boolean
  userMessage?: string
  supportCode?: string
  httpStatus?: number
  requestId?: string
}

export type PairingFailure = {
  stage: PairingDiagnosticStage
  reasonCode: PairingDiagnosticReasonCode
  retryable: boolean
  userMessage: string
  httpStatus?: number
  requestId?: string
}

const USER_MESSAGES: Record<PairingDiagnosticStage, string> = {
  WIFI_DISCONNECTED: 'Your mirror is not connected to Wi-Fi.',
  WIFI_CONNECTED: 'Connected to Wi-Fi.',
  INTERNET_UNAVAILABLE: 'Connected to Wi-Fi, but the internet is unavailable.',
  INTERNET_AVAILABLE: 'Internet available.',
  SERVICE_UNREACHABLE: 'The mirror is online, but Reflexion cannot be reached.',
  SERVICE_REACHABLE: 'Reflexion service reached.',
  AUTHENTICATION_READY: 'Device authentication ready.',
  DEVICE_IDENTITY_FAILED: 'This mirror could not complete its initial setup. Restart it and try again.',
  PAIRING_SESSION_CREATING: 'Preparing pairing code.',
  PAIRING_SESSION_FAILED: 'Reflexion could not prepare this mirror for pairing.',
  READY_TO_PAIR: 'Ready to pair.',
  PAIRING_CODE_EXPIRED: 'This pairing code has expired. A new code is being created.',
  PAIRING_CODE_INVALID: 'That pairing code is not valid. Check the code and try again.',
  PAIRING_CODE_USED: 'This pairing code has already been used. Generate a new code.',
  DEVICE_ALREADY_CLAIMED: 'This mirror is already connected to another household.',
  CLAIM_IN_PROGRESS: 'Completing secure setup.',
  ASSIGNMENT_FAILED: 'The mirror was found, but it could not be assigned to the selected loved one.',
  CREDENTIAL_ISSUANCE_FAILED: 'Pairing was started, but secure device setup could not be completed.',
  CREDENTIAL_STORAGE_FAILED: 'The mirror could not save its secure setup. Restart the mirror and try again.',
  AUTHENTICATION_FAILED: 'Reflexion could not verify this mirror.',
  PAIRED: 'Ready to use.',
}

const DEFAULT_RETRYABLE = new Set<PairingDiagnosticStage>([
  'WIFI_DISCONNECTED',
  'INTERNET_UNAVAILABLE',
  'SERVICE_UNREACHABLE',
  'DEVICE_IDENTITY_FAILED',
  'PAIRING_SESSION_FAILED',
  'PAIRING_CODE_EXPIRED',
  'PAIRING_CODE_USED',
  'CLAIM_IN_PROGRESS',
  'ASSIGNMENT_FAILED',
  'CREDENTIAL_ISSUANCE_FAILED',
  'CREDENTIAL_STORAGE_FAILED',
  'AUTHENTICATION_FAILED',
])

const STORAGE_KEY = 'reflexion:pairingDiagnostics'
const MAX_EVENTS = 32

/**
 * Diagnostics are deliberately non-sensitive. Device IDs, pairing tokens, codes and credentials stay
 * out of both the event and console; once the device has a credential, the next authenticated heartbeat
 * forwards this queue to the backend telemetry store.
 */
export async function recordPairingDiagnostic(
  stage: PairingDiagnosticStage,
  ok: boolean,
  reasonCode?: PairingDiagnosticReasonCode,
  options: PairingDiagnosticOptions = {},
): Promise<PairingDiagnosticEvent> {
  const at = new Date().toISOString()
  const event: PairingDiagnosticEvent = {
    stage,
    ok,
    at,
    retryable: options.retryable ?? DEFAULT_RETRYABLE.has(stage),
    userMessage: options.userMessage || USER_MESSAGES[stage],
    supportCode: options.supportCode || makeSupportCode(stage, at, options.requestId),
    ...(reasonCode ? { reasonCode } : {}),
    ...(typeof options.httpStatus === 'number' ? { httpStatus: options.httpStatus } : {}),
    ...(options.requestId ? { requestId: options.requestId } : {}),
  }
  try {
    const existing = await readPairingDiagnostics()
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify([...existing, event].slice(-MAX_EVENTS)))
  } catch {
    // Diagnostics must never block pairing or conversation startup.
  }
  console.info(`[pairing-diagnostic] ${JSON.stringify(event)}`)
  return event
}

export async function getPairingDiagnostics(): Promise<PairingDiagnosticEvent[]> {
  return readPairingDiagnostics()
}

export async function clearPairingDiagnostics() {
  try { await AsyncStorage.removeItem(STORAGE_KEY) } catch { /* best effort */ }
}

export function pairingFailureFromError(error: unknown, fallbackStage: PairingDiagnosticStage = 'PAIRING_SESSION_FAILED'): PairingFailure {
  const candidate = error as { code?: unknown; status?: unknown; retryable?: unknown; requestId?: unknown }
  const code = typeof candidate?.code === 'string'
    ? candidate.code
    : error instanceof Error ? error.message : ''
  const status = typeof candidate?.status === 'number' ? candidate.status : undefined
  const requestId = typeof candidate?.requestId === 'string' ? candidate.requestId : undefined
  let stage = fallbackStage
  let reasonCode: PairingDiagnosticReasonCode = 'PAIRING_FAILED'

  switch (code) {
    case 'DEVICE_ALREADY_CLAIMED': stage = 'DEVICE_ALREADY_CLAIMED'; reasonCode = 'DEVICE_ALREADY_CLAIMED'; break
    case 'DEVICE_RECOVERY_REQUIRED': stage = 'DEVICE_IDENTITY_FAILED'; reasonCode = code; break
    case 'PAIRING_CODE_INVALID': stage = 'PAIRING_CODE_INVALID'; reasonCode = code; break
    case 'PAIRING_CODE_EXPIRED': stage = 'PAIRING_CODE_EXPIRED'; reasonCode = code; break
    case 'PAIRING_CODE_USED': stage = 'PAIRING_CODE_USED'; reasonCode = code; break
    case 'PAIRING_ATTEMPTS_EXCEEDED': stage = 'PAIRING_SESSION_FAILED'; reasonCode = code; break
    case 'EXCHANGE_TICKET_INVALID': stage = 'CREDENTIAL_ISSUANCE_FAILED'; reasonCode = code; break
    case 'ASSIGNMENT_NOT_ACTIVE': stage = 'ASSIGNMENT_FAILED'; reasonCode = code; break
    case 'CREDENTIAL_STORAGE_FAILED': stage = 'CREDENTIAL_STORAGE_FAILED'; reasonCode = 'SECURE_STORAGE_FAILURE'; break
    case 'SERVICE_UNREACHABLE': stage = 'SERVICE_UNREACHABLE'; reasonCode = code; break
    case 'TIMEOUT': stage = 'SERVICE_UNREACHABLE'; reasonCode = code; break
    case 'ROUTE_NOT_FOUND': stage = 'PAIRING_SESSION_FAILED'; reasonCode = code; break
    case 'UNAUTHORIZED': stage = 'PAIRING_SESSION_FAILED'; reasonCode = code; break
    case 'DEVICE_IDENTITY_FAILED': stage = 'DEVICE_IDENTITY_FAILED'; reasonCode = 'SECURE_STORAGE_FAILURE'; break
    case 'SECURE_STORAGE_FAILURE': stage = 'CREDENTIAL_STORAGE_FAILED'; reasonCode = code; break
    default:
      if (status === 404) reasonCode = 'ROUTE_NOT_FOUND'
      else if (status === 401) reasonCode = 'UNAUTHORIZED'
      else if (status && status >= 500) reasonCode = 'HTTP_ERROR'
      else if (error instanceof TypeError) reasonCode = 'NETWORK_UNAVAILABLE'
  }

  const retryable = typeof candidate?.retryable === 'boolean'
    ? candidate.retryable
    : DEFAULT_RETRYABLE.has(stage)
  return {
    stage,
    reasonCode,
    retryable,
    userMessage: USER_MESSAGES[stage],
    ...(status !== undefined ? { httpStatus: status } : {}),
    ...(requestId ? { requestId } : {}),
  }
}

export function userMessageForPairingStage(stage: PairingDiagnosticStage) {
  return USER_MESSAGES[stage]
}

function makeSupportCode(stage: PairingDiagnosticStage, at: string, requestId?: string) {
  let hash = 2166136261
  for (const char of `${stage}:${at}:${requestId || ''}`) {
    hash ^= char.charCodeAt(0)
    hash = Math.imul(hash, 16777619)
  }
  return `P${(hash >>> 0).toString(36).toUpperCase().slice(0, 6).padStart(6, '0')}`
}

async function readPairingDiagnostics(): Promise<PairingDiagnosticEvent[]> {
  let raw: string | null = null
  try { raw = await AsyncStorage.getItem(STORAGE_KEY) } catch { return [] }
  if (!raw) return []
  try {
    const parsed = JSON.parse(raw) as unknown
    if (!Array.isArray(parsed)) return []
    return parsed.filter(isPairingDiagnosticEvent).slice(-MAX_EVENTS)
  } catch {
    return []
  }
}

function isPairingDiagnosticEvent(value: unknown): value is PairingDiagnosticEvent {
  if (!value || typeof value !== 'object') return false
  const item = value as Record<string, unknown>
  return typeof item.stage === 'string'
    && (PAIRING_STAGES as readonly string[]).includes(item.stage)
    && typeof item.ok === 'boolean'
    && typeof item.at === 'string'
    && typeof item.retryable === 'boolean'
    && typeof item.userMessage === 'string'
    && typeof item.supportCode === 'string'
    && (item.reasonCode === undefined || typeof item.reasonCode === 'string')
    && (item.httpStatus === undefined || typeof item.httpStatus === 'number')
    && (item.requestId === undefined || typeof item.requestId === 'string')
}
