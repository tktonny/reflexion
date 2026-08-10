// Typed API client for the reflexion-server /api/v1 admin surface. Handles bearer auth + single-flight
// 401 refresh. In dev the /api path is proxied by Vite; in prod set VITE_API_BASE_URL.

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
const ACCESS_KEY = 'reflexion.admin.accessToken'
const REFRESH_KEY = 'reflexion.admin.refreshToken'
const ACTOR_KEY = 'reflexion.admin.actor'

export type Actor = { userId: string; tenantId?: string; name?: string; email?: string; roles?: string[] }

export function getActor(): Actor | null {
  const raw = localStorage.getItem(ACTOR_KEY)
  if (!raw) return null
  try { return JSON.parse(raw) as Actor } catch { return null }
}
export function isAuthed(): boolean { return Boolean(localStorage.getItem(ACCESS_KEY)) }
function setTokens(accessToken: string, refreshToken: string) {
  localStorage.setItem(ACCESS_KEY, accessToken)
  localStorage.setItem(REFRESH_KEY, refreshToken)
}
export function clearSession() {
  localStorage.removeItem(ACCESS_KEY); localStorage.removeItem(REFRESH_KEY); localStorage.removeItem(ACTOR_KEY)
}

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) { super(message) }
}

function url(path: string) { return `${API_BASE}/api/v1${path}` }

async function unwrap<T>(response: Response): Promise<T> {
  const text = await response.text()
  const json = text ? JSON.parse(text) : {}
  if (!response.ok) {
    const err = json?.error || {}
    throw new ApiError(response.status, err.code || 'ERROR', err.message || response.statusText)
  }
  return (json?.data ?? json) as T
}

let refreshInFlight: Promise<boolean> | null = null
async function refreshOnce(): Promise<boolean> {
  if (refreshInFlight) return refreshInFlight
  refreshInFlight = (async () => {
    const refreshToken = localStorage.getItem(REFRESH_KEY)
    if (!refreshToken) return false
    try {
      const response = await fetch(url('/auth/session-refreshes'), {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ refreshToken }),
      })
      if (!response.ok) return false
      const data = (await response.json())?.data
      if (!data?.accessToken || !data?.refreshToken) return false
      setTokens(data.accessToken, data.refreshToken)
      return true
    } catch { return false } finally { refreshInFlight = null }
  })()
  return refreshInFlight
}

async function request<T>(method: string, path: string, body?: unknown, retry = true): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  const token = localStorage.getItem(ACCESS_KEY)
  if (token) headers.Authorization = `Bearer ${token}`
  if (method !== 'GET') headers['Idempotency-Key'] = crypto.randomUUID()
  const response = await fetch(url(path), { method, headers, body: body === undefined ? undefined : JSON.stringify(body) })
  if (response.status === 401 && retry && (await refreshOnce())) return request<T>(method, path, body, false)
  return unwrap<T>(response)
}

// --- Auth ---
export async function login(email: string, password: string): Promise<Actor> {
  const response = await fetch(url('/auth/sessions'), {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email, password }),
  })
  const data = await unwrap<{ accessToken: string; refreshToken: string; actor: Actor }>(response)
  setTokens(data.accessToken, data.refreshToken)
  const roles = data.actor.roles || []
  if (!roles.includes('operator') && !roles.includes('tenant_admin')) {
    clearSession()
    throw new ApiError(403, 'NOT_ADMIN', 'This account does not have admin access.')
  }
  localStorage.setItem(ACTOR_KEY, JSON.stringify(data.actor))
  return data.actor
}
export async function logout() {
  try { await request('DELETE', '/auth/sessions/current') } catch { /* ignore */ }
  clearSession()
}

// --- Types ---
export type Overview = { users: number; patients: number; openThreads: number; devices: number }
export type AdminUser = { userId: string; name: string; email: string; roles: string[]; status: string; createdAt: string | null }
export type Patient = { patientId: string; displayName: string; preferredLanguage: string; timezone: string; ageBand: string | null; status: string; version: number; createdAt: string | null; updatedAt: string | null }
export type Thread = { threadId: string; subject: string; status: 'open' | 'closed'; caregiverUserId: string; caregiverName: string; lastMessageAt: string; lastMessagePreview: string; adminUnread: boolean; caregiverUnread: boolean; createdAt: string }
export type Message = { messageId: string; threadId: string; authorType: 'caregiver' | 'admin'; authorId: string; authorName: string; body: string; createdAt: string }

// --- Endpoints ---
export const api = {
  overview: () => request<Overview>('GET', '/admin/overview'),
  users: () => request<AdminUser[]>('GET', '/admin/users'),
  patients: (q?: string) => request<Patient[]>('GET', `/admin/patients${q ? `?q=${encodeURIComponent(q)}` : ''}`),
  createPatient: (input: { displayName: string; preferredLanguage: string; timezone: string; ageBand?: string; caregiverUserId?: string }) =>
    request<Patient>('POST', '/admin/patients', input),
  updatePatient: (patientId: string, input: Partial<{ displayName: string; preferredLanguage: string; timezone: string; ageBand: string; status: string }>) =>
    request<Patient>('PATCH', `/admin/patients/${encodeURIComponent(patientId)}`, input),
  threads: (status?: 'open' | 'closed') => request<Thread[]>('GET', `/admin/support/threads${status ? `?status=${status}` : ''}`),
  thread: (threadId: string) => request<Thread & { messages: Message[] }>('GET', `/admin/support/threads/${encodeURIComponent(threadId)}`),
  reply: (threadId: string, body: string) => request<Message>('POST', `/admin/support/threads/${encodeURIComponent(threadId)}/messages`, { body }),
  setThreadStatus: (threadId: string, status: 'open' | 'closed') => request<Thread>('PATCH', `/admin/support/threads/${encodeURIComponent(threadId)}`, { status }),
}
