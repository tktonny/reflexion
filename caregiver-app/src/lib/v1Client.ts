import { useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query';
import {
  clearV1Session,
  getV1Session,
  hasV1Session,
  setV1Session,
  updateV1Tokens,
  type V1Actor,
  type V1Session,
} from './v1AuthSession';
import { getV1Url } from './apiUrl';
import type { V1PatientStatus } from './v1Status';

// Client for the authoritative v1 API (reflexion-implementation-baseline.md §4/§5). All v1 responses are
// enveloped as { data, meta }; errors as { error: { code, message }, meta }.
//
// The URL builder lives in ./apiUrl next to the legacy one, so the two mounting rules can be read together
// (and unit-tested without pulling React Native into the test process).

export class V1ApiError extends Error {
  status: number;
  code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = 'V1ApiError';
    this.status = status;
    this.code = code;
  }
}

type Envelope<T> = { data: T; meta?: { requestId?: string; nextCursor?: string | null } };

async function parseEnvelope<T>(response: Response, path: string): Promise<Envelope<T>> {
  const text = await response.text();
  let body: any = {};
  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    throw new V1ApiError(`Expected JSON from ${path} (received ${response.status}).`, response.status);
  }
  if (!response.ok) {
    const message: string =
      body?.error?.message || body?.error?.code || `Request failed with ${response.status}.`;
    throw new V1ApiError(String(message), response.status, body?.error?.code);
  }
  return body as Envelope<T>;
}

// Client-side idempotency keys (crypto.randomUUID with an RFC4122 v4 fallback for older runtimes).
export function generateIdempotencyKey(): string {
  const cryptoRef = (globalThis as { crypto?: { randomUUID?: () => string } }).crypto;
  if (cryptoRef?.randomUUID) {
    return cryptoRef.randomUUID();
  }
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (char) => {
    const rand = Math.floor(Math.random() * 16);
    const value = char === 'x' ? rand : (rand & 0x3) | 0x8;
    return value.toString(16);
  });
}

// Single-flight refresh so concurrent 401s do not race the server's refresh-token rotation.
let refreshInFlight: Promise<string | null> | null = null;

async function doRefresh(refreshToken: string): Promise<string | null> {
  try {
    const response = await fetch(getV1Url('/auth/session-refreshes'), {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ refreshToken }),
    });
    if (!response.ok) {
      await clearV1Session();
      return null;
    }
    const body = (await response.json()) as Envelope<{
      accessToken?: string;
      accessTokenExpiresAt?: string;
      refreshToken?: string;
      refreshTokenExpiresAt?: string;
    }>;
    const data = body?.data;
    if (!data?.accessToken) {
      await clearV1Session();
      return null;
    }
    await updateV1Tokens({
      accessToken: data.accessToken,
      accessTokenExpiresAt: data.accessTokenExpiresAt,
      refreshToken: data.refreshToken || refreshToken,
      refreshTokenExpiresAt: data.refreshTokenExpiresAt,
    });
    return data.accessToken;
  } catch {
    return null;
  }
}

async function refreshAccessToken(): Promise<string | null> {
  const session = getV1Session();
  if (!session?.refreshToken) {
    return null;
  }
  if (!refreshInFlight) {
    refreshInFlight = doRefresh(session.refreshToken).finally(() => {
      refreshInFlight = null;
    });
  }
  return refreshInFlight;
}

function buildHeaders(accessToken: string | undefined, extra?: Record<string, string>): Headers {
  const headers = new Headers(extra);
  headers.set('content-type', 'application/json');
  headers.set('accept', 'application/json');
  if (accessToken) {
    headers.set('Authorization', `Bearer ${accessToken}`);
  }
  return headers;
}

// Authenticated request: attaches Bearer, and on 401 refreshes once then retries with the same body
// and headers (so an Idempotency-Key survives the retry).
async function v1Fetch<T>(
  path: string,
  init: { method?: string; body?: unknown; headers?: Record<string, string> } = {},
): Promise<Envelope<T>> {
  const session = getV1Session();
  const baseInit: RequestInit = {
    method: init.method || 'GET',
    headers: buildHeaders(session?.accessToken, init.headers),
  };
  if (init.body !== undefined) {
    baseInit.body = JSON.stringify(init.body);
  }

  const response = await fetch(getV1Url(path), baseInit);
  if (response.status === 401 && getV1Session()?.refreshToken) {
    const nextToken = await refreshAccessToken();
    if (nextToken) {
      const retryInit: RequestInit = {
        method: init.method || 'GET',
        headers: buildHeaders(nextToken, init.headers),
      };
      if (init.body !== undefined) {
        retryInit.body = JSON.stringify(init.body);
      }
      const retry = await fetch(getV1Url(path), retryInit);
      return parseEnvelope<T>(retry, path);
    }
  }
  return parseEnvelope<T>(response, path);
}

export async function v1Get<T>(path: string): Promise<T> {
  return (await v1Fetch<T>(path, { method: 'GET' })).data;
}

export async function v1Post<T>(
  path: string,
  body?: unknown,
  options?: { idempotencyKey?: string },
): Promise<T> {
  const headers = options?.idempotencyKey ? { 'Idempotency-Key': options.idempotencyKey } : undefined;
  return (await v1Fetch<T>(path, { method: 'POST', body, headers })).data;
}

// ── Auth ──────────────────────────────────────────────────────────────────

type LoginResponse = {
  accessToken: string;
  accessTokenExpiresAt?: string;
  refreshToken: string;
  refreshTokenExpiresAt?: string;
  actor: V1Actor;
};

export async function v1Login(email: string, password: string): Promise<V1Session> {
  const envelope = await parseEnvelope<LoginResponse>(
    await fetch(getV1Url('/auth/sessions'), {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'application/json' },
      body: JSON.stringify({ email: email.trim().toLowerCase(), password }),
    }),
    '/auth/sessions',
  );
  const data = envelope.data;
  const session: V1Session = {
    accessToken: data.accessToken,
    refreshToken: data.refreshToken,
    accessTokenExpiresAt: data.accessTokenExpiresAt,
    refreshTokenExpiresAt: data.refreshTokenExpiresAt,
    actor: {
      userId: data.actor?.userId || '',
      tenantId: data.actor?.tenantId || '',
      name: data.actor?.name || '',
      email: data.actor?.email || email.trim().toLowerCase(),
      roles: Array.isArray(data.actor?.roles) ? data.actor.roles : [],
    },
  };
  await setV1Session(session);
  return session;
}

export async function v1Logout(): Promise<void> {
  // Best-effort server-side revoke; always clear local tokens regardless of outcome.
  try {
    if (hasV1Session()) {
      await v1Fetch('/auth/sessions/current', { method: 'DELETE' });
    }
  } catch {
    // ignore — local clear below is what matters
  }
  await clearV1Session();
}

// ── Patients / status / caregiver actions ───────────────────────────────────

export type V1Patient = {
  patientId: string;
  displayName: string;
  preferredLanguage: string;
  timezone: string;
  ageBand: string | null;
  status: string;
  version: number;
};

export async function listPatientsV1(limit = 100): Promise<{ data: V1Patient[]; nextCursor: string | null }> {
  const envelope = await v1Fetch<V1Patient[]>(`/patients?limit=${encodeURIComponent(String(limit))}`, {
    method: 'GET',
  });
  return { data: Array.isArray(envelope.data) ? envelope.data : [], nextCursor: envelope.meta?.nextCursor ?? null };
}

export async function getPatientStatusV1(patientId: string): Promise<V1PatientStatus> {
  return v1Get<V1PatientStatus>(`/patients/${encodeURIComponent(patientId)}/status`);
}

export async function createManualFlagV1(
  patientId: string,
  severity: 'worth_checking' | 'needs_attention',
  reason: string,
): Promise<{ manualFlagId: string }> {
  return v1Post(
    `/patients/${encodeURIComponent(patientId)}/manual-flags`,
    { severity, reason },
    { idempotencyKey: generateIdempotencyKey() },
  );
}

export async function createAwayPeriodV1(
  patientId: string,
  input: { startsOn: string; endsOn: string; timezone: string; reason?: string },
): Promise<{ awayPeriodId: string }> {
  return v1Post(
    `/patients/${encodeURIComponent(patientId)}/away-periods`,
    input,
    { idempotencyKey: generateIdempotencyKey() },
  );
}

// ── Notifications (the caregiver alert feed) ────────────────────────────────

export type V1NotificationType =
  | 'completion' | 'missed_7pm' | 'red_missed_streak' | 'technical_issue' | 'late_completion'
  | 'worth_checking' | 'needs_attention';

export type V1Notification = {
  notificationId: string;
  patientId: string;
  type: V1NotificationType | string;
  state: 'unread' | 'read';
  title: string;
  body: string;
  source?: { type: string; id: string } | null;
  localDate: string | null;
  createdAt: string;
  readAt: string | null;
};

export async function listNotificationsV1(options: { limit?: number; cursor?: string | null } = {}): Promise<{
  data: V1Notification[];
  nextCursor: string | null;
}> {
  const params = new URLSearchParams({ limit: String(options.limit || 20) });
  if (options.cursor) params.set('cursor', options.cursor);
  const envelope = await v1Fetch<V1Notification[]>(`/notifications?${params.toString()}`, { method: 'GET' });
  return {
    data: Array.isArray(envelope.data) ? envelope.data : [],
    nextCursor: envelope.meta?.nextCursor ?? null,
  };
}

export async function markNotificationReadV1(notificationId: string): Promise<V1Notification> {
  return v1Post<V1Notification>(`/notifications/${encodeURIComponent(notificationId)}/read`);
}

/**
 * Registers this phone to receive push notifications. Identity comes from the bearer token, so the
 * server needs no caregiver id. The Expo token is the device key, making repeat calls a safe upsert —
 * hence no Idempotency-Key.
 */
export async function registerNotificationDeviceV1(input: {
  expoPushToken: string;
  platform: 'ios' | 'android' | 'web' | 'unknown';
  appVersion?: string;
}): Promise<{ deviceId: string; state: string }> {
  return v1Post('/notification-devices', input);
}

// ── Support threads (the in-app "Support" conversation) ─────────────────────

export type V1SupportThread = {
  threadId: string;
  subject: string;
  status: 'open' | 'closed' | string;
  lastMessageAt: string;
  lastMessagePreview: string;
  caregiverUnread: boolean;
  createdAt: string;
};

export async function listSupportThreadsV1(): Promise<V1SupportThread[]> {
  const threads = await v1Get<V1SupportThread[]>('/support/threads');
  return Array.isArray(threads) ? threads : [];
}

export async function openSupportThreadV1(subject: string, body: string): Promise<V1SupportThread> {
  return v1Post<V1SupportThread>('/support/threads', { subject, body }, { idempotencyKey: generateIdempotencyKey() });
}

export async function postSupportMessageV1(threadId: string, body: string): Promise<{ messageId: string }> {
  return v1Post(`/support/threads/${encodeURIComponent(threadId)}/messages`, { body });
}

// ── react-query hooks ───────────────────────────────────────────────────────

export const PATIENT_STATUS_QUERY_ROOT = 'patientStatusV1';
export const patientStatusQueryKey = (patientId: string | null | undefined) => [PATIENT_STATUS_QUERY_ROOT, patientId];

// The app-wide default is staleTime: Infinity (queryClient.ts) with explicit per-screen refetching. That
// default is wrong for status: it is the one value the mirror changes behind the app's back, so a cached
// answer goes stale on its own within minutes. A bounded staleTime lets a remount pick up a new check-in
// instead of serving the same dot for the whole process lifetime.
const STATUS_STALE_TIME_MS = 60_000;

/** Marks every patient's status stale so the visible screens refetch. Call from useFocusEffect. */
export function invalidatePatientStatuses(queryClient: QueryClient) {
  return queryClient.invalidateQueries({ queryKey: [PATIENT_STATUS_QUERY_ROOT] });
}

export function usePatientStatusV1(patientId: string | null | undefined) {
  return useQuery({
    queryKey: patientStatusQueryKey(patientId),
    queryFn: () => getPatientStatusV1(patientId as string),
    enabled: Boolean(patientId) && hasV1Session(),
    staleTime: STATUS_STALE_TIME_MS,
    refetchOnMount: true,
  });
}

/** Per-id outcome from `GET /patient-statuses`. See the route comment for why each id keeps its own. */
export type V1PatientStatusOutcome = {
  patientId: string;
  outcome: 'ok' | 'unavailable' | 'failed';
  status: V1PatientStatus | null;
};

export async function getPatientStatusesV1(patientIds: string[]): Promise<V1PatientStatusOutcome[]> {
  if (!patientIds.length) return [];
  return v1Get<V1PatientStatusOutcome[]>(`/patient-statuses?ids=${encodeURIComponent(patientIds.join(','))}`);
}

/** What the dashboard needs to know about one loved one's status, and why it is missing when it is. */
export type PatientStatusSlot = {
  data: V1PatientStatus | undefined;
  isLoading: boolean;
  /** The request itself failed, or the server could not compute this one. Retrying may help. */
  isError: boolean;
  /** No v1 session — a normal state, since v1 login is best-effort. Retrying will NOT help. */
  isSignedOut: boolean;
  /** No monitoring record yet, typically a loved one whose mirror has not been paired. */
  isUnavailable: boolean;
};

/**
 * Every loved one's status in ONE request.
 *
 * The dashboard used to issue a status request per patient on top of the list request. Each round trip is a
 * cross-region hop from Singapore, so the batch matters more here than the request count suggests.
 *
 * Successful rows are also written into the per-patient cache entry that usePatientStatusV1 reads, so
 * opening a profile shows the status the dashboard already has instead of starting from "Checking in…", and
 * one invalidatePatientStatuses() call still refreshes both.
 *
 * Each slot reports WHY a status is missing. Collapsing that into a bare `undefined` is what let the
 * dashboard show "0 needs attention" for a patient whose status simply had not been fetched.
 */
export function usePatientStatusesV1(patientIds: string[]): PatientStatusSlot[] {
  const queryClient = useQueryClient();
  const signedOut = !hasV1Session();
  const enabled = !signedOut && patientIds.length > 0;
  // Sorted so re-ordering the loved-one list reuses the same cache entry instead of refetching.
  const key = [...patientIds].sort().join(',');

  const query = useQuery({
    enabled,
    queryKey: [PATIENT_STATUS_QUERY_ROOT, 'batch', key],
    queryFn: async () => {
      const outcomes = await getPatientStatusesV1(patientIds);
      for (const outcome of outcomes) {
        // Only a real status seeds the per-patient cache. Writing null would make the profile screen
        // believe it had fetched and suppress its own "sign in again" / retry affordances.
        if (outcome.outcome === 'ok' && outcome.status) {
          queryClient.setQueryData(patientStatusQueryKey(outcome.patientId), outcome.status);
        }
      }
      return outcomes;
    },
    staleTime: STATUS_STALE_TIME_MS,
    refetchOnMount: true,
  });

  const byId = new Map((query.data || []).map((outcome) => [outcome.patientId, outcome]));

  return patientIds.map((patientId) => {
    const outcome = byId.get(patientId);
    return {
      data: outcome?.outcome === 'ok' ? outcome.status ?? undefined : undefined,
      isLoading: enabled && query.isLoading,
      // A patient the server could not compute is an error for that patient even though the request was 200.
      isError: Boolean(query.isError) || outcome?.outcome === 'failed',
      isSignedOut: signedOut,
      isUnavailable: outcome?.outcome === 'unavailable',
    };
  });
}
