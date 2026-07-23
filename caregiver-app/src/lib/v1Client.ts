import { useQueries, useQuery } from '@tanstack/react-query';
import {
  clearV1Session,
  getV1Session,
  hasV1Session,
  setV1Session,
  updateV1Tokens,
  type V1Actor,
  type V1Session,
} from './v1AuthSession';
import type { V1PatientStatus } from './v1Status';

// Client for the authoritative v1 API (reflexion-implementation-baseline.md §4/§5).
// Base URL env has no /api segment; v1 is mounted at /api/v1. getApiUrl (legacy) strips a leading
// /api, so it cannot build v1 URLs — hence this dedicated builder. All v1 responses are enveloped
// as { data, meta }; errors as { error: { code, message }, meta }.

export function getV1Url(path: string): string {
  const baseUrl = process.env.EXPO_PUBLIC_CAREGIVER_APP_BACKEND_URL?.trim();
  if (!baseUrl) {
    throw new Error('EXPO_PUBLIC_CAREGIVER_APP_BACKEND_URL is not set');
  }
  const normalizedBase = baseUrl.replace(/\/+$/, '');
  const apiBase = /\/api$/.test(normalizedBase) ? normalizedBase : `${normalizedBase}/api`;
  const normalizedPath = path.replace(/^\/?/, '/');
  return `${apiBase}/v1${normalizedPath}`;
}

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

// ── react-query hooks ───────────────────────────────────────────────────────

export const patientStatusQueryKey = (patientId: string | null | undefined) => ['patientStatusV1', patientId];

export function usePatientStatusV1(patientId: string | null | undefined) {
  return useQuery({
    queryKey: patientStatusQueryKey(patientId),
    queryFn: () => getPatientStatusV1(patientId as string),
    enabled: Boolean(patientId) && hasV1Session(),
  });
}

export function usePatientStatusesV1(patientIds: string[]) {
  const enabled = hasV1Session();
  return useQueries({
    queries: patientIds.map((patientId) => ({
      queryKey: patientStatusQueryKey(patientId),
      queryFn: () => getPatientStatusV1(patientId),
      enabled: Boolean(patientId) && enabled,
    })),
  });
}
