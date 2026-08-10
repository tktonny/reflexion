import { getApiUrl } from './apiUrl';

export async function apiGet<T>(path: string): Promise<T> {
  const response = await fetchLegacy(path, getApiUrl(path));
  return readJsonResponse<T>(response, path);
}

export async function apiSend<T>(path: string, init: RequestInit): Promise<T> {
  const response = await fetchLegacy(path, getApiUrl(path), {
    ...init,
    headers: {
      'content-type': 'application/json',
      ...(init.headers || {}),
    },
  });
  return readJsonResponse<T>(response, path);
}

/**
 * Carries the HTTP status alongside the message so callers can choose caregiver-facing wording by kind of
 * failure instead of putting the raw server text on screen. The message itself is for logs.
 */
export class LegacyApiError extends Error {
  status: number;
  readonly wireMessage: string;

  constructor(message: string, status: number) {
    super(legacyFacingApiMessage(status));
    this.name = 'LegacyApiError';
    this.status = status;
    this.wireMessage = message;
  }
}

function legacyFacingApiMessage(status: number): string {
  if (status === 401) return 'Your session has expired. Sign in again.';
  if (status === 403) return 'You do not have permission to make this change.';
  if (status === 404) return 'We could not find that item. Refresh and try again.';
  if (status === 409) return 'This changed elsewhere. Refresh and try again.';
  if (status === 429) return 'Too many requests just now. Wait a moment and try again.';
  if (status >= 500) return 'We could not connect to Reflexion right now. Check your connection and try again.';
  if (status === 400) return 'Check the details and try again.';
  return 'We could not connect to Reflexion right now. Check your connection and try again.';
}

async function fetchLegacy(path: string, url: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch (cause) {
    const wireMessage = cause instanceof Error ? cause.message : String(cause);
    console.warn(`[legacy] ${path} request failed`, wireMessage);
    throw new LegacyApiError(wireMessage, 0);
  }
}

async function readJsonResponse<T>(response: Response, path: string): Promise<T> {
  const text = await response.text();
  let body: unknown = {};

  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    const preview = text.replace(/\s+/g, ' ').trim().slice(0, 120);
    throw new LegacyApiError(`Expected JSON from ${path}, received ${response.status}: ${preview}`, response.status);
  }

  if (!response.ok) {
    const error = body && typeof body === 'object' && 'error' in body
      ? String((body as { error?: unknown }).error || '')
      : '';
    throw new LegacyApiError(error || `Request failed with ${response.status}`, response.status);
  }

  return body as T;
}
