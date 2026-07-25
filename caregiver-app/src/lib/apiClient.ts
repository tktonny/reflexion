import { getApiUrl } from './apiUrl';

export async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(getApiUrl(path));
  return readJsonResponse<T>(response, path);
}

export async function apiSend<T>(path: string, init: RequestInit): Promise<T> {
  const response = await fetch(getApiUrl(path), {
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

  constructor(message: string, status: number) {
    super(message);
    this.name = 'LegacyApiError';
    this.status = status;
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
