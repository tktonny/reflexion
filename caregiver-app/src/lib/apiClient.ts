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

async function readJsonResponse<T>(response: Response, path: string): Promise<T> {
  const text = await response.text();
  let body: unknown = {};

  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    const preview = text.replace(/\s+/g, ' ').trim().slice(0, 120);
    throw new Error(`Expected JSON from ${path}, received ${response.status}: ${preview}`);
  }

  if (!response.ok) {
    const error = body && typeof body === 'object' && 'error' in body
      ? String((body as { error?: unknown }).error || '')
      : '';
    throw new Error(error || `Request failed with ${response.status}`);
  }

  return body as T;
}
