import * as FileSystem from 'expo-file-system/legacy';
import { Platform } from 'react-native';

// v1 human JWT session (see reflexion-implementation-baseline.md §5). Stored ALONGSIDE the legacy
// AuthSession — this holds the tokens the /api/v1 status/flag/away routes require. Storage mirrors
// authSession.ts exactly: localStorage on web, an app-document file on native, with a memory cache.

export type V1Actor = {
  userId: string;
  tenantId: string;
  name?: string;
  email?: string;
  roles?: string[];
};

export type V1Session = {
  accessToken: string;
  refreshToken: string;
  accessTokenExpiresAt?: string;
  refreshTokenExpiresAt?: string;
  actor: V1Actor;
};

const V1_SESSION_KEY = 'reflexion.v1Session';
const V1_SESSION_FILE = FileSystem.documentDirectory
  ? `${FileSystem.documentDirectory}reflexion-v1-session.json`
  : '';

let memorySession: V1Session | null = null;

function getStorage(): Storage | null {
  if (typeof globalThis === 'undefined') {
    return null;
  }
  return (globalThis as typeof globalThis & { localStorage?: Storage }).localStorage ?? null;
}

function parseSession(raw: string | null | undefined): V1Session | null {
  if (!raw) {
    return null;
  }
  try {
    const parsed = JSON.parse(raw) as Partial<V1Session>;
    if (!parsed.accessToken || !parsed.refreshToken || !parsed.actor?.userId) {
      return null;
    }
    return {
      accessToken: parsed.accessToken,
      refreshToken: parsed.refreshToken,
      accessTokenExpiresAt: parsed.accessTokenExpiresAt,
      refreshTokenExpiresAt: parsed.refreshTokenExpiresAt,
      actor: {
        userId: parsed.actor.userId,
        tenantId: parsed.actor.tenantId || '',
        name: parsed.actor.name || '',
        email: parsed.actor.email || '',
        roles: Array.isArray(parsed.actor.roles) ? parsed.actor.roles : [],
      },
    };
  } catch {
    return null;
  }
}

export function getV1Session(): V1Session | null {
  if (memorySession) {
    return memorySession;
  }
  const storage = getStorage();
  if (!storage) {
    return memorySession;
  }
  memorySession = parseSession(storage.getItem(V1_SESSION_KEY));
  return memorySession;
}

export function hasV1Session(): boolean {
  return Boolean(getV1Session()?.accessToken);
}

// Native cold-start hydration (mirrors loadStoredAuthSession). Call once on app boot before any v1 read.
export async function loadV1Session(): Promise<V1Session | null> {
  const existing = getV1Session();
  if (existing || Platform.OS === 'web' || !V1_SESSION_FILE) {
    return existing;
  }
  try {
    const raw = await FileSystem.readAsStringAsync(V1_SESSION_FILE);
    memorySession = parseSession(raw);
    return memorySession;
  } catch {
    return null;
  }
}

export async function setV1Session(session: V1Session): Promise<void> {
  memorySession = session;
  const serialized = JSON.stringify(session);
  const storage = getStorage();
  if (storage) {
    storage.setItem(V1_SESSION_KEY, serialized);
    return;
  }
  if (Platform.OS !== 'web' && V1_SESSION_FILE) {
    await FileSystem.writeAsStringAsync(V1_SESSION_FILE, serialized);
  }
}

// Persist rotated tokens after a refresh, keeping the existing actor.
export async function updateV1Tokens(tokens: {
  accessToken: string;
  refreshToken: string;
  accessTokenExpiresAt?: string;
  refreshTokenExpiresAt?: string;
}): Promise<void> {
  const current = getV1Session();
  if (!current) {
    return;
  }
  await setV1Session({ ...current, ...tokens });
}

export async function clearV1Session(): Promise<void> {
  memorySession = null;
  const storage = getStorage();
  if (storage) {
    storage.removeItem(V1_SESSION_KEY);
    return;
  }
  if (Platform.OS !== 'web' && V1_SESSION_FILE) {
    await FileSystem.deleteAsync(V1_SESSION_FILE, { idempotent: true });
  }
}
