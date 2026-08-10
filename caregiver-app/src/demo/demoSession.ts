import type { AuthSession } from '../lib/authSession';
import type { V1Session } from '../lib/v1AuthSession';

const DEMO_V1_SESSION_KEY = 'reflexion.demo.v1Session';
const DEMO_AUTH_SESSION_KEY = 'reflexion.demo.authSession';

const DEMO_V1_SESSION: V1Session = {
  accessToken: 'demo-access-token',
  refreshToken: 'demo-refresh-token',
  accessTokenExpiresAt: '2099-01-01T00:00:00.000Z',
  refreshTokenExpiresAt: '2099-01-01T00:00:00.000Z',
  actor: {
    userId: 'demo-caregiver',
    tenantId: 'demo-tenant',
    name: 'Chloe',
    email: 'demo@reflexion.local',
    roles: ['caregiver'],
  },
};

const DEMO_AUTH_SESSION: AuthSession = {
  userId: 'demo-caregiver',
  name: 'Chloe',
  email: 'demo@reflexion.local',
};

let memoryV1Session: V1Session | null = null;
let memoryAuthSession: AuthSession | null = null;

function storage(): Storage | null {
  if (typeof globalThis === 'undefined') return null;
  return (globalThis as typeof globalThis & { localStorage?: Storage }).localStorage ?? null;
}

function read<T>(key: string): T | null {
  try {
    const raw = storage()?.getItem(key);
    return raw ? JSON.parse(raw) as T : null;
  } catch {
    return null;
  }
}

function write<T>(key: string, value: T): void {
  try {
    storage()?.setItem(key, JSON.stringify(value));
  } catch {
    // Browser storage can be disabled; the in-memory fixture remains usable.
  }
}

export function getDemoV1Session(): V1Session | null {
  return memoryV1Session || read<V1Session>(DEMO_V1_SESSION_KEY);
}

export function getDemoAuthSession(): AuthSession | null {
  return memoryAuthSession || read<AuthSession>(DEMO_AUTH_SESSION_KEY);
}

export function ensureDemoSessions(): void {
  memoryV1Session = getDemoV1Session() || DEMO_V1_SESSION;
  memoryAuthSession = getDemoAuthSession() || DEMO_AUTH_SESSION;
  write(DEMO_V1_SESSION_KEY, memoryV1Session);
  write(DEMO_AUTH_SESSION_KEY, memoryAuthSession);
}

export function setDemoV1Session(session: V1Session): void {
  memoryV1Session = session;
  write(DEMO_V1_SESSION_KEY, session);
}

export function setDemoAuthSession(session: AuthSession): void {
  memoryAuthSession = session;
  write(DEMO_AUTH_SESSION_KEY, session);
}

export function clearDemoSessions(): void {
  memoryV1Session = null;
  memoryAuthSession = null;
  try {
    storage()?.removeItem(DEMO_V1_SESSION_KEY);
    storage()?.removeItem(DEMO_AUTH_SESSION_KEY);
  } catch {
    // Nothing else to clear.
  }
}

