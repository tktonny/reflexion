import * as FileSystem from 'expo-file-system/legacy';
import { getWebStorage, migrateLegacyPlaintextFile, secureDelete, secureGet, secureSet } from './secureStorage';

export type AuthSession = {
  nurseId: string;
  name: string;
  email: string;
};

const AUTH_SESSION_KEY = 'reflexion.authSession';
// Where builds before the SecureStore migration kept this session in plaintext.
const LEGACY_AUTH_SESSION_FILE = FileSystem.documentDirectory
  ? `${FileSystem.documentDirectory}reflexion-auth-session.json`
  : '';

let memorySession: AuthSession | null = null;

function parseSession(raw: string | null | undefined): AuthSession | null {
  if (!raw) {
    return null;
  }
  try {
    const parsed = JSON.parse(raw) as Partial<AuthSession>;
    if (!parsed.nurseId || !parsed.email) {
      return null;
    }
    return {
      nurseId: parsed.nurseId,
      name: parsed.name || '',
      email: parsed.email,
    };
  } catch {
    return null;
  }
}

/**
 * Synchronous read for render bodies and route guards. Native storage is async, so this serves the
 * in-memory copy that loadStoredAuthSession() hydrates once on boot; on web it reads localStorage
 * directly. Callers must not treat a null here as "signed out" before hydration has run.
 */
export function getStoredAuthSession(): AuthSession | null {
  if (memorySession) {
    return memorySession;
  }

  const storage = getWebStorage();
  if (!storage) {
    return memorySession;
  }

  const raw = storage.getItem(AUTH_SESSION_KEY);
  const parsed = parseSession(raw);
  if (!parsed && raw) {
    storage.removeItem(AUTH_SESSION_KEY);
  }
  return parsed;
}

export async function loadStoredAuthSession(): Promise<AuthSession | null> {
  const existingSession = getStoredAuthSession();
  if (existingSession) {
    return existingSession;
  }

  memorySession = parseSession(await secureGet(AUTH_SESSION_KEY));
  if (memorySession) {
    return memorySession;
  }

  // One-time move off the plaintext file an earlier build wrote.
  memorySession = parseSession(await migrateLegacyPlaintextFile(AUTH_SESSION_KEY, LEGACY_AUTH_SESSION_FILE));
  return memorySession;
}

export async function setStoredAuthSession(session: AuthSession) {
  memorySession = session;
  await secureSet(AUTH_SESSION_KEY, JSON.stringify(session));
}

export async function clearStoredAuthSession() {
  memorySession = null;
  await secureDelete(AUTH_SESSION_KEY);
}
