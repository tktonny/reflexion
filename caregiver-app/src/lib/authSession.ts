import * as FileSystem from 'expo-file-system/legacy';
import { getWebStorage, migrateLegacyPlaintextFile, secureDelete, secureGet, secureSet } from './secureStorage';

/**
 * `userId` is the v1 user id. It was called `nurseId` while identity travelled in legacy query strings;
 * v1 takes identity from the bearer token, so the app only keeps it to key caches and to address the
 * caregiver's own resources. For a caregiver migrated from legacy the value is unchanged — the v1 user id
 * IS the legacy nurse ObjectId hex — so a session written by an older build stays valid.
 */
export type AuthSession = {
  userId: string;
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
    const parsed = JSON.parse(raw) as Partial<AuthSession> & { nurseId?: string };
    // Accept the pre-v1 shape: an installed build wrote `nurseId`, and rejecting it here would sign every
    // existing caregiver out on upgrade. The value carries over as-is.
    const userId = parsed.userId || parsed.nurseId;
    if (!userId || !parsed.email) {
      return null;
    }
    return {
      userId,
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
