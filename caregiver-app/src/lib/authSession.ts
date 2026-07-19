import * as FileSystem from 'expo-file-system/legacy';
import { Platform } from 'react-native';

export type AuthSession = {
  nurseId: string;
  name: string;
  email: string;
};

const AUTH_SESSION_KEY = 'reflexion.authSession';
const AUTH_SESSION_FILE = FileSystem.documentDirectory
  ? `${FileSystem.documentDirectory}reflexion-auth-session.json`
  : '';

let memorySession: AuthSession | null = null;

function getStorage() {
  if (typeof globalThis === 'undefined') {
    return null;
  }

  return (globalThis as typeof globalThis & { localStorage?: Storage }).localStorage ?? null;
}

export function getStoredAuthSession(): AuthSession | null {
  if (memorySession) {
    return memorySession;
  }

  const storage = getStorage();
  if (!storage) {
    return memorySession;
  }

  const rawSession = storage.getItem(AUTH_SESSION_KEY);
  if (!rawSession) {
    return null;
  }

  try {
    const parsed = JSON.parse(rawSession) as Partial<AuthSession>;
    if (!parsed.nurseId || !parsed.email) {
      return null;
    }

    return {
      nurseId: parsed.nurseId,
      name: parsed.name || '',
      email: parsed.email,
    };
  } catch {
    storage.removeItem(AUTH_SESSION_KEY);
    return null;
  }
}

export async function loadStoredAuthSession(): Promise<AuthSession | null> {
  const existingSession = getStoredAuthSession();
  if (existingSession || Platform.OS === 'web' || !AUTH_SESSION_FILE) {
    return existingSession;
  }

  try {
    const rawSession = await FileSystem.readAsStringAsync(AUTH_SESSION_FILE);
    const parsed = JSON.parse(rawSession) as Partial<AuthSession>;
    if (!parsed.nurseId || !parsed.email) {
      return null;
    }

    memorySession = {
      nurseId: parsed.nurseId,
      name: parsed.name || '',
      email: parsed.email,
    };
    return memorySession;
  } catch {
    return null;
  }
}

export async function setStoredAuthSession(session: AuthSession) {
  memorySession = session;
  const serializedSession = JSON.stringify(session);
  const storage = getStorage();
  if (storage) {
    storage.setItem(AUTH_SESSION_KEY, serializedSession);
    return;
  }

  if (Platform.OS !== 'web' && AUTH_SESSION_FILE) {
    await FileSystem.writeAsStringAsync(AUTH_SESSION_FILE, serializedSession);
  }
}

export async function clearStoredAuthSession() {
  memorySession = null;
  const storage = getStorage();
  if (storage) {
    storage.removeItem(AUTH_SESSION_KEY);
    return;
  }

  if (Platform.OS !== 'web' && AUTH_SESSION_FILE) {
    await FileSystem.deleteAsync(AUTH_SESSION_FILE, { idempotent: true });
  }
}
