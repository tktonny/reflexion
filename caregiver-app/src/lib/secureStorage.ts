import * as FileSystem from 'expo-file-system/legacy';
import * as SecureStore from 'expo-secure-store';
import { Platform } from 'react-native';

// Session storage for both the legacy caregiver session and the v1 token pair.
//
// These values include a 30-day v1 refresh token, so on device they belong in the OS keystore (iOS
// Keychain / Android Keystore) — not in a plaintext JSON file under documentDirectory, where Android's
// auto-backup would sweep them up. On web there is no keystore, so localStorage stays the target, which
// is what this app already did there.
//
// Values are small (a compact JWT plus a 43-character refresh secret), comfortably inside SecureStore's
// per-value size limit.

export function getWebStorage(): Storage | null {
  if (typeof globalThis === 'undefined') {
    return null;
  }
  return (globalThis as typeof globalThis & { localStorage?: Storage }).localStorage ?? null;
}

export async function secureGet(key: string): Promise<string | null> {
  const storage = getWebStorage();
  if (storage) {
    return storage.getItem(key);
  }
  try {
    return await SecureStore.getItemAsync(key);
  } catch {
    return null;
  }
}

export async function secureSet(key: string, value: string): Promise<void> {
  const storage = getWebStorage();
  if (storage) {
    storage.setItem(key, value);
    return;
  }
  await SecureStore.setItemAsync(key, value);
}

export async function secureDelete(key: string): Promise<void> {
  const storage = getWebStorage();
  if (storage) {
    storage.removeItem(key);
    return;
  }
  try {
    await SecureStore.deleteItemAsync(key);
  } catch {
    // Already absent — nothing to do.
  }
}

/**
 * Moves a session that an earlier build wrote to a plaintext file into secure storage, then removes the
 * file. Runs once per install; without it, upgrading would silently sign every existing caregiver out.
 * Returns the migrated value, or null when there was nothing to migrate.
 */
export async function migrateLegacyPlaintextFile(key: string, fileUri: string): Promise<string | null> {
  if (Platform.OS === 'web' || !fileUri) {
    return null;
  }
  try {
    const raw = await FileSystem.readAsStringAsync(fileUri);
    if (!raw) return null;
    await secureSet(key, raw);
    await FileSystem.deleteAsync(fileUri, { idempotent: true });
    return raw;
  } catch {
    // No legacy file (the normal case on a fresh install).
    return null;
  }
}
