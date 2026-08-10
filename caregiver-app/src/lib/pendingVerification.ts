import { secureDelete, secureGet, secureSet } from './secureStorage';

const PENDING_VERIFICATION_KEY = 'reflexion.pendingVerification';

export type PendingVerification = { email: string; createdAt: string };

export async function savePendingVerification(email: string): Promise<void> {
  await secureSet(PENDING_VERIFICATION_KEY, JSON.stringify({ email: email.trim().toLowerCase(), createdAt: new Date().toISOString() } satisfies PendingVerification));
}

export async function loadPendingVerification(): Promise<PendingVerification | null> {
  const raw = await secureGet(PENDING_VERIFICATION_KEY);
  if (!raw) return null;
  try {
    const value = JSON.parse(raw) as Partial<PendingVerification>;
    return typeof value.email === 'string' && value.email.length > 0
      ? { email: value.email, createdAt: typeof value.createdAt === 'string' ? value.createdAt : new Date().toISOString() }
      : null;
  } catch {
    return null;
  }
}

export function clearPendingVerification(): Promise<void> {
  return secureDelete(PENDING_VERIFICATION_KEY);
}

export const pendingVerificationStorageKey = PENDING_VERIFICATION_KEY;
