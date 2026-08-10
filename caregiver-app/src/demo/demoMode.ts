import { useSyncExternalStore } from 'react';

import { clearDemoSessions, ensureDemoSessions } from './demoSession';

const DEMO_MODE_KEY = 'reflexion.demo.mode';

// The selector is available in development builds and in an explicitly opted-in test build only.
// Production builds keep the normal sign-in path and do not expose a demo entry point.
export function isDemoFeatureEnabled(): boolean {
  return process.env.NODE_ENV !== 'production' || process.env.EXPO_PUBLIC_ENABLE_DEMO_MODE === 'true';
}

let enabled = false;
let hydrated = false;
const listeners = new Set<() => void>();

function storage(): Storage | null {
  if (typeof globalThis === 'undefined') return null;
  return (globalThis as typeof globalThis & { localStorage?: Storage }).localStorage ?? null;
}

function notify() {
  listeners.forEach((listener) => listener());
}

function readPersistedMode(): boolean {
  if (!isDemoFeatureEnabled()) return false;
  try {
    return storage()?.getItem(DEMO_MODE_KEY) === 'true';
  } catch {
    return false;
  }
}

export function isDemoMode(): boolean {
  if (!isDemoFeatureEnabled()) return false;
  return enabled || (!hydrated && readPersistedMode());
}

export async function loadDemoMode(): Promise<boolean> {
  if (!isDemoFeatureEnabled()) {
    enabled = false;
    hydrated = true;
    return false;
  }
  enabled = readPersistedMode();
  hydrated = true;
  if (enabled) ensureDemoSessions();
  notify();
  return enabled;
}

export async function enterDemoMode(): Promise<void> {
  if (!isDemoFeatureEnabled()) return;
  enabled = true;
  hydrated = true;
  try {
    storage()?.setItem(DEMO_MODE_KEY, 'true');
  } catch {
    // A memory-only demo is still safe and useful when browser storage is unavailable.
  }
  ensureDemoSessions();
  notify();
}

export async function exitDemoMode(): Promise<void> {
  enabled = false;
  hydrated = true;
  try {
    storage()?.removeItem(DEMO_MODE_KEY);
  } catch {
    // Nothing else is persisted by this helper.
  }
  clearDemoSessions();
  notify();
}

export function subscribeDemoMode(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useDemoMode(): boolean {
  return useSyncExternalStore(subscribeDemoMode, isDemoMode, () => false);
}

