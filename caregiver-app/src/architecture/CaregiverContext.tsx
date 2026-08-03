import React, { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';
import * as SecureStore from 'expo-secure-store';

import type { SetupCategory, SetupStatus } from './models';
import { hasV1Session } from '../lib/v1AuthSession';
import { getSetupProgressV1, updateSetupProgressV1 } from '../lib/v1Caregiver';

type CaregiverState = {
  setup: Record<SetupCategory, SetupStatus>;
  notificationsEnabled: boolean;
  language: 'en' | 'zh';
  setupLoading: boolean;
  setupError: string | null;
  loadSetupProgress: () => Promise<void>;
  setSetupStatus: (category: SetupCategory, status: SetupStatus) => void;
  setNotificationsEnabled: (value: boolean) => void;
  setLanguage: (value: 'en' | 'zh') => void;
};

const INITIAL_SETUP: Record<SetupCategory, SetupStatus> = {
  household: 'not-started',
  'pair-device': 'not-started',
  'language-accessibility': 'not-started',
  routines: 'not-started',
  notifications: 'not-started',
  'consent-control': 'not-started',
  'care-circle': 'not-started',
  'research-participation': 'not-started',
};

const CaregiverContext = createContext<CaregiverState | null>(null);

export function CaregiverProvider({ children }: { children: React.ReactNode }) {
  const [setup, setSetup] = useState(INITIAL_SETUP);
  const [setupVersion, setSetupVersion] = useState(1);
  const [setupLoading, setSetupLoading] = useState(false);
  const [setupError, setSetupError] = useState<string | null>(null);
  const setupVersionRef = useRef(1);
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);
  const [language, setLanguageState] = useState<'en' | 'zh'>('en');
  React.useEffect(() => {
    void SecureStore.getItemAsync('reflexion.caregiver.language').then((stored) => {
      if (stored === 'en' || stored === 'zh') setLanguageState(stored);
    }).catch(() => undefined);
  }, []);
  const setLanguage = useCallback((value: 'en' | 'zh') => {
    setLanguageState(value);
    void SecureStore.setItemAsync('reflexion.caregiver.language', value).catch(() => undefined);
  }, []);
  const loadSetupProgress = useCallback(async () => {
    if (!hasV1Session()) return;
    setSetupLoading(true);
    setSetupError(null);
    try {
      const progress = await getSetupProgressV1();
      setSetup(progress.categories);
      setSetupVersion(progress.version);
      setupVersionRef.current = progress.version;
    } catch (cause) {
      setSetupError(cause instanceof Error ? cause.message : 'Setup progress could not be loaded.');
    } finally {
      setSetupLoading(false);
    }
  }, []);
  const setSetupStatus = useCallback((category: SetupCategory, status: SetupStatus) => {
    setSetup((current) => ({ ...current, [category]: status }));
    if (!hasV1Session()) return;
    const expectedVersion = setupVersionRef.current;
    void updateSetupProgressV1(category, status, expectedVersion)
      .then((progress) => {
        setSetup(progress.categories);
        setSetupVersion(progress.version);
        setupVersionRef.current = progress.version;
        setSetupError(null);
      })
      .catch((cause) => {
        setSetupError(cause instanceof Error ? cause.message : 'Setup progress could not be saved.');
        void loadSetupProgress();
      });
  }, [loadSetupProgress]);
  const value = useMemo<CaregiverState>(() => ({
    setup,
    notificationsEnabled,
    language,
    setupLoading,
    setupError,
    loadSetupProgress,
    setLanguage,
    setNotificationsEnabled,
    setSetupStatus,
  }), [language, loadSetupProgress, notificationsEnabled, setSetupStatus, setup, setupError, setupLoading, setupVersion]);

  return <CaregiverContext.Provider value={value}>{children}</CaregiverContext.Provider>;
}

export function useCaregiver() {
  const context = useContext(CaregiverContext);
  if (!context) throw new Error('useCaregiver must be used inside CaregiverProvider');
  return context;
}
