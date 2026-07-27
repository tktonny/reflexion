import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert } from 'react-native';
import { getStoredAuthSession, setStoredAuthSession } from '../../lib/authSession';
import { registerPushNotificationDevice } from '../../lib/pushNotifications';
import { invalidateCaregiverConfig, settingsConfigKey } from '../../lib/queryKeys';
import {
  loadCaregiverSettings,
  putCarePlanV1,
  updateCaregiverProfileV1,
  updatePatientV1,
  type CaregiverSettingsPatient,
} from '../../lib/v1Caregiver';
import { normalizeSettingsPatient } from './helpers';
import type { AlertSensitivity, SettingsConfig, SettingsPatient, SummaryTime } from './types';

/**
 * The settings data layer, shared by the hub and every sub-page.
 *
 * Settings used to be one screen holding one copy of this. Splitting it into a hub plus sub-pages means
 * several screens now read the same thing, so the query lives here under one key: the hub shows a summary of
 * a value, the sub-page edits it, and neither refetches what the other already has.
 *
 * The mutations take PARTIAL input on purpose. PATCH /me only writes the keys present in the body, so an
 * "Account" page that never loaded the notification preferences cannot blank them — which is the property
 * that makes it safe to have more than one screen writing to the same resource.
 */

/** The v1 loved one, flattened into the row the edit form works with. */
export function toSettingsPatient(patient: CaregiverSettingsPatient): SettingsPatient {
  const communication = patient.carePlan?.communicationPreferences;
  return normalizeSettingsPatient({
    id: patient.patientId,
    patientId: patient.patientId,
    version: patient.version,
    planVersion: patient.carePlan?.version ?? 0,
    name: patient.displayName,
    phoneNumber: patient.profile.phoneNumber || '',
    age: patient.profile.age ?? 0,
    gender: patient.profile.gender ?? '',
    preferredLanguage: patient.preferredLanguage as SettingsPatient['preferredLanguage'],
    photoUrl: patient.profile.photoUrl || '',
    usualWakeTime: String(patient.carePlan?.dailyRoutine?.wakeTime || ''),
    speechOrHearingConditions: String(communication?.speechOrHearingNotes || ''),
    keyTopics: (communication?.topics || []) as SettingsPatient['keyTopics'],
    keyTopicsOtherText: String(communication?.otherTopic || ''),
  });
}

export function useCaregiverSettings() {
  const session = getStoredAuthSession();
  return useQuery({
    enabled: Boolean(session?.userId),
    queryKey: settingsConfigKey(session?.userId),
    queryFn: async (): Promise<SettingsConfig> => {
      const view = await loadCaregiverSettings();
      return {
        nurseId: view.caregiver.userId,
        caregiverName: view.caregiver.name,
        email: view.caregiver.email,
        phoneNumber: view.caregiver.phoneNumber,
        pushNotificationsEnabled: view.caregiver.notificationPreferences.pushNotificationsEnabled,
        alertSensitivity: view.caregiver.notificationPreferences.alertSensitivity as AlertSensitivity,
        preferredDailySummaryTime: view.caregiver.notificationPreferences.preferredDailySummaryTime as SummaryTime,
        storeSessionSummaries: view.caregiver.storeSessionSummaries,
        patients: view.patients.map(toSettingsPatient),
      };
    },
  });
}

export type CaregiverProfilePatch = {
  name?: string;
  phoneNumber?: string;
  notificationPreferences?: Partial<{
    pushNotificationsEnabled: boolean;
    alertSensitivity: AlertSensitivity;
    preferredDailySummaryTime: SummaryTime;
  }>;
  storeSessionSummaries?: boolean;
};

/**
 * Writes whichever slice of the caregiver's own profile a page is responsible for.
 *
 * `onSaved` runs after the cache is invalidated so a sub-page can navigate back on success — the hub then
 * renders the new summary from fresh data rather than from whatever the sub-page happened to hold.
 */
export function useSaveCaregiverProfile(onSaved?: () => void) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: CaregiverProfilePatch) => updateCaregiverProfileV1(input),
    onSuccess: async (body) => {
      // Turning push on is the moment to claim this phone; doing it here rather than on the toggle means a
      // caregiver who flips it and leaves without saving has not registered a device they did not confirm.
      if (body.notificationPreferences.pushNotificationsEnabled) {
        const registration = await registerPushNotificationDevice({ nurseId: body.userId });
        if (!registration.ok) console.warn('[settings] push registration failed', registration.reason);
      }
      await setStoredAuthSession({ userId: body.userId, name: body.name, email: body.email });
      await invalidateCaregiverConfig(queryClient);
      onSaved?.();
    },
    onError: (error) => {
      Alert.alert('Unable to save', error instanceof Error ? error.message : 'Unable to save settings.');
    },
  });
}

/**
 * Saves a loved one across the two resources v1 splits them into: their own details on the patient record,
 * and everything that changes how Aria talks on the care plan — which is what the mirror already receives
 * through GET /devices/:id/configuration. Both writes are versioned, so the row carries the versions it was
 * rendered from and a concurrent edit on another phone conflicts instead of being silently overwritten.
 */
export function useSaveLovedOne(onSaved?: () => void) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (patient: SettingsPatient) => {
      const patientId = patient.patientId || patient.id;
      const saved = await updatePatientV1(patientId, patient.version, {
        displayName: patient.name,
        preferredLanguage: patient.preferredLanguage || 'english',
        profile: {
          age: patient.age || null,
          gender: patient.gender || null,
          photoUrl: patient.photoUrl || null,
          phoneNumber: patient.phoneNumber || null,
        },
      });
      const plan = await putCarePlanV1(patientId, patient.planVersion, {
        dailyRoutine: { wakeTime: patient.usualWakeTime || '' },
        communicationPreferences: {
          topics: patient.keyTopics,
          otherTopic: patient.keyTopicsOtherText || '',
          speechOrHearingNotes: patient.speechOrHearingConditions || '',
        },
      });
      return normalizeSettingsPatient({ ...patient, version: saved.version, planVersion: plan.version });
    },
    onSuccess: async () => {
      await invalidateCaregiverConfig(queryClient);
      // The trend and the day view render this loved one's name and language, so they go stale too.
      await queryClient.invalidateQueries({ queryKey: ['patientTrend'] });
      await queryClient.invalidateQueries({ queryKey: ['sessionDay'] });
      onSaved?.();
    },
    onError: (error) => {
      Alert.alert('Unable to save', error instanceof Error ? error.message : 'Unable to save this profile.');
    },
  });
}
