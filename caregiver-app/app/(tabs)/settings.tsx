import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  loadCaregiverSettings,
  putCarePlanV1,
  updateCaregiverProfileV1,
  updatePatientV1,
  type CaregiverSettingsPatient,
} from '../../src/lib/v1Caregiver';
import { clearStoredAuthSession, getStoredAuthSession, setStoredAuthSession } from '../../src/lib/authSession';
import { v1Logout } from '../../src/lib/v1Client';
import { registerPushNotificationDevice } from '../../src/lib/pushNotifications';
import { clearCaregiverCache, invalidateCaregiverConfig, settingsConfigKey } from '../../src/lib/queryKeys';
import { PatientEditModal } from '../../src/screens/settings/PatientEditModal';
import { SettingsPlaceholder } from '../../src/screens/settings/SettingsPlaceholder';
import {
  ActionRow,
  InputRow,
  PickerRow,
  SectionHeader,
  SettingRow,
  SwitchRow,
} from '../../src/screens/settings/SettingsRows';
import {
  formatLanguage,
  normalizeSettingsPatient,
  parsePatientAge,
  resolveSettingsState,
  toPatientForm,
} from '../../src/screens/settings/helpers';
import type {
  AlertSensitivity,
  PatientForm,
  SettingsConfig,
  SettingsPatient,
  SummaryTime,
} from '../../src/screens/settings/types';
import { colors, fontFamily, fontSize, radius, spacing, scaleSize, MIN_TOUCH_TARGET } from '../../src/theme';

/** The v1 loved one, flattened into the row the form edits. */
function toSettingsPatient(patient: CaregiverSettingsPatient): SettingsPatient {
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

export default function SettingsScreen() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const session = getStoredAuthSession();
  const [notifs, setNotifs] = useState(true);
  const [summaryTime, setSummaryTime] = useState<SummaryTime>('09:00');
  const [alertLevel, setAlertLevel] = useState<AlertSensitivity>('only_important_changes');
  const [storeSummaries, setStoreSummaries] = useState(true);
  const [caregiverName, setCaregiverName] = useState('');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [config, setConfig] = useState<SettingsConfig | null>(null);
  const [editingPatient, setEditingPatient] = useState<PatientForm | null>(null);
  const latestConfigQuery = useQuery({
    // nurseId is required by the endpoint — asking without one used to return an arbitrary caregiver.
    enabled: Boolean(session?.userId),
    queryKey: settingsConfigKey(session?.userId),
    queryFn: async () => {
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
      } satisfies SettingsConfig;
    },
  });
  const { refetch: refetchLatestConfig } = latestConfigQuery;
  useFocusEffect(
    useCallback(() => {
      if (session?.userId) void refetchLatestConfig();
    }, [refetchLatestConfig, session?.userId]),
  );
  const saveNurseMutation = useMutation({
    mutationFn: (input: {
      name: string;
      phoneNumber: string;
      notificationPreferences: {
        pushNotificationsEnabled: boolean;
        alertSensitivity: AlertSensitivity;
        preferredDailySummaryTime: SummaryTime;
      };
      storeSessionSummaries: boolean;
    }) => updateCaregiverProfileV1(input),
    onSuccess: async (body) => {
      if (body.notificationPreferences.pushNotificationsEnabled) {
        const registration = await registerPushNotificationDevice({ nurseId: body.userId });
        if (!registration.ok) console.warn('[SettingsScreen] push registration failed', registration.reason);
      }
      await setStoredAuthSession({
        userId: body.userId,
        name: body.name || caregiverName,
        email: body.email,
      });
      setConfig((current) => current ? {
        ...current,
        caregiverName: body.name || caregiverName,
        phoneNumber: body.phoneNumber || phoneNumber,
        pushNotificationsEnabled: body.notificationPreferences.pushNotificationsEnabled,
        alertSensitivity: body.notificationPreferences.alertSensitivity as AlertSensitivity,
        preferredDailySummaryTime: body.notificationPreferences.preferredDailySummaryTime as SummaryTime,
        storeSessionSummaries: body.storeSessionSummaries,
      } : current);
      await invalidateCaregiverConfig(queryClient);
      Alert.alert('Saved', 'Settings updated.');
    },
    onError: (err) => {
      Alert.alert('Unable to save', err instanceof Error ? err.message : 'Unable to save settings.');
    },
  });
  // Two resources, because v1 splits them by who consumes them: the loved one's own details on the patient
  // record, and everything that changes how Aria talks on the care plan — which is what the mirror already
  // receives through GET /devices/:id/configuration. The legacy route wrote both into one document where the
  // conversational half sat unread.
  const savePatientMutation = useMutation({
    mutationFn: async ({ patient }: { patient: SettingsPatient }) => {
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
      return { patient: { ...patient, version: saved.version, planVersion: plan.version } };
    },
    onSuccess: async (body) => {
      const normalized = normalizeSettingsPatient(body.patient);
      setConfig((current) => current ? {
        ...current,
        patients: current.patients.map((patient) =>
          patient.id === normalized.id || patient.patientId === normalized.id ? normalized : patient,
        ),
      } : current);
      setEditingPatient(null);
      await invalidateCaregiverConfig(queryClient);
      await queryClient.invalidateQueries({ queryKey: ['patientTrend'] });
      await queryClient.invalidateQueries({ queryKey: ['sessionDay'] });
    },
    onError: (err) => {
      Alert.alert('Unable to save', err instanceof Error ? err.message : 'Unable to save loved one profile.');
    },
  });

  useEffect(() => {
    if (!latestConfigQuery.data) return;
    setConfig(latestConfigQuery.data);
    setCaregiverName(latestConfigQuery.data.caregiverName);
    setPhoneNumber(latestConfigQuery.data.phoneNumber);
    setNotifs(latestConfigQuery.data.pushNotificationsEnabled);
    setAlertLevel(latestConfigQuery.data.alertSensitivity);
    setSummaryTime(latestConfigQuery.data.preferredDailySummaryTime);
    setStoreSummaries(latestConfigQuery.data.storeSessionSummaries);
  }, [latestConfigQuery.data]);

  async function saveNurseSettings() {
    if (saveNurseMutation.isPending || !config) return;
    // No id in the body: PATCH /me writes the account the bearer token belongs to.
    saveNurseMutation.mutate({
      name: caregiverName,
      phoneNumber,
      notificationPreferences: {
        pushNotificationsEnabled: notifs,
        alertSensitivity: alertLevel,
        preferredDailySummaryTime: summaryTime,
      },
      storeSessionSummaries: storeSummaries,
    });
  }

  async function savePatientProfile() {
    if (!editingPatient || !config || savePatientMutation.isPending) return;

    const age = parsePatientAge(editingPatient.age);
    if (age === null) {
      Alert.alert('Check the age', 'Please enter their age as a whole number between 1 and 130.');
      return;
    }

    savePatientMutation.mutate({ patient: { ...editingPatient, age } });
  }

  async function logout() {
    await Promise.all([clearStoredAuthSession(), v1Logout()]);
    // Before navigating away, so the next caregiver on this phone cannot read the previous one's cached
    // loved ones, alerts or statuses. See clearCaregiverCache.
    clearCaregiverCache(queryClient);
    router.replace('/sign-in');
  }

  const settingsState = resolveSettingsState({
    hasNurseId: Boolean(session?.userId),
    hasFailed: Boolean(latestConfigQuery.error),
    hasSettings: Boolean(config?.nurseId || latestConfigQuery.data?.nurseId),
    isLoading: latestConfigQuery.isLoading,
  });

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
        <View style={styles.titleRow}>
          <Text maxFontSizeMultiplier={1.3} style={styles.pageTitle}>Settings</Text>
        </View>

        {settingsState !== 'ready' ? (
          // Loading / signed-out / failed / genuinely-empty used to collapse into one loading card: a
          // failed load fell through to the form with blank fields, so a caregiver could "save" empties
          // over their own settings and never be told the load had failed. Live region so a failed retry
          // is announced rather than silently swapping the card.
          <>
            <View accessibilityLiveRegion="polite" style={styles.stateWrap}>
              <SettingsPlaceholder onRetry={() => void latestConfigQuery.refetch()} state={settingsState} />
            </View>

            {/* These two do not depend on the failed query, and hiding them was the worst possible moment
                to do it: this row is the app's ONLY route to /mirror-management, so a caregiver whose
                settings would not load also lost every path to repairing the mirror connection. */}
            <SectionHeader title="Mirrors" />
            <ActionRow label="Manage linked mirrors" onPress={() => router.push('/mirror-management')} />
            <ActionRow label="Add a loved one" onPress={() => router.push('/onboarding?mode=add-patient&returnTo=settings')} />
            {/* No log-out button here: the one below sits outside this branch and already renders in every
                state, so adding another produced two identical controls on the failed/loading screen. */}
          </>
        ) : (
          <>
            <SectionHeader title="Account" />
            <InputRow label="Name" value={caregiverName} onChangeText={setCaregiverName} />
            <SettingRow label="Email" value={config?.email || 'Not connected'} />
            <InputRow label="Phone" value={phoneNumber} onChangeText={setPhoneNumber} keyboardType="phone-pad" />

            <SectionHeader title="Notifications" />
            <SwitchRow label="Enable push notifications" value={notifs} onChange={setNotifs} />
            <PickerRow
              label="Daily summary"
              options={[
                { value: '09:00', label: 'Morning (9am)' },
                { value: '19:00', label: 'Evening (7pm)' },
              ]}
              selected={summaryTime}
              onSelect={v => setSummaryTime(v as SummaryTime)}
            />
            <PickerRow
              label="Alert sensitivity"
              options={[
                { value: 'notify_me_about_everything', label: 'Notify me about everything' },
                { value: 'only_important_changes', label: 'Only important changes' },
                { value: 'only_urgent_alerts', label: 'Only urgent alerts' },
              ]}
              selected={alertLevel}
              onSelect={v => setAlertLevel(v as AlertSensitivity)}
            />

            <SectionHeader title="Mirrors" />
            <ActionRow label="Manage linked mirrors" onPress={() => router.push('/mirror-management')} />

            <SectionHeader title="Loved one profiles" />
            {(config?.patients || []).map(patient => (
              <ActionRow
                key={patient.id}
                label={patient.name}
                value={formatLanguage(patient.preferredLanguage)}
                imageUrl={patient.photoUrl}
                fallbackName={patient.name}
                onPress={() => setEditingPatient(toPatientForm(patient))}
              />
            ))}
            <ActionRow label="Add a loved one" onPress={() => router.push('/onboarding?mode=add-patient&returnTo=settings')} />

            <SectionHeader title="Privacy & Data" />
            <SwitchRow label="Store session summaries" value={storeSummaries} onChange={setStoreSummaries} />
            <TouchableOpacity
              // While saving the button shows only a spinner, so the label cannot come from its children.
              accessibilityLabel={saveNurseMutation.isPending ? 'Saving changes' : 'Save changes'}
              accessibilityRole="button"
              accessibilityState={{ busy: saveNurseMutation.isPending, disabled: saveNurseMutation.isPending }}
              disabled={saveNurseMutation.isPending}
              onPress={() => void saveNurseSettings()}
              style={[styles.saveBtn, saveNurseMutation.isPending && styles.saveBtnDisabled]}
            >
              {saveNurseMutation.isPending ? <ActivityIndicator color={colors.text.onAccent} /> : <Text style={styles.saveBtnText}>Save changes</Text>}
            </TouchableOpacity>
            <ActionRow label="Export my data" onPress={() => Alert.alert('Export', 'Data export coming in V2.')} />

            <SectionHeader title="Support" />
            <ActionRow label="FAQ & Guide" onPress={() => router.push('/faq')} />
            <ActionRow label="Chat with support" onPress={() => router.push('/chatbot')} />
            <ActionRow label="Send feedback" onPress={() => router.push('/feedback')} />
            <ActionRow label="Subscription & Billing" onPress={() => Alert.alert('Billing', 'Billing portal coming soon.')} />
          </>
        )}

        <TouchableOpacity accessibilityRole="button" onPress={() => void logout()} style={styles.logoutBtn}>
          <Text style={styles.logoutText}>Log out</Text>
        </TouchableOpacity>
      </ScrollView>
      <PatientEditModal
        patient={editingPatient}
        isSaving={savePatientMutation.isPending}
        onChange={setEditingPatient}
        onClose={() => setEditingPatient(null)}
        onSave={() => void savePatientProfile()}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.surface.page },
  scroll: { flex: 1 },
  content: { paddingBottom: scaleSize(60) },
  titleRow: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.xl,
    paddingTop: spacing.xl,
    paddingBottom: spacing.sm,
    gap: spacing.md,
  },
  pageTitle: { fontSize: fontSize.display, fontWeight: '500', color: colors.text.primary, fontFamily: fontFamily.display },
  // The rows below are full-bleed; the shared state cards are not, so they get the page's own gutter.
  stateWrap: { marginHorizontal: spacing.xl, marginTop: spacing.xl },
  saveBtn: {
    alignItems: 'center',
    backgroundColor: colors.accent,
    justifyContent: 'center',
    marginHorizontal: spacing.xl,
    marginTop: spacing.md,
    minHeight: MIN_TOUCH_TARGET,
    borderRadius: radius.md,
  },
  saveBtnDisabled: { opacity: 0.7 },
  saveBtnText: { color: colors.text.onAccent, fontSize: fontSize.subheading, fontWeight: '700' },
  logoutBtn: {
    margin: spacing.xl,
    marginTop: scaleSize(36),
    padding: spacing.lg,
    backgroundColor: colors.surface.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border.strong,
    alignItems: 'center',
  },
  logoutText: { color: colors.accent, fontSize: fontSize.subheading, fontWeight: '600' },
});
