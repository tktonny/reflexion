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
import { apiGet, apiSend } from '../../src/lib/apiClient';
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
      const query = `?nurseId=${encodeURIComponent(session?.userId || '')}`;
      const body = await apiGet<Record<string, unknown> & { patients?: unknown[] }>(`/api/nurse-patient-config/latest${query}`);
      return {
        nurseId: String(body?.nurseId || ''),
        caregiverName: String(body?.caregiverName || ''),
        email: String(body?.email || ''),
        phoneNumber: String(body?.phoneNumber || ''),
        pushNotificationsEnabled: Boolean(body?.pushNotificationsEnabled),
        alertSensitivity: String(body?.alertSensitivity || 'only_important_changes') as AlertSensitivity,
        preferredDailySummaryTime: String(body?.preferredDailySummaryTime || '09:00') as SummaryTime,
        storeSessionSummaries: body?.storeSessionSummaries !== false,
        patients: Array.isArray(body?.patients) ? body.patients.map((patient) => normalizeSettingsPatient(patient as Partial<SettingsPatient>)) : [],
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
    mutationFn: (body: unknown) => apiSend<SettingsConfig>('/api/nurse-patient-config/settings', {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
    onSuccess: async (body) => {
      if (body.pushNotificationsEnabled) {
        const registration = await registerPushNotificationDevice({ nurseId: body.nurseId || config?.nurseId || '' });
        if (!registration.ok) console.warn('[SettingsScreen] push registration failed', registration.reason);
      }
      await setStoredAuthSession({
        userId: body.nurseId || config?.nurseId || '',
        name: body.caregiverName || caregiverName,
        email: body.email || config?.email || '',
      });
      setConfig((current) => current ? {
        ...current,
        caregiverName: body.caregiverName || caregiverName,
        phoneNumber: body.phoneNumber || phoneNumber,
        pushNotificationsEnabled: Boolean(body.pushNotificationsEnabled),
        alertSensitivity: body.alertSensitivity || alertLevel,
        preferredDailySummaryTime: body.preferredDailySummaryTime || summaryTime,
        storeSessionSummaries: body.storeSessionSummaries !== false,
      } : current);
      await invalidateCaregiverConfig(queryClient);
      Alert.alert('Saved', 'Settings updated.');
    },
    onError: (err) => {
      Alert.alert('Unable to save', err instanceof Error ? err.message : 'Unable to save settings.');
    },
  });
  const savePatientMutation = useMutation({
    mutationFn: ({ patientId, body }: { patientId: string; body: unknown }) =>
      apiSend<{ patient: SettingsPatient }>(`/api/nurse-patient-config/settings/patients/${patientId}`, {
        method: 'PATCH',
        body: JSON.stringify(body),
      }),
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
    saveNurseMutation.mutate({
      nurseId: config.nurseId,
      name: caregiverName,
      phoneNumber,
      pushNotificationsEnabled: notifs,
      alertSensitivity: alertLevel,
      preferredDailySummaryTime: summaryTime,
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

    savePatientMutation.mutate({
      patientId: editingPatient.patientId || editingPatient.id,
      body: {
        nurseId: config.nurseId,
        ...editingPatient,
        age,
      },
    });
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
