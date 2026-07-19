import React, { useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  View, Text, StyleSheet, ScrollView, TouchableOpacity, Switch, Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Feather } from '@expo/vector-icons';
import { getApiUrl } from '../../src/lib/apiUrl';
import { clearStoredAuthSession, getStoredAuthSession } from '../../src/lib/authSession';

type AlertSensitivity =
  | 'notify_me_about_everything'
  | 'only_important_changes'
  | 'only_urgent_alerts';
type SummaryTime = '09:00' | '19:00';

type SettingsPatient = {
  id: string;
  name: string;
  preferredLanguage: string;
  speechSpeed: string;
};

type SettingsConfig = {
  nurseId: string;
  caregiverName: string;
  email: string;
  phoneNumber: string;
  pushNotificationsEnabled: boolean;
  alertSensitivity: AlertSensitivity;
  preferredDailySummaryTime: SummaryTime;
  patients: SettingsPatient[];
};

export default function SettingsScreen() {
  const router = useRouter();
  const [notifs, setNotifs] = useState(true);
  const [summaryTime, setSummaryTime] = useState<SummaryTime>('09:00');
  const [alertLevel, setAlertLevel] = useState<AlertSensitivity>('only_important_changes');
  const [config, setConfig] = useState<SettingsConfig | null>(null);
  const [isLoadingConfig, setIsLoadingConfig] = useState(true);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    let isMounted = true;

    const loadSettings = async () => {
      setIsLoadingConfig(true);
      try {
        const session = getStoredAuthSession();
        const query = session?.nurseId ? `?nurseId=${encodeURIComponent(session.nurseId)}` : '';
        const response = await fetch(getApiUrl(`/api/nurse-patient-config/latest${query}`));
        const body = await response.json();

        if (!response.ok) {
          throw new Error(body?.error || 'Unable to load settings.');
        }

        if (isMounted) {
          setConfig({
            nurseId: body?.nurseId || '',
            caregiverName: body?.caregiverName || '',
            email: body?.email || '',
            phoneNumber: body?.phoneNumber || '',
            pushNotificationsEnabled: Boolean(body?.pushNotificationsEnabled),
            alertSensitivity: body?.alertSensitivity || 'only_important_changes',
            preferredDailySummaryTime: body?.preferredDailySummaryTime || '09:00',
            patients: Array.isArray(body?.patients) ? body.patients : [],
          });
          setNotifs(Boolean(body?.pushNotificationsEnabled));
          setAlertLevel(body?.alertSensitivity || 'only_important_changes');
          setSummaryTime(body?.preferredDailySummaryTime || '09:00');
        }
      } catch (err) {
        console.warn('[SettingsScreen] load settings failed', err);
      } finally {
        if (isMounted) {
          setIsLoadingConfig(false);
        }
      }
    };

    void loadSettings();
    return () => {
      isMounted = false;
    };
  }, []);

  const accountRows = useMemo(
    () => [
      { label: 'Name', value: config?.caregiverName || 'Not connected' },
      { label: 'Email', value: config?.email || 'Not connected' },
      { label: 'Phone', value: config?.phoneNumber || 'Not connected' },
    ],
    [config],
  );

  async function saveNotificationChanges() {
    if (isSaving) {
      return;
    }

    setIsSaving(true);
    try {
      const response = await fetch(getApiUrl('/api/nurse-patient-config/notifications'), {
        method: 'PATCH',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({
          nurseId: config?.nurseId || undefined,
          pushNotificationsEnabled: notifs,
          alertSensitivity: alertLevel,
          preferredDailySummaryTime: summaryTime,
        }),
      });
      const body = await response.json();

      if (!response.ok) {
        throw new Error(body?.error || 'Unable to save notification settings.');
      }

      setConfig((current) =>
        current
          ? {
              ...current,
              pushNotificationsEnabled: body.pushNotificationsEnabled,
              alertSensitivity: body.alertSensitivity,
              preferredDailySummaryTime: body.preferredDailySummaryTime,
            }
          : current,
      );
      Alert.alert('Saved', 'Notification settings updated.');
    } catch (err) {
      Alert.alert(
        'Unable to save',
        err instanceof Error ? err.message : 'Unable to save notification settings.',
      );
    } finally {
      setIsSaving(false);
    }
  }

  async function logout() {
    await clearStoredAuthSession();
    router.replace('/sign-in');
  }

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
        <View style={styles.titleRow}>
          <Text style={styles.pageTitle}>Settings</Text>
        </View>

        {isLoadingConfig && !config ? (
          <View style={styles.loadingCard}>
            <ActivityIndicator color="#87566A" />
            <Text style={styles.loadingTitle}>Bear with us</Text>
            <Text style={styles.loadingText}>We are loading your settings.</Text>
          </View>
        ) : (
          <>

            {/* Account */}
            <SectionHeader title="Account" />
            {accountRows.map((row) => (
              <SettingRow key={row.label} label={row.label} value={row.value} />
            ))}

            {/* Notifications */}
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
            <TouchableOpacity
              disabled={isSaving}
              onPress={saveNotificationChanges}
              style={[styles.saveBtn, isSaving && styles.saveBtnDisabled]}
            >
              {isSaving ? (
                <ActivityIndicator color="#FFFFFF" />
              ) : (
                <Text style={styles.saveBtnText}>Save changes</Text>
              )}
            </TouchableOpacity>

            {/* Voice Companion */}
            <SectionHeader title="Voice Companion (Aria)" />
            {(config?.patients || []).map(patient => (
              <ActionRow
                key={patient.id}
                label={`${patient.name} - Speech speed`}
                value={patient.speechSpeed || 'Slow'}
                onPress={() => Alert.alert('Speech Speed', `Adjust Aria's speed for ${patient.name}. (Coming soon)`)}
              />
            ))}
            <ActionRow label="Manage linked mirrors" onPress={() => router.push('/mirror-management')} />

            {/* Loved One Profiles */}
            <SectionHeader title="Loved one profiles" />
            {(config?.patients || []).map(patient => (
              <ActionRow
                key={patient.id}
                label={patient.name}
                value={formatLanguage(patient.preferredLanguage)}
                onPress={() => Alert.alert('Edit Profile', `Edit profile for ${patient.name}. (Coming soon)`)}
              />
            ))}
            <ActionRow label="Add a loved one" onPress={() => router.push('/onboarding?mode=add-patient')} />

            {/* Privacy */}
            <SectionHeader title="Privacy & Data" />
            <SwitchRow label="Store session summaries" value={false} onChange={() => {}} disabled />
            <ActionRow label="Export my data" onPress={() => Alert.alert('Export', 'Data export coming in V2.')} />

            {/* Support */}
            <SectionHeader title="Support" />
            <ActionRow label="FAQ & Guide" onPress={() => router.push('/faq')} />
            <ActionRow label="Chat with support" onPress={() => router.push('/chatbot')} />
            <ActionRow
              label="Give feedback"
              onPress={() =>
                Alert.alert(
                  'Give Feedback',
                  'How is Reflexion working for you?',
                  [
                    { text: 'Not helpful', style: 'destructive' },
                    { text: 'Could be better', style: 'cancel' },
                    { text: 'Really helpful!', onPress: () => Alert.alert('Thank you', 'Your feedback means a lot to us.') },
                  ],
                )
              }
            />
            <ActionRow label="Subscription & Billing" onPress={() => Alert.alert('Billing', 'Billing portal coming soon.')} />
          </>
        )}

        <TouchableOpacity onPress={() => void logout()} style={styles.logoutBtn}>
          <Text style={styles.logoutText}>Log out</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

function formatLanguage(value: string) {
  if (!value) return '';
  return value.slice(0, 1).toUpperCase() + value.slice(1);
}

function SectionHeader({ title }: { title: string }) {
  return <Text style={styles.sectionHeader}>{title}</Text>;
}

function SettingRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.row}>
      <Text style={styles.rowLabel}>{label}</Text>
      <Text style={styles.rowValue}>{value}</Text>
    </View>
  );
}

function ActionRow({ label, value, onPress }: { label: string; value?: string; onPress: () => void }) {
  return (
    <TouchableOpacity style={styles.row} onPress={onPress} activeOpacity={0.7}>
      <Text style={styles.rowLabel}>{label}</Text>
      <View style={styles.rowRight}>
        {value && <Text style={styles.rowValue}>{value}</Text>}
        <Feather name="chevron-right" size={16} color="#C4B9AF" />
      </View>
    </TouchableOpacity>
  );
}

function SwitchRow({
  disabled = false,
  label,
  value,
  onChange,
}: {
  disabled?: boolean;
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <View style={styles.row}>
      <Text style={styles.rowLabel}>{label}</Text>
      <Switch
        disabled={disabled}
        value={value}
        onValueChange={onChange}
        trackColor={{ false: '#D8CFC3', true: '#87566A' }}
        thumbColor="#FFFFFF"
      />
    </View>
  );
}

function PickerRow({ label, options, selected, onSelect }: {
  label: string;
  options: { value: string; label: string }[];
  selected: string;
  onSelect: (v: string) => void;
}) {
  return (
    <View style={styles.pickerBlock}>
      <Text style={styles.rowLabel}>{label}</Text>
      <View style={styles.pickerOptions}>
        {options.map(o => (
          <TouchableOpacity
            key={o.value}
            style={[styles.pill, selected === o.value && styles.pillActive]}
            onPress={() => onSelect(o.value)}
          >
            <Text style={[styles.pillText, selected === o.value && styles.pillTextActive]}>{o.label}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F8F3EC' },
  scroll: { flex: 1 },
  content: { paddingBottom: 60 },
  titleRow: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 8,
    gap: 12,
  },
  pageTitle: { fontSize: 26, fontWeight: '500', color: '#2B2522', fontFamily: 'Georgia' },
  loadingCard: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 16,
    borderWidth: 1,
    gap: 8,
    marginHorizontal: 20,
    marginTop: 20,
    padding: 24,
  },
  loadingTitle: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 20, fontWeight: '500' },
  loadingText: { color: '#756C64', fontSize: 14, lineHeight: 20, textAlign: 'center' },

  sectionHeader: {
    fontSize: 12,
    fontWeight: '600',
    color: '#A69C92',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    paddingHorizontal: 20,
    paddingTop: 28,
    paddingBottom: 8,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#F3EDE6',
  },
  rowLabel: { fontSize: 15, color: '#2B2522' },
  rowValue: { fontSize: 15, color: '#A69C92' },
  rowRight: { flexDirection: 'row', alignItems: 'center', gap: 6 },

  pickerBlock: {
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#F3EDE6',
  },
  pickerOptions: { flexDirection: 'row', gap: 8, marginTop: 10, flexWrap: 'wrap' },
  pill: {
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: 999,
    backgroundColor: '#F4F0EA',
    borderWidth: 1,
    borderColor: '#E7DED2',
  },
  pillActive: { backgroundColor: '#87566A', borderColor: '#87566A' },
  pillText: { fontSize: 13, color: '#756C64' },
  pillTextActive: { color: '#FFFFFF', fontWeight: '600' },
  saveBtn: {
    alignItems: 'center',
    backgroundColor: '#87566A',
    justifyContent: 'center',
    marginHorizontal: 20,
    marginTop: 14,
    minHeight: 46,
    borderRadius: 12,
  },
  saveBtnDisabled: {
    opacity: 0.7,
  },
  saveBtnText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '700',
  },

  logoutBtn: {
    margin: 20,
    marginTop: 36,
    padding: 16,
    backgroundColor: '#FFFFFF',
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#D8CFC3',
    alignItems: 'center',
  },
  logoutText: { color: '#87566A', fontSize: 15, fontWeight: '600' },
});
