import * as Linking from 'expo-linking';
import * as Notifications from 'expo-notifications';
import { useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Alert, StyleSheet, Switch, Text, View } from 'react-native';

import { NOTIFICATION_TRIGGERS, type NotificationTrigger, type SessionSummaryFrequency } from '../../src/architecture/models';
import { useCaregiver } from '../../src/architecture/CaregiverContext';
import { AppHeader, PrimaryButton, ScreenLayout, SecondaryButton } from '../../src/components/AppUI';
import { getCaregiverProfileV1, updateCaregiverProfileV1, type V1NotificationTrigger } from '../../src/lib/v1Caregiver';
import { colors, contentColumn, fontFamily, fontSize, radius, spacing } from '../../src/theme';

const FREQUENCIES: { id: SessionSummaryFrequency; label: string }[] = [
  { id: 'immediately-after-each-session', label: 'Immediately after each session' },
  { id: 'daily-summary', label: 'Daily summary' },
  { id: 'weekly-summary', label: 'Weekly summary' },
  { id: 'off', label: 'Off' },
];

const DEFAULT_TRIGGERS: Record<V1NotificationTrigger, boolean> = Object.fromEntries(
  NOTIFICATION_TRIGGERS.map(({ id }) => [id, true]),
) as Record<V1NotificationTrigger, boolean>;

export default function NotificationSettings() {
  const router = useRouter();
  const { setNotificationsEnabled } = useCaregiver();
  const [enabled, setEnabled] = useState(true);
  const [frequency, setFrequency] = useState<SessionSummaryFrequency>('daily-summary');
  const [triggers, setTriggers] = useState(DEFAULT_TRIGGERS);
  const [busy, setBusy] = useState(true);
  const [saving, setSaving] = useState(false);
  const [denied, setDenied] = useState(false);
  const [settingsError, setSettingsError] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void getCaregiverProfileV1().then((profile) => {
      setEnabled(profile.notificationPreferences.pushNotificationsEnabled);
      setNotificationsEnabled(profile.notificationPreferences.pushNotificationsEnabled);
      setFrequency(profile.notificationPreferences.summaryFrequency || 'daily-summary');
      setTriggers({ ...DEFAULT_TRIGGERS, ...(profile.notificationPreferences.triggers || {}) });
    }).catch((cause) => setError(cause instanceof Error ? cause.message : 'Could not load notification preferences.')).finally(() => setBusy(false));
  }, []);

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      if (enabled) {
        const permission = await Notifications.requestPermissionsAsync();
        if (permission.status !== 'granted') {
          setDenied(true);
          return;
        }
      }
      await updateCaregiverProfileV1({ notificationPreferences: {
        pushNotificationsEnabled: enabled,
        summaryFrequency: frequency,
        triggers,
      } });
      setNotificationsEnabled(enabled);
      Alert.alert('Preferences saved', 'These notification choices are used in onboarding and Settings.', [{ text: 'Done', onPress: () => router.back() }]);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not save notification preferences.');
    } finally {
      setSaving(false);
    }
  };

  return <ScreenLayout contentContainerStyle={styles.content}>
      <AppHeader title="Notifications" onBack={() => router.back()} />
      <Text accessibilityRole="header" style={styles.title}>Notification preferences</Text>
      <Text style={styles.copy}>Choose the updates you would like to receive. The same canonical list is used throughout Reflexion.</Text>
      {busy ? <ActivityIndicator color={colors.accent} /> : null}
      {error ? <Text style={styles.error}>{error}</Text> : null}
      <View style={styles.card}>
        <View style={styles.row}><View style={styles.copyColumn}><Text style={styles.label}>App notifications</Text><Text style={styles.help}>Allow alerts from Reflexion on this phone.</Text></View><Switch accessibilityLabel="App notifications" value={enabled} onValueChange={setEnabled} trackColor={{ false: '#D5D9DB', true: colors.accent }} /></View>
        {NOTIFICATION_TRIGGERS.map(({ id, title }) => <View key={id} style={styles.row}><View style={styles.copyColumn}><Text style={styles.label}>{title}</Text><Text style={styles.help}>{triggerDescription(id)}</Text></View><Switch accessibilityLabel={title} value={triggers[id]} onValueChange={(value) => setTriggers((current) => ({ ...current, [id]: value }))} trackColor={{ false: '#D5D9DB', true: colors.accent }} /></View>)}
      </View>
      <View style={styles.card}><Text style={styles.label}>Conversation summary frequency</Text><Text style={styles.help}>Choose how often session summaries are delivered.</Text>{FREQUENCIES.map((item) => <SecondaryButton key={item.id} label={`${frequency === item.id ? '✓ ' : ''}${item.label}`} onPress={() => setFrequency(item.id)} />)}</View>
      {denied ? <View style={styles.card}><Text style={styles.help}>Notifications are blocked by your phone. Open Settings to allow them, then return here to save.</Text>{settingsError ? <Text accessibilityRole="alert" style={styles.error}>{settingsError}</Text> : null}<SecondaryButton label="Open phone settings" onPress={() => { setSettingsError(''); void Linking.openSettings().catch(() => setSettingsError('Phone settings could not be opened. Open your device Settings app, allow notifications for Reflexion, then return here.')); }} /></View> : null}
      {saving ? <ActivityIndicator color={colors.accent} /> : <PrimaryButton label="Save preferences" onPress={() => void save()} />}
  </ScreenLayout>;
}

function triggerDescription(trigger: NotificationTrigger) {
  switch (trigger) {
    case 'conversation-session-summary': return 'A summary is ready after a conversation.';
    case 'no-interaction-yet-today': return 'There has not been an interaction yet today.';
    case 'repeated-missed-interactions': return 'Several expected interactions were missed.';
    case 'recent-interaction-shorter-than-usual': return 'A recent interaction was shorter than usual.';
    case 'device-may-be-offline': return 'The Mirror may need a connection check.';
    case 'reminder-not-completed-or-unclear': return 'A routine response was missed or unclear.';
    case 'new-chat-reply': return 'A loved one replied in Chat.';
    case 'weekly-summary': return 'A weekly summary is ready.';
  }
}

const styles = StyleSheet.create({
  content: { gap: spacing.lg },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', marginTop: spacing.lg },
  copy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 },
  error: { color: colors.status.red, fontSize: fontSize.body },
  card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.lg, borderWidth: 1, gap: spacing.md, padding: spacing.lg },
  row: { alignItems: 'center', borderBottomColor: colors.border.subtle, borderBottomWidth: 1, flexDirection: 'row', gap: spacing.md, paddingVertical: spacing.md },
  copyColumn: { flex: 1 },
  label: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 22 },
  help: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 20, marginTop: 2 },
});
