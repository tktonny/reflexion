import * as Linking from 'expo-linking';
import * as Notifications from 'expo-notifications';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { useCaregiver } from '../../src/architecture/CaregiverContext';
import { AppHeader, PrimaryButton, ScreenLayout, SelectionButton, SecondaryButton } from '../../src/components/AppUI';
import { ALERT_SENSITIVITY_OPTIONS, PUSH_NOTIFICATION_OPTIONS, SUMMARY_FREQUENCY_OPTIONS, SUMMARY_TIME_OPTIONS } from '../../src/data/notificationOptions';
import { getCaregiverProfileV1, updateCaregiverProfileV1, type V1AlertSensitivity, type V1NotificationTrigger, type V1SummaryFrequency, type V1SummaryTime } from '../../src/lib/v1Caregiver';
import { isDemoMode } from '../../src/demo/demoMode';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';

const DEFAULT_TRIGGERS: Record<V1NotificationTrigger, boolean> = {
  'conversation-session-summary': true,
  'no-interaction-yet-today': true,
  'repeated-missed-interactions': true,
  'recent-interaction-shorter-than-usual': true,
  'device-may-be-offline': true,
  'reminder-not-completed-or-unclear': true,
  'weekly-summary': true,
};

export default function NotificationSettings() {
  const router = useRouter();
  const { setNotificationsEnabled, setSetupStatus } = useCaregiver();
  const [enabled, setEnabled] = useState(false);
  const [alertSensitivity, setAlertSensitivity] = useState<V1AlertSensitivity>('notify_me_about_everything');
  const [summaryTime, setSummaryTime] = useState<V1SummaryTime>('19:00');
  const [summaryFrequency, setSummaryFrequency] = useState<V1SummaryFrequency>('daily-summary');
  const [triggers, setTriggers] = useState<Record<V1NotificationTrigger, boolean>>(DEFAULT_TRIGGERS);
  const [permissionStatus, setPermissionStatus] = useState<Notifications.PermissionStatus | null>(null);
  const [permissionCanAskAgain, setPermissionCanAskAgain] = useState(true);
  const [busy, setBusy] = useState(true);
  const [saving, setSaving] = useState(false);
  const [denied, setDenied] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    void (async () => {
      try {
        const profile = await getCaregiverProfileV1();
        const permission = isDemoMode()
          ? { status: 'granted' as Notifications.PermissionStatus, canAskAgain: true }
          : await Notifications.getPermissionsAsync();
        const preferences = profile.notificationPreferences;
        setPermissionStatus(permission.status);
        setPermissionCanAskAgain(permission.canAskAgain);
        const canNotify = permission.status === 'granted';
        setEnabled(preferences.pushNotificationsEnabled && canNotify);
        setDenied(preferences.pushNotificationsEnabled && !canNotify);
        setNotificationsEnabled(preferences.pushNotificationsEnabled && canNotify);
        setAlertSensitivity(preferences.alertSensitivity);
        setSummaryTime(preferences.preferredDailySummaryTime);
        setSummaryFrequency(preferences.summaryFrequency || 'daily-summary');
        setTriggers(preferences.triggers || DEFAULT_TRIGGERS);
      } catch {
        setError('We could not load notification preferences. Check your connection and try again.');
      } finally {
        setBusy(false);
      }
    })();
  }, [setNotificationsEnabled]);

  useFocusEffect(React.useCallback(() => {
    if (isDemoMode()) return undefined;
    let active = true;
    void Notifications.getPermissionsAsync().then((permission) => {
      if (!active) return;
      setPermissionStatus(permission.status);
      setPermissionCanAskAgain(permission.canAskAgain);
      if (permission.status === 'granted') setDenied(false);
    }).catch(() => undefined);
    return () => { active = false; };
  }, []));

  const toggleNotifications = async (nextValue: boolean) => {
    setError('');
    if (isDemoMode()) {
      setEnabled(nextValue);
      setDenied(false);
      setPermissionStatus('granted' as Notifications.PermissionStatus);
      setNotificationsEnabled(nextValue);
      return;
    }
    if (!nextValue) {
      setEnabled(false); setDenied(false); setNotificationsEnabled(false); return;
    }
    try {
      const current = permissionStatus ? { status: permissionStatus, canAskAgain: permissionCanAskAgain } : await Notifications.getPermissionsAsync();
      setPermissionStatus(current.status); setPermissionCanAskAgain(current.canAskAgain);
      if (current.status === 'granted') {
        setEnabled(true); setDenied(false); setNotificationsEnabled(true); return;
      }
      if (current.canAskAgain === false) {
        setEnabled(false); setDenied(true); setNotificationsEnabled(false); return;
      }
      const requested = await Notifications.requestPermissionsAsync();
      setPermissionStatus(requested.status); setPermissionCanAskAgain(requested.canAskAgain);
      const granted = requested.status === 'granted';
      setEnabled(granted); setDenied(!granted); setNotificationsEnabled(granted);
    } catch (cause) {
      setEnabled(false); setDenied(true); setNotificationsEnabled(false); setError(cause instanceof Error ? cause.message : 'Notifications could not be enabled.');
    }
  };

  const save = async () => {
    setSaving(true); setError(''); setDenied(false);
    try {
      if (enabled && permissionStatus !== 'granted') {
        setDenied(true);
        return;
      }
      await updateCaregiverProfileV1({ notificationPreferences: {
        pushNotificationsEnabled: enabled,
        alertSensitivity,
        preferredDailySummaryTime: summaryTime,
        summaryFrequency,
        triggers: { ...(triggers ?? {}) },
      } });
      setNotificationsEnabled(enabled);
      setSetupStatus('notifications', 'complete');
      router.back();
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'We could not save notification preferences. Check your connection and try again.'); }
    finally { setSaving(false); }
  };

  return <ScreenLayout contentContainerStyle={styles.content}>
    <AppHeader title="Notification Preferences" onBack={() => router.back()} />
    <Text accessibilityRole="header" style={styles.title}>Notification Preferences</Text>
    <Text style={styles.copy}>Stay informed with factual updates about your loved one and connected device.</Text>
    {busy ? <ActivityIndicator color={colors.accent} /> : null}
    {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
    <View style={styles.card}><Text style={styles.label}>Push notifications</Text><Text style={styles.help}>Allow Reflexion to tell you when an update may need your attention.</Text>{PUSH_NOTIFICATION_OPTIONS.map((option) => <SelectionButton key={String(option.value)} label={option.label} onPress={() => { void toggleNotifications(option.value); }} selected={enabled === option.value} />)}</View>
    <View style={styles.card}><Text style={styles.label}>Alert sensitivity</Text><Text style={styles.help}>Use the same alert choices selected during setup.</Text>{ALERT_SENSITIVITY_OPTIONS.map((option) => <SelectionButton key={option.value} label={option.label} onPress={() => setAlertSensitivity(option.value)} selected={alertSensitivity === option.value} />)}</View>
    <View style={styles.card}><Text style={styles.label}>Preferred daily summary time</Text><Text style={styles.help}>Choose when a daily summary is delivered.</Text>{SUMMARY_TIME_OPTIONS.map((option) => <SelectionButton key={option.value} label={option.label} onPress={() => setSummaryTime(option.value)} selected={summaryTime === option.value} />)}</View>
    <View style={styles.card}><Text style={styles.label}>Summary frequency</Text><Text style={styles.help}>Choose how often summary updates are sent.</Text>{SUMMARY_FREQUENCY_OPTIONS.map((option) => <SelectionButton key={option.value} label={option.label} onPress={() => setSummaryFrequency(option.value)} selected={summaryFrequency === option.value} />)}</View>
    {denied ? <View style={styles.card}><Text style={styles.label}>Notifications are off</Text><Text style={styles.help}>{permissionCanAskAgain ? 'Reflexion needs notification permission before it can send updates.' : 'Your phone has blocked notification permission for Reflexion. Open phone settings to allow it, then return here.'}</Text><SecondaryButton label="Open phone settings" onPress={() => { void Linking.openSettings().catch(() => setError('Phone settings could not be opened. Open your device Settings app and allow notifications for Reflexion.')); }} /></View> : null}
    {saving ? <ActivityIndicator color={colors.accent} /> : <PrimaryButton label="Save preferences" onPress={() => void save()} />}
  </ScreenLayout>;
}

const styles = StyleSheet.create({
  content: { gap: spacing.lg, minWidth: 0 },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, marginTop: spacing.lg, minWidth: 0 },
  copy: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 24, minWidth: 0 },
  error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 },
  card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, minWidth: 0, padding: spacing.lg },
  label: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 22, minWidth: 0 },
  help: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 20, marginTop: 2, minWidth: 0 },
});
