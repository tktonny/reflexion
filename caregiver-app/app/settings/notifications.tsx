import { useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { getStoredAuthSession } from '../../src/lib/authSession';
import { SettingsPlaceholder } from '../../src/screens/settings/SettingsPlaceholder';
import { PickerRow, SectionHeader, SwitchRow } from '../../src/screens/settings/SettingsRows';
import { SettingsSubPage } from '../../src/screens/settings/SettingsSubPage';
import { resolveSettingsState } from '../../src/screens/settings/helpers';
import type { AlertSensitivity, SummaryTime } from '../../src/screens/settings/types';
import { useCaregiverSettings, useSaveCaregiverProfile } from '../../src/screens/settings/useCaregiverSettings';

const SUMMARY_TIMES = [
  { value: '09:00', label: 'Morning (9am)' },
  { value: '19:00', label: 'Evening (7pm)' },
];

const SENSITIVITIES = [
  { value: 'notify_me_about_everything', label: 'Notify me about everything' },
  { value: 'only_important_changes', label: 'Only important changes' },
  { value: 'only_urgent_alerts', label: 'Only urgent alerts' },
];

export default function NotificationSettingsScreen() {
  const router = useRouter();
  const session = getStoredAuthSession();
  const settings = useCaregiverSettings();
  const [pushEnabled, setPushEnabled] = useState(true);
  const [summaryTime, setSummaryTime] = useState<SummaryTime>('19:00');
  const [sensitivity, setSensitivity] = useState<AlertSensitivity>('only_important_changes');

  useEffect(() => {
    if (!settings.data) return;
    setPushEnabled(settings.data.pushNotificationsEnabled);
    setSummaryTime(settings.data.preferredDailySummaryTime);
    setSensitivity(settings.data.alertSensitivity);
  }, [settings.data]);

  const save = useSaveCaregiverProfile(() => router.back());
  const state = resolveSettingsState({
    hasNurseId: Boolean(session?.userId),
    hasFailed: Boolean(settings.error),
    hasSettings: Boolean(settings.data),
    isLoading: settings.isLoading,
  });

  if (state !== 'ready' && state !== 'empty') {
    return (
      <SettingsSubPage title="Notifications">
        <SettingsPlaceholder onRetry={() => void settings.refetch()} state={state} />
      </SettingsSubPage>
    );
  }

  return (
    <SettingsSubPage
      isSaving={save.isPending}
      onSave={() => save.mutate({
        notificationPreferences: {
          pushNotificationsEnabled: pushEnabled,
          preferredDailySummaryTime: summaryTime,
          alertSensitivity: sensitivity,
        },
      })}
      // Says plainly that the list is not conditional on any of this, because the honest answer to "will I
      // miss something if push does not work" is no.
      subtitle="Alerts always appear in the Notifications tab. These settings only change what reaches your phone."
      title="Notifications"
    >
      <SectionHeader title="On this phone" />
      <SwitchRow label="Enable push notifications" onChange={setPushEnabled} value={pushEnabled} />

      <SectionHeader title="Daily summary" />
      <PickerRow
        label="When to send it"
        onSelect={(value) => setSummaryTime(value as SummaryTime)}
        options={SUMMARY_TIMES}
        selected={summaryTime}
      />

      <SectionHeader title="How much to tell you" />
      <PickerRow
        label="Alert sensitivity"
        onSelect={(value) => setSensitivity(value as AlertSensitivity)}
        options={SENSITIVITIES}
        selected={sensitivity}
      />
    </SettingsSubPage>
  );
}
