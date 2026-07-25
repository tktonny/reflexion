import React from 'react';
import { View } from 'react-native';
import { Label, OptionGrid } from './fields';
import type { AlertSensitivity, NotificationForm, SummaryTime } from './types';

const ALERT_OPTIONS: { value: AlertSensitivity; label: string }[] = [
  { value: 'notify_me_about_everything', label: 'Notify me about everything' },
  { value: 'only_important_changes', label: 'Only important changes' },
  { value: 'only_urgent_alerts', label: 'Only urgent alerts' },
];

const SUMMARY_OPTIONS: { value: SummaryTime; label: string }[] = [
  { value: '09:00', label: 'Morning push at 9am' },
  { value: '19:00', label: 'Evening push at 7pm' },
];

export function NotificationStep({
  notifications,
  setNotifications,
}: {
  notifications: NotificationForm;
  setNotifications: React.Dispatch<React.SetStateAction<NotificationForm>>;
}) {
  return (
    <View>
      <Label>Push notifications</Label>
      <OptionGrid
        groupLabel="Push notifications"
        options={[
          { value: true, label: 'Enable (recommended)' },
          { value: false, label: 'Disable' },
        ]}
        selected={notifications.pushNotificationsEnabled}
        onSelect={(pushNotificationsEnabled) =>
          setNotifications((current) => ({ ...current, pushNotificationsEnabled }))
        }
      />

      <Label>Alert sensitivity</Label>
      <OptionGrid
        groupLabel="Alert sensitivity"
        options={ALERT_OPTIONS}
        selected={notifications.alertSensitivity}
        onSelect={(alertSensitivity) =>
          setNotifications((current) => ({ ...current, alertSensitivity }))
        }
      />

      <Label>Preferred daily summary time</Label>
      <OptionGrid
        groupLabel="Preferred daily summary time"
        options={SUMMARY_OPTIONS}
        selected={notifications.preferredDailySummaryTime}
        onSelect={(preferredDailySummaryTime) =>
          setNotifications((current) => ({ ...current, preferredDailySummaryTime }))
        }
      />
    </View>
  );
}
