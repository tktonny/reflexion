import React from 'react';
import { View } from 'react-native';
import { ALERT_SENSITIVITY_OPTIONS, PUSH_NOTIFICATION_OPTIONS, SUMMARY_FREQUENCY_OPTIONS, SUMMARY_TIME_OPTIONS } from '../../data/notificationOptions';
import { Label, OptionGrid } from './fields';
import type { NotificationForm } from './types';

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
        options={PUSH_NOTIFICATION_OPTIONS}
        selected={notifications.pushNotificationsEnabled}
        onSelect={(pushNotificationsEnabled) =>
          setNotifications((current) => ({ ...current, pushNotificationsEnabled }))
        }
      />

      <Label>Alert sensitivity</Label>
      <OptionGrid
        groupLabel="Alert sensitivity"
        options={ALERT_SENSITIVITY_OPTIONS}
        selected={notifications.alertSensitivity}
        onSelect={(alertSensitivity) =>
          setNotifications((current) => ({ ...current, alertSensitivity }))
        }
      />

      <Label>Preferred daily summary time</Label>
      <OptionGrid
        groupLabel="Preferred daily summary time"
        options={SUMMARY_TIME_OPTIONS}
        selected={notifications.preferredDailySummaryTime}
        onSelect={(preferredDailySummaryTime) =>
          setNotifications((current) => ({ ...current, preferredDailySummaryTime }))
        }
      />

      <Label>Summary frequency</Label>
      <OptionGrid
        groupLabel="Summary frequency"
        options={SUMMARY_FREQUENCY_OPTIONS}
        selected={notifications.summaryFrequency}
        onSelect={(summaryFrequency) =>
          setNotifications((current) => ({ ...current, summaryFrequency }))
        }
      />
    </View>
  );
}
