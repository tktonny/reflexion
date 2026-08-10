import type { V1AlertSensitivity, V1SummaryFrequency, V1SummaryTime } from '../lib/v1Caregiver';

export const PUSH_NOTIFICATION_OPTIONS: { value: boolean; label: string }[] = [
  { value: true, label: 'Enable (recommended)' },
  { value: false, label: 'Disable' },
];

export const ALERT_SENSITIVITY_OPTIONS: { value: V1AlertSensitivity; label: string }[] = [
  { value: 'notify_me_about_everything', label: 'Notify me about everything' },
  { value: 'only_important_changes', label: 'Only important changes' },
  { value: 'only_urgent_alerts', label: 'Only urgent alerts' },
];

export const SUMMARY_TIME_OPTIONS: { value: V1SummaryTime; label: string }[] = [
  { value: '09:00', label: 'Morning push at 9am' },
  { value: '19:00', label: 'Evening push at 7pm' },
];

export const SUMMARY_FREQUENCY_OPTIONS: { value: V1SummaryFrequency; label: string }[] = [
  { value: 'immediately-after-each-session', label: 'After each completed session' },
  { value: 'daily-summary', label: 'Daily summary' },
  { value: 'weekly-summary', label: 'Weekly summary' },
  { value: 'off', label: 'No summaries' },
];
