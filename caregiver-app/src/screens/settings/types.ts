export type AlertSensitivity =
  | 'notify_me_about_everything'
  | 'only_important_changes'
  | 'only_urgent_alerts';
export type SummaryTime = '09:00' | '19:00';
export type Gender = 'male' | 'female' | 'other';
export type Language = 'english' | 'mandarin' | 'other';
export type KeyTopic = 'family' | 'food' | 'travel' | 'work' | 'others';

export type SettingsPatient = {
  id: string;
  patientId?: string;
  name: string;
  phoneNumber: string;
  age: number;
  gender: Gender | '';
  preferredLanguage: Language | '';
  usualWakeTime: string;
  speechOrHearingConditions: string;
  photoUrl?: string;
  keyTopics: KeyTopic[];
  keyTopicsOtherText: string;
};

export type SettingsConfig = {
  nurseId: string;
  caregiverName: string;
  email: string;
  phoneNumber: string;
  pushNotificationsEnabled: boolean;
  alertSensitivity: AlertSensitivity;
  preferredDailySummaryTime: SummaryTime;
  storeSessionSummaries: boolean;
  patients: SettingsPatient[];
};

export type PatientForm = Omit<SettingsPatient, 'age'> & { age: string };
export type SettingsState = 'ready' | 'signed-out' | 'loading' | 'failed' | 'empty';
