// Form shapes shared by the onboarding route and its step components.

export type Relationship = 'parent' | 'sibling' | 'spouse' | 'inlaw' | 'grandpa' | 'grandma' | 'other';
export type Gender = 'male' | 'female' | 'other';
export type PreferredLanguage = 'english' | 'mandarin' | 'other';
export type Topic = 'family' | 'food' | 'travel' | 'work' | 'others';
export type AlertSensitivity =
  | 'notify_me_about_everything'
  | 'only_important_changes'
  | 'only_urgent_alerts';
export type SummaryTime = '09:00' | '19:00';

export type AccountForm = {
  name: string;
  email: string;
  password: string;
  /** Stored separately from the national number so every phone field has the same contract. */
  countryCode: string;
  phoneNumber: string;
  relationshipToElderly: Relationship;
};

export type PatientForm = {
  name: string;
  phoneNumber: string;
  age: string;
  gender: Gender;
  preferredLanguage: PreferredLanguage;
  usualWakeTime: string;
  speechOrHearingConditions: string;
  photoUrl: string;
  keyTopics: Topic[];
  keyTopicsOtherText: string;
  mirrorName: string;
  mirrorPairingCode: string;
  timezone: string;
};

export type NotificationForm = {
  pushNotificationsEnabled: boolean;
  alertSensitivity: AlertSensitivity;
  preferredDailySummaryTime: SummaryTime;
};
