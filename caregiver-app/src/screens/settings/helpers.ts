import type { Gender, KeyTopic, Language, PatientForm, SettingsPatient, SettingsState } from './types';

/**
 * Which of the four situations the settings load is in. Settings we already hold win over everything else,
 * so a background refetch that fails never yanks a working form away from the caregiver.
 */
export function resolveSettingsState({
  hasFailed,
  hasNurseId,
  hasSettings,
  isLoading,
}: {
  hasFailed: boolean;
  hasNurseId: boolean;
  hasSettings: boolean;
  isLoading: boolean;
}): SettingsState {
  if (hasSettings) return 'ready';
  // A disabled query never reports isLoading, which is why the signed-out case needs its own branch.
  if (!hasNurseId) return 'signed-out';
  if (isLoading) return 'loading';
  if (hasFailed) return 'failed';
  return 'empty';
}

export function toPatientForm(patient: SettingsPatient): PatientForm {
  return {
    ...patient,
    patientId: patient.patientId || patient.id,
    age: String(patient.age || ''),
    speechOrHearingConditions: patient.speechOrHearingConditions || '',
    photoUrl: patient.photoUrl || '',
    keyTopics: normalizeKeyTopics(patient.keyTopics),
    keyTopicsOtherText: patient.keyTopicsOtherText || '',
  };
}

export function normalizeSettingsPatient(patient: Partial<SettingsPatient>): SettingsPatient {
  return {
    id: patient.id || patient.patientId || '',
    patientId: patient.patientId || patient.id || '',
    name: patient.name || '',
    phoneNumber: patient.phoneNumber || '',
    age: Number(patient.age || 0),
    gender: normalizeGender(patient.gender),
    preferredLanguage: normalizeLanguage(patient.preferredLanguage),
    usualWakeTime: patient.usualWakeTime || '',
    speechOrHearingConditions: patient.speechOrHearingConditions || '',
    photoUrl: patient.photoUrl || '',
    keyTopics: normalizeKeyTopics(patient.keyTopics),
    keyTopicsOtherText: patient.keyTopicsOtherText || '',
  };
}

export function normalizeGender(value: unknown): Gender | '' {
  const normalized = typeof value === 'string' ? value.toLowerCase() : '';
  return normalized === 'male' || normalized === 'female' || normalized === 'other' ? normalized : '';
}

export function normalizeLanguage(value: unknown): Language | '' {
  const normalized = typeof value === 'string' ? value.toLowerCase() : '';
  return normalized === 'english' || normalized === 'mandarin' || normalized === 'other' ? normalized : '';
}

export function normalizeKeyTopics(value: unknown): KeyTopic[] {
  if (!Array.isArray(value)) return [];

  return value
    .map((item) => (typeof item === 'string' ? item.trim().toLowerCase() : ''))
    .filter(isKeyTopic);
}

export function isKeyTopic(value: string): value is KeyTopic {
  return value === 'family' || value === 'food' || value === 'travel' || value === 'work' || value === 'others';
}

export function isTopicSelected(topics: unknown, topic: KeyTopic) {
  return normalizeKeyTopics(topics).includes(topic);
}

export function formatLanguage(value: string) {
  if (!value) return '';
  return value.slice(0, 1).toUpperCase() + value.slice(1);
}

export function getInitials(name: string) {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return '?';
  return parts.slice(0, 2).map(part => part[0]?.toUpperCase()).join('');
}

/**
 * Parses the age field, returning null when it is not a usable whole age.
 *
 * `Number('')` is 0 and 0 is an integer, so a plain `Number.isInteger(Number(value))` check let a cleared
 * age field through and PATCHed `age: 0`. The server rejects that (it requires 1-130), but the caregiver
 * deserves to be told before the round trip rather than seeing a generic save failure.
 */
export function parsePatientAge(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const age = Number(trimmed);
  if (!Number.isInteger(age) || age < 1 || age > 130) return null;
  return age;
}
