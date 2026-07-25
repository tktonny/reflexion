// Pure helpers for the onboarding funnel: form seeding, input formatting, step copy, and the per-step
// validator. Nothing here touches state or the network, so each one is directly testable.

import type { AccountForm, PatientForm } from './types';

// Hermes builds without full Intl timezone data throw on resolvedOptions().timeZone. blankPatient runs
// at mount (useState initializer), so an unguarded throw crashes the whole onboarding/sign-up screen.
export function deviceTimeZone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || 'Asia/Singapore';
  } catch {
    return 'Asia/Singapore';
  }
}

export const blankPatient = (index: number): PatientForm => ({
  name: '',
  phoneNumber: '',
  age: '',
  gender: 'male',
  preferredLanguage: 'english',
  usualWakeTime: '07:30',
  speechOrHearingConditions: '',
  photoUrl: '',
  keyTopics: ['family'],
  keyTopicsOtherText: '',
  mirrorName: `Mirror ${index + 1}`,
  mirrorPairingCode: '',
  timezone: deviceTimeZone(),
});

export function formatPairingInput(value: string) {
  const digits = value.replace(/\D/g, '').slice(0, 6);
  return digits.length > 3 ? `${digits.slice(0, 3)} ${digits.slice(3)}` : digits;
}

export function getStepSubtitle(step: number, patientCount: number) {
  if (step === 1) return 'Set up the caregiver account details.';
  if (step === 2) return `Add one or more elderly profiles. Current total: ${patientCount}.`;
  if (step === 3) return 'Link each profile to the pairing code displayed on their mirror.';
  return 'Choose alert and daily summary preferences.';
}

/** Returns the caregiver-facing problem with the current step, or '' when it is ready to submit. */
export function validateStep(step: number, account: AccountForm, patients: PatientForm[]) {
  if (step === 1) {
    if (!account.name.trim() || !account.email.trim() || !account.password || !account.phoneNumber.trim()) {
      return 'Enter your name, email, password, and phone number.';
    }
    if (!account.email.includes('@')) {
      return 'Enter a valid email address.';
    }
    if (account.password.length < 8) {
      return 'Use a password with at least 8 characters.';
    }
  }

  if (step === 2) {
    for (const patient of patients) {
      if (!patient.name.trim() || !patient.age.trim() || !patient.usualWakeTime.trim()) {
        return 'Each elderly profile needs a name, age, and usual wake time.';
      }
      const age = Number(patient.age);
      if (!Number.isInteger(age) || age < 1 || age > 130) {
        return 'Enter a valid age for each elderly profile.';
      }
      if (patient.keyTopics.length === 0) {
        return 'Choose at least one topic for each elderly profile.';
      }
      if (patient.keyTopics.includes('others') && !patient.keyTopicsOtherText.trim()) {
        return 'Add free text for any profile using the Others topic.';
      }
    }
  }

  if (step === 3) {
    if (patients.some((patient) => !patient.mirrorName.trim())) {
      return 'Give each mirror a name.';
    }
    if (patients.some((patient) => patient.mirrorPairingCode.trim() && patient.mirrorPairingCode.replace(/\D/g, '').length !== 6)) {
      return 'Pairing codes must be 6 digits.';
    }
  }

  return '';
}
