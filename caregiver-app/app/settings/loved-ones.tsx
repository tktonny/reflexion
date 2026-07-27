import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { getStoredAuthSession } from '../../src/lib/authSession';
import { PatientEditModal } from '../../src/screens/settings/PatientEditModal';
import { SettingsPlaceholder } from '../../src/screens/settings/SettingsPlaceholder';
import { ActionRow, SectionHeader } from '../../src/screens/settings/SettingsRows';
import { SettingsSubPage } from '../../src/screens/settings/SettingsSubPage';
import { formatLanguage, parsePatientAge, resolveSettingsState, toPatientForm } from '../../src/screens/settings/helpers';
import type { PatientForm } from '../../src/screens/settings/types';
import { useCaregiverSettings, useSaveLovedOne } from '../../src/screens/settings/useCaregiverSettings';

/** A row's secondary line: what this loved one's setup actually is, not a bare language name. */
function describe(patient: { preferredLanguage: string; age: number }): string {
  const parts = [formatLanguage(patient.preferredLanguage)];
  if (patient.age > 0) parts.push(`${patient.age} years old`);
  return parts.join(' · ');
}

export default function LovedOnesSettingsScreen() {
  const router = useRouter();
  const session = getStoredAuthSession();
  const settings = useCaregiverSettings();
  const [editing, setEditing] = useState<PatientForm | null>(null);

  const save = useSaveLovedOne(() => setEditing(null));
  const state = resolveSettingsState({
    hasNurseId: Boolean(session?.userId),
    hasFailed: Boolean(settings.error),
    hasSettings: Boolean(settings.data),
    isLoading: settings.isLoading,
  });

  if (state !== 'ready' && state !== 'empty') {
    return (
      <SettingsSubPage title="Your loved ones">
        <SettingsPlaceholder onRetry={() => void settings.refetch()} state={state} />
      </SettingsSubPage>
    );
  }

  const patients = settings.data?.patients || [];

  function saveEditing() {
    if (!editing || save.isPending) return;
    const age = parsePatientAge(editing.age);
    if (age === null) return;
    const original = patients.find((patient) => patient.id === editing.id || patient.patientId === editing.id);
    if (!original) return;
    // The versions come from the row this form was opened on: v1 requires If-Match on both writes, so a
    // second phone editing the same loved one conflicts rather than silently winning.
    save.mutate({ ...original, ...editing, age });
  }

  return (
    <SettingsSubPage
      subtitle={patients.length
        ? 'Tap a name to change how Aria talks with them.'
        : 'Nobody is set up yet. Add the first person you want to check in on.'}
      title="Your loved ones"
    >
      <SectionHeader title={patients.length === 1 ? '1 person' : `${patients.length} people`} />
      {patients.map((patient) => (
        <ActionRow
          fallbackName={patient.name}
          imageUrl={patient.photoUrl}
          key={patient.id}
          // Falls back to the id's own wording rather than rendering an empty row: a nameless entry used to
          // show as an avatar, a gap, and a language, with nothing saying who it was.
          label={patient.name || 'Unnamed loved one'}
          onPress={() => setEditing(toPatientForm(patient))}
          value={describe(patient)}
        />
      ))}
      <ActionRow
        label="Add a loved one"
        onPress={() => router.push('/onboarding?mode=add-patient&returnTo=settings')}
      />

      <PatientEditModal
        isSaving={save.isPending}
        onChange={setEditing}
        onClose={() => setEditing(null)}
        onSave={saveEditing}
        patient={editing}
      />
    </SettingsSubPage>
  );
}
