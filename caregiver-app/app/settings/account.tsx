import { useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { SettingsPlaceholder } from '../../src/screens/settings/SettingsPlaceholder';
import { InputRow, SectionHeader, SettingRow } from '../../src/screens/settings/SettingsRows';
import { SettingsSubPage } from '../../src/screens/settings/SettingsSubPage';
import { resolveSettingsState } from '../../src/screens/settings/helpers';
import { useCaregiverSettings, useSaveCaregiverProfile } from '../../src/screens/settings/useCaregiverSettings';
import { getStoredAuthSession } from '../../src/lib/authSession';

export default function AccountSettingsScreen() {
  const router = useRouter();
  const session = getStoredAuthSession();
  const settings = useCaregiverSettings();
  const [name, setName] = useState('');
  const [phoneNumber, setPhoneNumber] = useState('');

  useEffect(() => {
    if (!settings.data) return;
    setName(settings.data.caregiverName);
    setPhoneNumber(settings.data.phoneNumber);
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
      <SettingsSubPage title="Your account">
        <SettingsPlaceholder onRetry={() => void settings.refetch()} state={state} />
      </SettingsSubPage>
    );
  }

  return (
    <SettingsSubPage
      isSaving={save.isPending}
      onSave={() => save.mutate({ name: name.trim(), phoneNumber: phoneNumber.trim() })}
      subtitle="This is you — the person we contact about your loved ones."
      title="Your account"
    >
      <SectionHeader title="Account" />
      <InputRow label="Name" onChangeText={setName} value={name} />
      {/* Read-only: the email is the account's login identity, and v1 deliberately refuses to change it here
          because moving it needs a verified flow — email uniqueness is only enforced per tenant, so an
          unverified change could create the duplicate-account state that used to break sign-in. */}
      <SettingRow label="Email" value={settings.data?.email || ''} />
      <InputRow keyboardType="phone-pad" label="Phone" onChangeText={setPhoneNumber} value={phoneNumber} />
    </SettingsSubPage>
  );
}
