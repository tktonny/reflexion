import { useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { Alert } from 'react-native';
import { getStoredAuthSession } from '../../src/lib/authSession';
import { SettingsPlaceholder } from '../../src/screens/settings/SettingsPlaceholder';
import { ActionRow, SectionHeader, SwitchRow } from '../../src/screens/settings/SettingsRows';
import { SettingsSubPage } from '../../src/screens/settings/SettingsSubPage';
import { resolveSettingsState } from '../../src/screens/settings/helpers';
import { useCaregiverSettings, useSaveCaregiverProfile } from '../../src/screens/settings/useCaregiverSettings';

export default function PrivacySettingsScreen() {
  const router = useRouter();
  const session = getStoredAuthSession();
  const settings = useCaregiverSettings();
  const [storeSummaries, setStoreSummaries] = useState(true);

  useEffect(() => {
    if (!settings.data) return;
    setStoreSummaries(settings.data.storeSessionSummaries);
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
      <SettingsSubPage title="Privacy & data">
        <SettingsPlaceholder onRetry={() => void settings.refetch()} state={state} />
      </SettingsSubPage>
    );
  }

  return (
    <SettingsSubPage
      isSaving={save.isPending}
      onSave={() => save.mutate({ storeSessionSummaries: storeSummaries })}
      title="Privacy & data"
    >
      <SectionHeader title="Conversations" />
      <SwitchRow label="Keep written summaries" onChange={setStoreSummaries} value={storeSummaries} />
      {/* Says what turning it off costs, since the honest trade-off is not obvious from the label: the
          check-ins still happen and the status still works, you just lose the readable record afterwards. */}
      <SectionHeader title="Turning this off means daily check-ins still run and the status still updates — you just will not be able to read back what was said." />

      <SectionHeader title="Your data" />
      <ActionRow
        label="Export my data"
        onPress={() => Alert.alert('Export', 'Data export is coming in a later version.')}
      />
    </SettingsSubPage>
  );
}
