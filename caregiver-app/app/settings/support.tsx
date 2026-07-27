import { useRouter } from 'expo-router';
import React from 'react';
import { ActionRow, SectionHeader, SettingsGroup } from '../../src/screens/settings/SettingsRows';
import { SettingsSubPage } from '../../src/screens/settings/SettingsSubPage';

/** Pure navigation, so no save button — nothing on this page holds an unsaved value. */
export default function SupportSettingsScreen() {
  const router = useRouter();

  return (
    <SettingsSubPage subtitle="If something looks wrong, start here." title="Help & support">
      <SectionHeader title="Learn" />
      <SettingsGroup>
        <ActionRow label="FAQ & guide" onPress={() => router.push('/faq')} />
      </SettingsGroup>

      <SectionHeader title="Talk to us" />
      <SettingsGroup>
        <ActionRow label="Chat with support" onPress={() => router.push('/chatbot')} />
        <ActionRow label="Send feedback" onPress={() => router.push('/feedback')} />
      </SettingsGroup>
    </SettingsSubPage>
  );
}
