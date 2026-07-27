import { useQueryClient } from '@tanstack/react-query';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useCallback } from 'react';
import { Alert, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { clearStoredAuthSession, getStoredAuthSession } from '../../src/lib/authSession';
import { clearCaregiverCache } from '../../src/lib/queryKeys';
import { useTabBarClearance } from '../../src/lib/useTabBarClearance';
import { v1Logout } from '../../src/lib/v1Client';
import { SettingsPlaceholder } from '../../src/screens/settings/SettingsPlaceholder';
import { ActionRow, SectionHeader } from '../../src/screens/settings/SettingsRows';
import { resolveSettingsState } from '../../src/screens/settings/helpers';
import type { AlertSensitivity, SettingsConfig, SummaryTime } from '../../src/screens/settings/types';
import { useCaregiverSettings } from '../../src/screens/settings/useCaregiverSettings';
import { colors, contentColumn, fontFamily, fontSize, MIN_TOUCH_TARGET, radius, spacing } from '../../src/theme';

/*
 * Settings is a hub: every row navigates, and nothing here holds an unsaved value.
 *
 * It used to be one long form. A single "Save changes" button sat between the privacy switch and "Export my
 * data" and wrote the caregiver's name, phone, three notification preferences and that switch — while the
 * loved-one rows immediately above it saved themselves somewhere else entirely. Its position told you none of
 * that, so pressing it was a guess about what would be written.
 *
 * Each group now owns a page, which is what makes "Save" mean something: the button on a sub-page can only be
 * about the fields above it, because that page has nothing else on it. This screen's job is to show what each
 * setting currently IS, so a caregiver can answer most questions without opening anything.
 */

const SENSITIVITY_SUMMARY: Record<AlertSensitivity, string> = {
  notify_me_about_everything: 'Everything',
  only_important_changes: 'Important changes',
  only_urgent_alerts: 'Urgent only',
};

const TIME_SUMMARY: Record<SummaryTime, string> = { '09:00': '9am', '19:00': '7pm' };

function notificationSummary(config: SettingsConfig): string {
  const sensitivity = SENSITIVITY_SUMMARY[config.alertSensitivity] || 'Important changes';
  const time = TIME_SUMMARY[config.preferredDailySummaryTime] || '7pm';
  // Push being off is the one fact worth leading with — it changes whether the phone buzzes at all.
  return config.pushNotificationsEnabled ? `${sensitivity} · ${time}` : `Push off · ${sensitivity}`;
}

/** Names, not a count: "Grandma, Nana" answers the question a count only hints at. */
function lovedOnesSummary(config: SettingsConfig): string {
  const names = config.patients.map((patient) => patient.name?.trim()).filter(Boolean) as string[];
  if (!names.length) return 'Nobody added yet';
  if (names.length <= 2) return names.join(', ');
  return `${names.slice(0, 2).join(', ')} +${names.length - 2}`;
}

export default function SettingsScreen() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const session = getStoredAuthSession();
  const bottomClearance = useTabBarClearance();
  const settings = useCaregiverSettings();

  const { refetch } = settings;
  useFocusEffect(
    // Coming back from a sub-page must show the value that was just saved, and React Query is configured
    // never to refetch on its own (staleTime: Infinity), so every screen asks explicitly.
    useCallback(() => {
      if (session?.userId) void refetch();
    }, [refetch, session?.userId]),
  );

  const state = resolveSettingsState({
    hasNurseId: Boolean(session?.userId),
    hasFailed: Boolean(settings.error),
    hasSettings: Boolean(settings.data),
    isLoading: settings.isLoading,
  });

  async function signOut() {
    await Promise.all([clearStoredAuthSession(), v1Logout()]);
    // Before navigating: the next caregiver on this phone must not read the previous one's cached loved
    // ones, alerts or statuses. React Query holds them under gcTime: Infinity.
    clearCaregiverCache(queryClient);
    router.replace('/sign-in');
  }

  function confirmSignOut() {
    Alert.alert('Sign out?', 'You will need your email and password to get back in.', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Sign out', style: 'destructive', onPress: () => void signOut() },
    ]);
  }

  const config = settings.data;

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <View style={styles.header}>
        <Text accessibilityRole="header" maxFontSizeMultiplier={1.3} style={styles.title}>Settings</Text>
      </View>

      <ScrollView contentContainerStyle={[styles.content, { paddingBottom: bottomClearance }]}>
        {state !== 'ready' && state !== 'empty' ? (
          <SettingsPlaceholder onRetry={() => void settings.refetch()} state={state} />
        ) : (
          <>
            <SectionHeader title="You" />
            <ActionRow
              label="Your account"
              onPress={() => router.push('/settings/account')}
              value={config?.caregiverName || session?.name || ''}
            />
            <ActionRow
              label="Notifications"
              onPress={() => router.push('/settings/notifications')}
              value={config ? notificationSummary(config) : ''}
            />

            <SectionHeader title="Care" />
            <ActionRow
              label="Your loved ones"
              onPress={() => router.push('/settings/loved-ones')}
              value={config ? lovedOnesSummary(config) : ''}
            />
            <ActionRow
              label="Mirrors"
              onPress={() => router.push('/mirror-management')}
            />

            <SectionHeader title="More" />
            <ActionRow label="Privacy & data" onPress={() => router.push('/settings/privacy')} />
            <ActionRow label="Help & support" onPress={() => router.push('/settings/support')} />

            <TouchableOpacity
              accessibilityLabel="Sign out"
              accessibilityRole="button"
              onPress={confirmSignOut}
              style={styles.signOutButton}
            >
              <Text style={styles.signOutText}>Sign out</Text>
            </TouchableOpacity>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { backgroundColor: colors.surface.page, flex: 1 },
  header: { paddingHorizontal: spacing.xl, paddingTop: spacing.lg, paddingBottom: spacing.sm },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.display },
  // paddingBottom is supplied at render time from useTabBarClearance; a literal here was shorter than the bar.
  content: { ...contentColumn, paddingTop: spacing.sm },
  signOutButton: {
    alignItems: 'center',
    borderColor: colors.border.strong,
    borderRadius: radius.lg,
    borderWidth: 1,
    justifyContent: 'center',
    marginHorizontal: spacing.xl,
    marginTop: spacing.xxl,
    minHeight: MIN_TOUCH_TARGET,
    paddingVertical: spacing.md,
  },
  signOutText: { color: colors.error.text, fontSize: fontSize.subheading, fontWeight: '600' },
});
