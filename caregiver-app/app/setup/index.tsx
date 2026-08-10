import { useRouter } from 'expo-router';
import React, { useEffect } from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { useCaregiver } from '../../src/architecture/CaregiverContext';
import { SETUP_CATEGORIES } from '../../src/architecture/models';
import { AppHeader, PrimaryButton, ScreenLayout, SetupProgressCard, TertiaryButton } from '../../src/components/AppUI';
import { BotanicalCorner, BrandLockup } from '../../src/components/BrandLockup';
import { colors, contentColumn, fontFamily, fontSize, spacing } from '../../src/theme';

export default function SetupOverviewScreen() {
  const router = useRouter(); const { setup, loadSetupProgress, setupLoading, setupError, setSetupStatus } = useCaregiver();
  useEffect(() => { void loadSetupProgress(); }, [loadSetupProgress]);
  const complete = SETUP_CATEGORIES.filter((category) => setup[category.id] === 'complete' || setup[category.id] === 'not-applicable').length;
  const openCategory = (category: (typeof SETUP_CATEGORIES)[number]['id']) => {
    const destinations: Record<typeof category, string> = {
      household: '/setup/household',
      'pair-device': '/device/select',
      'language-accessibility': '/settings/language',
      routines: '/settings/routines',
      notifications: '/settings/notifications',
      'consent-control': '/settings/consent',
      'research-participation': '/research/overview',
    };
    // Research is optional and the current API only exposes an active study when an invitation exists.
    // Do not turn a not-invited state into an unfinished setup task just by opening the page.
    if (category !== 'research-participation') setSetupStatus(category, 'in-progress');
    router.push(destinations[category]);
  };
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Setup" onBack={() => router.back()} /><BrandLockup compact /><Text accessibilityRole="header" style={styles.title}>Setup overview</Text><Text style={styles.subtitle}>Let’s get everything set up. You can pause and return anytime.</Text>{setupLoading ? <Text style={styles.status}>Loading your saved progress…</Text> : null}{setupError ? <Text accessibilityRole="alert" style={styles.error}>{setupError}</Text> : null}<View style={styles.progress}><Text style={styles.progressTitle}>Setup steps (7)</Text><Text style={styles.progressValue}>{complete} of 7 sections complete</Text></View><View style={styles.cards}>{SETUP_CATEGORIES.map((category) => <SetupProgressCard key={category.id} description={category.description} status={setup[category.id]} title={category.title} onPress={() => openCategory(category.id)} />)}</View><PrimaryButton label="Review setup" onPress={() => router.push('/setup/review')} /><TertiaryButton label="Set up later" onPress={() => router.replace('/(tabs)')} /><BotanicalCorner /></ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg, position: 'relative' }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 34, marginTop: spacing.md }, subtitle: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 25 }, status: { color: colors.text.secondary, fontSize: fontSize.caption }, error: { color: colors.status.red, fontSize: fontSize.body, lineHeight: 22 }, progress: { backgroundColor: '#EEF3E9', borderRadius: 18, gap: 4, marginTop: spacing.md, padding: spacing.xl }, progressTitle: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '700' }, progressValue: { color: colors.text.secondary, fontSize: fontSize.body }, cards: { gap: spacing.md, marginTop: spacing.md } });
