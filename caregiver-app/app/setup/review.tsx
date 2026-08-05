import { useRouter } from 'expo-router';
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { useCaregiver } from '../../src/architecture/CaregiverContext';
import { SETUP_CATEGORIES } from '../../src/architecture/models';
import { AppHeader, PrimaryButton, ScreenLayout, SetupProgressCard, TertiaryButton } from '../../src/components/AppUI';
import { colors, contentColumn, fontFamily, fontSize, spacing } from '../../src/theme';

export default function SetupReviewScreen() {
  const router = useRouter(); const { setup, loadSetupProgress } = useCaregiver();
  React.useEffect(() => { void loadSetupProgress(); }, [loadSetupProgress]);
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Setup" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Review setup</Text><Text style={styles.subtitle}>You can finish now and return to any category whenever you need to.</Text><View style={styles.cards}>{SETUP_CATEGORIES.map((category) => <SetupProgressCard key={category.id} description="Edit this category" status={setup[category.id]} title={category.title} onPress={() => router.push(`/setup/${category.id}`)} />)}</View><PrimaryButton label="Finish setup" onPress={() => router.replace('/setup/complete')} /><TertiaryButton label="Set up later" onPress={() => router.replace('/(tabs)')} /></ScreenLayout>;
}
const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', marginTop: spacing.xl }, subtitle: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 25 }, cards: { gap: spacing.md, marginBottom: spacing.lg, marginTop: spacing.md } });
