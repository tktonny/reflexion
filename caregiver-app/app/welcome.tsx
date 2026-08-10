import { useRouter } from 'expo-router';
import React, { useEffect } from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { BotanicalCorner, BrandLockup } from '../src/components/BrandLockup';
import { PrimaryButton, ScreenLayout, TertiaryButton } from '../src/components/AppUI';
import { useCaregiver } from '../src/architecture/CaregiverContext';
import { colors, contentColumn, fontFamily, fontSize, spacing } from '../src/theme';

export default function WelcomeScreen() {
  const router = useRouter();
  const { loadSetupProgress } = useCaregiver();
  useEffect(() => { void loadSetupProgress(); }, [loadSetupProgress]);
  return <ScreenLayout contentContainerStyle={styles.content} scroll={false}><BrandLockup /><View style={styles.copy}><Text accessibilityRole="header" style={styles.title}>Welcome to Reflexion</Text><Text style={styles.subtitle}>A calm way to stay connected to the people that matter, wherever you are.</Text></View><View style={styles.promise}><Text style={styles.promiseTitle}>Care. Connected.</Text><Text style={styles.promiseText}>Set up the parts that are useful to you now. You can return to any category later.</Text></View><PrimaryButton label="Start setup" onPress={() => router.replace('/setup')} /><TertiaryButton label="Set up later" onPress={() => router.replace('/(tabs)')} /><BotanicalCorner /></ScreenLayout>;
}

const styles = StyleSheet.create({
  content: { flex: 1, justifyContent: 'center', position: 'relative' }, copy: { marginTop: spacing.welcome }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.display, fontWeight: '500', lineHeight: 42, textAlign: 'center' }, subtitle: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 25, marginTop: spacing.md, textAlign: 'center' }, promise: { backgroundColor: '#EEF3E9', borderRadius: 18, gap: 6, marginBottom: spacing.xxl, marginTop: spacing.editorial, padding: spacing.xl }, promiseTitle: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500' }, promiseText: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 21 },
});
