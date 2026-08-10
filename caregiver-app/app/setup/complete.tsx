import { useRouter } from 'expo-router';
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { BotanicalCorner, BrandLockup } from '../../src/components/BrandLockup';
import { PrimaryButton, ScreenLayout } from '../../src/components/AppUI';
import { colors, contentColumn, fontFamily, fontSize, spacing } from '../../src/theme';

export default function SetupCompleteScreen() { const router = useRouter(); return <ScreenLayout contentContainerStyle={styles.content} scroll={false}><BrandLockup /><View style={styles.icon}><Text>✓</Text></View><Text accessibilityRole="header" style={styles.title}>Setup complete</Text><Text style={styles.subtitle}>Your household, device readiness, routines, notifications, product consent and optional research status can all be reviewed in Settings.</Text><PrimaryButton label="Go to Home" onPress={() => router.replace('/(tabs)')} /><BotanicalCorner side="left" /></ScreenLayout>; }
const styles = StyleSheet.create({ content: { flex: 1, justifyContent: 'center', position: 'relative' }, icon: { alignItems: 'center', backgroundColor: colors.status.greenBg, borderRadius: 999, height: 64, justifyContent: 'center', marginBottom: spacing.xl, marginTop: spacing.welcome, width: 64 }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.display, lineHeight: 40, textAlign: 'center' }, subtitle: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 25, marginBottom: spacing.xxl, marginTop: spacing.md, textAlign: 'center' } });
