import * as Linking from 'expo-linking';
import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout } from '../../src/components/AppUI';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';

export default function PaymentMethodScreen() {
  const router = useRouter();
  const [error, setError] = useState('');
  const contact = () => { void Linking.openURL('mailto:support@reflexion.care?subject=Reflexion%20payment%20method').catch(() => setError('Your mail app is unavailable. Email support@reflexion.care for payment help.')); };
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Payment Method" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Payment Method</Text><View style={styles.card}><Text style={styles.label}>No payment method is stored in this pilot</Text><Text style={styles.copy}>This screen does not collect or display card details. Contact support for account-specific billing help.</Text></View>{error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}<PrimaryButton label="Contact support" onPress={contact} /><PrimaryButton label="Back to subscription" onPress={() => router.replace('/settings/subscription')} /></ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, copy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 }, error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 }, card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, padding: spacing.lg }, label: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 22 } });
