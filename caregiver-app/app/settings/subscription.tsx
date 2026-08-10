import * as Linking from 'expo-linking';
import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, SecondaryButton } from '../../src/components/AppUI';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';

export default function SubscriptionScreen() {
  const router = useRouter();
  const [error, setError] = useState('');
  const contact = () => { void Linking.openURL('mailto:support@reflexion.care?subject=Reflexion%20subscription').catch(() => setError('Your mail app is unavailable. Email support@reflexion.care for subscription help.')); };
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Subscription" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Subscription</Text><Text style={styles.copy}>Plan and billing details are shown here when subscription management is enabled for your account.</Text><View style={styles.card}><Text style={styles.label}>Subscription management is not available in this pilot</Text><Text style={styles.copy}>No subscription or payment data is read or changed by this screen. Contact support if you need help with billing.</Text></View>{error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}<PrimaryButton label="Contact support" onPress={contact} /><SecondaryButton label="Payment method" onPress={() => router.push('/settings/payment-method')} /></ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, copy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 }, error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 }, card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, padding: spacing.lg }, label: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 22 } });
