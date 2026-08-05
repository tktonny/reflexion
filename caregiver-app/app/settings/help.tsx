import * as Linking from 'expo-linking';
import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout } from '../../src/components/AppUI';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';

export default function Help() {
  const router = useRouter();
  const [error, setError] = useState('');
  const openSupport = () => { void Linking.openURL('mailto:support@reflexion.care?subject=Reflexion%20support').catch(() => setError('Your mail app is unavailable. Email support@reflexion.care for help.')); };
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Help & Support" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Help & Support</Text><Text style={styles.copy}>For help with pairing, messages or an account, contact our support team.</Text><View style={styles.card}><Text style={styles.label}>support@reflexion.care</Text><Text style={styles.copy}>We’ll open your email app with a new support message.</Text></View>{error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}<PrimaryButton label="Email support" onPress={openSupport} /></ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, copy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 }, error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 }, card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.lg, borderWidth: 1, gap: spacing.sm, padding: spacing.lg }, label: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '700' } });
