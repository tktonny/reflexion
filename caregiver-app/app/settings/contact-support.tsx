import * as Linking from 'expo-linking';
import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout } from '../../src/components/AppUI';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';

export default function ContactSupportScreen() {
  const router = useRouter();
  const [error, setError] = useState('');
  const contact = () => { void Linking.openURL('mailto:support@reflexion.care?subject=Reflexion%20support').catch(() => setError('Your mail app is unavailable. Email support@reflexion.care for help.')); };
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Contact Support" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Contact Support</Text><Text style={styles.copy}>Tell us what happened and include the loved one or device involved. We will use the details only to respond to your request.</Text><View style={styles.card}><Text style={styles.label}>support@reflexion.care</Text><Text style={styles.copy}>Support can help with accounts, pairing, device connection and family-message delivery.</Text></View>{error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}<PrimaryButton label="Email support" onPress={contact} /></ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, copy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 }, error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 }, card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.sm, padding: spacing.lg }, label: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '700' } });
