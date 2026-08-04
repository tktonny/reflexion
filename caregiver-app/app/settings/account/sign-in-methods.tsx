import { useRouter } from 'expo-router';
import React from 'react';
import { Alert, StyleSheet, Text, View } from 'react-native';

import { AppHeader, ScreenLayout, SettingsRow } from '../../../src/components/AppUI';
import { colors, fontFamily, fontSize, radius, spacing } from '../../../src/theme';

function showUnavailable(method: 'Phone' | 'Google' | 'Apple', continueWithEmail: () => void) {
  const copy = method === 'Phone'
    ? 'Phone sign-in is not available during the current pilot. Please sign in using your email.'
    : method === 'Google'
      ? 'Google sign-in is not available during the current pilot. Please sign in using your email.'
      : 'Apple sign-in is not available during the current pilot. Please sign in using your email.';
  Alert.alert(`${method} sign-in unavailable`, copy, [{ text: 'Continue with email', onPress: continueWithEmail }]);
}

export default function Methods() {
  const router = useRouter();
  const continueWithEmail = () => router.replace('/sign-in');
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Sign-in methods" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Sign-in methods</Text><Text style={styles.copy}>Email and password are configured for this pilot. Phone, Google and Apple remain visible but are not available yet.</Text><View style={styles.group}><SettingsRow icon="mail" label="Email and password" value="Configured" /><SettingsRow icon="smartphone" label="Phone" value="Unavailable during pilot" onPress={() => showUnavailable('Phone', continueWithEmail)} /><SettingsRow icon="chrome" label="Google" value="Unavailable during pilot" onPress={() => showUnavailable('Google', continueWithEmail)} /><SettingsRow icon="command" label="Apple" value="Unavailable during pilot" onPress={() => showUnavailable('Apple', continueWithEmail)} /></View></ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, copy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 }, group: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, overflow: 'hidden' } });
