import { useRouter } from 'expo-router';
import React from 'react';
import { Alert, StyleSheet, Text, View } from 'react-native';

import { AppHeader, ScreenLayout, SettingsRow, TertiaryButton } from '../../src/components/AppUI';
import { v1Logout } from '../../src/lib/v1Client';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';

export default function AccountScreen() {
  const router = useRouter();
  const signOut = () => Alert.alert('Sign out?', 'You will need your email and password to sign in again.', [
    { text: 'Cancel', style: 'cancel' },
    { text: 'Sign out', style: 'destructive', onPress: () => void v1Logout().finally(() => router.replace('/sign-in')) },
  ]);
  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <AppHeader title="Account" onBack={() => router.back()} />
      <Text accessibilityRole="header" style={styles.title}>Account</Text>
      <Text style={styles.subtitle}>Manage your personal information and sign-in security.</Text>
      <View style={styles.group}>
        <SettingsRow icon="user" label="Edit personal information" value="Name and relationship" onPress={() => router.push('/settings/account/personal')} />
        <SettingsRow icon="mail" label="Change email" value="Verified by email link" onPress={() => router.push('/settings/account/email')} />
        <SettingsRow icon="smartphone" label="Change phone number" value="Verification code by SMS" onPress={() => router.push('/settings/account/phone')} />
        <SettingsRow icon="key" label="Change password" value="Use your current password" onPress={() => router.push('/settings/account/password')} />
        <SettingsRow icon="shield" label="Sign-in methods" value="Email and password" onPress={() => router.push('/settings/account/sign-in-methods')} />
      </View>
      <TertiaryButton label="Sign out" onPress={signOut} />
    </ScreenLayout>
  );
}

const styles = StyleSheet.create({
  content: { gap: spacing.lg },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg },
  subtitle: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 },
  group: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, overflow: 'hidden' },
});
