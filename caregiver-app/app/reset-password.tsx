import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useMemo, useState } from 'react';
import { Alert, StyleSheet, Text } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, TertiaryButton } from '../src/components/AppUI';
import { BrandLockup } from '../src/components/BrandLockup';
import { Field } from '../src/components/Field';
import { validatePasswordPair } from '../src/lib/authValidation';
import { MIN_PASSWORD_LENGTH, passwordResetMessage } from '../src/lib/authMessages';
import { resetPasswordV1 } from '../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, spacing } from '../src/theme';

export default function ResetPasswordScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ token?: string }>();
  const token = useMemo(() => Array.isArray(params.token) ? params.token[0] : params.token || '', [params.token]);
  const [password, setPassword] = useState('');
  const [repeat, setRepeat] = useState('');
  const [errors, setErrors] = useState<{ password?: string; repeatPassword?: string }>({});
  const [requestError, setRequestError] = useState('');
  const [working, setWorking] = useState(false);

  const reset = async () => {
    const nextErrors = validatePasswordPair(password, repeat);
    setErrors(nextErrors);
    setRequestError('');
    if (!token) { setRequestError('This reset link is missing. Request a new reset code and try again.'); return; }
    if (Object.keys(nextErrors).length) return;
    setWorking(true);
    try {
      await resetPasswordV1(token, password);
      Alert.alert('Password updated', 'Sign in with your new password.', [{ text: 'Sign in', onPress: () => router.replace('/sign-in') }]);
    } catch (cause) {
      setRequestError(passwordResetMessage(cause));
    } finally {
      setWorking(false);
    }
  };

  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <AppHeader onBack={() => router.back()} />
      <BrandLockup compact />
      <Text accessibilityRole="header" style={styles.title}>Create new password</Text>
      <Text style={styles.subtitle}>New passwords must be at least {MIN_PASSWORD_LENGTH} characters. Active sessions will be signed out after the change.</Text>
      <Field error={errors.password} helperText={`At least ${MIN_PASSWORD_LENGTH} characters.`} label="New password" onChangeText={(value) => { setPassword(value); setErrors((current) => ({ ...current, password: undefined })); }} placeholder={`At least ${MIN_PASSWORD_LENGTH} characters`} secure value={password} />
      <Field error={errors.repeatPassword} label="Repeat password" onChangeText={(value) => { setRepeat(value); setErrors((current) => ({ ...current, repeatPassword: undefined })); }} placeholder="Enter it again" secure value={repeat} />
      {requestError ? <Text accessibilityRole="alert" style={styles.requestError}>{requestError}</Text> : null}
      <PrimaryButton disabled={working} label={working ? 'Updating…' : 'Reset password'} onPress={() => void reset()} />
      <TertiaryButton label="Back to sign in" onPress={() => router.replace('/sign-in')} />
    </ScreenLayout>
  );
}

const styles = StyleSheet.create({
  content: { gap: spacing.lg },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, marginTop: spacing.xl },
  subtitle: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 },
  requestError: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 },
});
