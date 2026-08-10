import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { StyleSheet, Text } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, TertiaryButton } from '../src/components/AppUI';
import { BrandLockup } from '../src/components/BrandLockup';
import { Field } from '../src/components/Field';
import { validateEmail } from '../src/lib/authValidation';
import { passwordResetRequestMessage } from '../src/lib/authMessages';
import { requestPasswordResetV1 } from '../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, spacing } from '../src/theme';

export default function ForgotPasswordScreen() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [requestError, setRequestError] = useState('');
  const [working, setWorking] = useState(false);

  const sendCode = async () => {
    const validation = validateEmail(email);
    setError(validation || '');
    setRequestError('');
    if (validation) return;
    setWorking(true);
    try {
      await requestPasswordResetV1(email);
      router.push({ pathname: '/reset-verification', params: { email: email.trim().toLowerCase() } });
    } catch (cause) {
      setRequestError(passwordResetRequestMessage(cause));
    } finally {
      setWorking(false);
    }
  };

  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <AppHeader onBack={() => router.back()} />
      <BrandLockup compact />
      <Text accessibilityRole="header" style={styles.title}>Forgot password?</Text>
      <Text style={styles.subtitle}>Enter your email and we’ll request a six-digit reset code.</Text>
      <Field error={error} label="Email" keyboardType="email-address" autoComplete="email" onChangeText={(value) => { setEmail(value); setError(''); }} placeholder="you@email.com" value={email} />
      <Text style={styles.note}>For your security, the same response is shown whether an account exists.</Text>
      {requestError ? <Text accessibilityRole="alert" style={styles.requestError}>{requestError}</Text> : null}
      <PrimaryButton disabled={working} label={working ? 'Requesting…' : 'Send code'} onPress={() => void sendCode()} />
      <TertiaryButton label="Back to sign in" onPress={() => router.replace('/sign-in')} />
    </ScreenLayout>
  );
}

const styles = StyleSheet.create({
  content: { gap: spacing.lg },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, marginTop: spacing.xl },
  subtitle: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 },
  note: { color: colors.text.secondary, fontSize: fontSize.caption, lineHeight: 18 },
  requestError: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 },
});
