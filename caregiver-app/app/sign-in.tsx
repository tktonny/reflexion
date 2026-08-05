import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { Alert, StyleSheet, Text, View } from 'react-native';

import { BrandLockup } from '../src/components/BrandLockup';
import { PrimaryButton, ScreenLayout, SecondaryButton, TertiaryButton } from '../src/components/AppUI';
import { Field } from '../src/components/Field';
import { validateSignIn } from '../src/lib/authValidation';
import { signInMessage } from '../src/lib/authMessages';
import { v1Login } from '../src/lib/v1Client';
import { V1ApiError } from '../src/lib/v1Client';
import { savePendingVerification } from '../src/lib/pendingVerification';
import { colors, fontFamily, fontSize, spacing } from '../src/theme';

type DeferredMethod = 'Phone' | 'Google' | 'Apple';

export default function SignInScreen() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errors, setErrors] = useState<{ identifier?: string; password?: string }>({});
  const [requestError, setRequestError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const showDeferredMethod = (method: DeferredMethod) => {
    const copy = method === 'Phone'
      ? 'Phone sign-in is not available during the current pilot. Please sign in using your email.'
      : method === 'Google'
        ? 'Google sign-in is not available during the current pilot. Please sign in using your email.'
        : 'Apple sign-in is not available during the current pilot. Please sign in using your email.';
    Alert.alert(`${method} sign-in unavailable`, copy, [{ text: 'Continue with email', onPress: () => { setErrors({}); setRequestError(''); } }]);
  };

  const signIn = async () => {
    const nextErrors = validateSignIn(email, password, 'email');
    setErrors(nextErrors);
    setRequestError('');
    if (Object.keys(nextErrors).length) return;
    setSubmitting(true);
    try {
      await v1Login(email.trim(), password);
      router.replace('/(tabs)');
    } catch (cause) {
      if (cause instanceof V1ApiError && cause.code === 'EMAIL_NOT_VERIFIED') {
        await savePendingVerification(email.trim().toLowerCase());
        router.replace({ pathname: '/account-verification', params: { email: email.trim().toLowerCase() } });
        return;
      }
      setRequestError(signInMessage(cause));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <BrandLockup />
      <Text accessibilityRole="header" style={styles.title}>Welcome back</Text>
      <Text style={styles.subtitle}>Sign in to continue caring with confidence.</Text>
      <Field error={errors.identifier} label="Email" keyboardType="email-address" autoComplete="email" onChangeText={(value) => { setEmail(value); setErrors((current) => ({ ...current, identifier: undefined })); }} placeholder="you@email.com" value={email} />
      <Field error={errors.password} label="Password" autoComplete="current-password" onChangeText={(value) => { setPassword(value); setErrors((current) => ({ ...current, password: undefined })); }} placeholder="Enter your password" secure value={password} />
      {requestError ? <Text accessibilityRole="alert" style={styles.requestError}>{requestError}</Text> : null}
      <PrimaryButton disabled={submitting} label={submitting ? 'Signing in…' : 'Sign in'} onPress={() => void signIn()} />
      <TertiaryButton label="Forgot password?" onPress={() => router.push('/forgot-password')} />
      <View style={styles.divider}><View style={styles.rule} /><Text style={styles.or}>or continue with</Text><View style={styles.rule} /></View>
      <View style={styles.deferredRow}>
        <SecondaryButton accessibilityLabel="Phone sign-in, unavailable during the pilot" label="Phone" onPress={() => showDeferredMethod('Phone')} />
        <SecondaryButton accessibilityLabel="Google sign-in, unavailable during the pilot" label="Google" onPress={() => showDeferredMethod('Google')} />
        <SecondaryButton accessibilityLabel="Apple sign-in, unavailable during the pilot" label="Apple" onPress={() => showDeferredMethod('Apple')} />
      </View>
      <View style={styles.create}><Text style={styles.createText}>Don’t have an account?</Text><TertiaryButton label="Create account" onPress={() => router.push('/create-account')} /></View>
      <Text style={styles.legal}>By continuing, you agree to the Terms of Service and Privacy Policy.</Text>
    </ScreenLayout>
  );
}

const styles = StyleSheet.create({
  content: { gap: spacing.lg, paddingTop: spacing.welcome },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, marginTop: spacing.xl },
  subtitle: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 25 },
  requestError: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 },
  divider: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, marginVertical: spacing.xs },
  rule: { backgroundColor: colors.border.default, flex: 1, height: 1 },
  or: { color: colors.text.secondary, fontSize: fontSize.caption },
  deferredRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  create: { alignItems: 'center', flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', marginTop: spacing.sm },
  createText: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 20 },
  legal: { color: colors.text.secondary, fontSize: fontSize.caption, lineHeight: 18, textAlign: 'center' },
});
