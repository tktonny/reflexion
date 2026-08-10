import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { Alert, Image, StyleSheet, Text, View, useWindowDimensions } from 'react-native';

import { PrimaryButton, ScreenLayout, SecondaryButton, SelectionButton, TertiaryButton } from '../src/components/AppUI';
import { Field, PhoneField } from '../src/components/Field';
import { normalizePhone, validateSignIn } from '../src/lib/authValidation';
import { signInMessage } from '../src/lib/authMessages';
import { v1Login } from '../src/lib/v1Client';
import { enterDemoMode, isDemoFeatureEnabled } from '../src/demo/demoMode';
import { colors, fontFamily, fontSize, scaleSize, spacing } from '../src/theme';

type DeferredMethod = 'Google' | 'Apple';
type IdentifierMethod = 'email' | 'phone';

export default function SignInScreen() {
  const router = useRouter();
  const { width } = useWindowDimensions();
  const [method, setMethod] = useState<IdentifierMethod>('email');
  const [email, setEmail] = useState('');
  const [countryCode, setCountryCode] = useState('+65');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [password, setPassword] = useState('');
  const [errors, setErrors] = useState<{ identifier?: string; password?: string }>({});
  const [requestError, setRequestError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const showDeferredMethod = (method: DeferredMethod) => {
    const copy = method === 'Google'
        ? 'Google sign-in is not available during the current pilot. Please sign in using your email.'
        : 'Apple sign-in is not available during the current pilot. Please sign in using your email.';
    Alert.alert(`${method} sign-in unavailable`, copy, [{ text: 'Continue with email', onPress: () => { setErrors({}); setRequestError(''); } }]);
  };

  const signIn = async () => {
    const identifier = method === 'email' ? email.trim() : normalizePhone(countryCode, phoneNumber);
    const nextErrors = validateSignIn(identifier, password, method);
    setErrors(nextErrors);
    setRequestError('');
    if (Object.keys(nextErrors).length) return;
    setSubmitting(true);
    try {
      await v1Login(identifier, password);
      router.replace('/(tabs)');
    } catch (cause) {
      setRequestError(signInMessage(cause));
    } finally {
      setSubmitting(false);
    }
  };

  const logoWidth = Math.min(width * 0.47, scaleSize(180));
  const branchWidth = Math.min(width * 0.23, scaleSize(90));

  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <View pointerEvents="none" style={styles.artLayer}>
        <Image accessibilityElementsHidden importantForAccessibility="no-hide-descendants" resizeMode="contain" source={require('../assets/auth/botanical-left.png')} style={[styles.topBranch, { height: branchWidth * (650 / 240), width: branchWidth }]} />
        <Image accessibilityElementsHidden importantForAccessibility="no-hide-descendants" resizeMode="contain" source={require('../assets/auth/botanical-right.png')} style={[styles.bottomBranch, { height: branchWidth * (470 / 250), width: branchWidth }]} />
      </View>
      <View style={styles.foreground}>
        <Image accessibilityElementsHidden importantForAccessibility="no-hide-descendants" resizeMode="contain" source={require('../assets/auth/reflexion-logo.png')} style={[styles.logo, { height: logoWidth * (260 / 450), width: logoWidth }]} />
        <Text accessibilityRole="header" style={styles.title}>Welcome back</Text>
        <Text style={styles.subtitle}>Sign in to continue caring with confidence.</Text>
        <View accessibilityLabel="Sign-in method" style={styles.methodRow}>
          <View style={styles.methodOption}><SelectionButton label="Email" selected={method === 'email'} onPress={() => { setMethod('email'); setErrors({}); setRequestError(''); }} /></View>
          <View style={styles.methodOption}><SelectionButton label="Phone" selected={method === 'phone'} onPress={() => { setMethod('phone'); setErrors({}); setRequestError(''); }} /></View>
        </View>
        {method === 'email' ? <Field error={errors.identifier} label="Email" keyboardType="email-address" autoComplete="email" onChangeText={(value) => { setEmail(value); setErrors((current) => ({ ...current, identifier: undefined })); }} placeholder="you@email.com" value={email} /> : <PhoneField countryCode={countryCode} error={errors.identifier} helperText="Use the country code and phone number saved to your Reflexion account." label="Phone number" onCountryCodeChange={setCountryCode} onPhoneNumberChange={(value) => { setPhoneNumber(value); setErrors((current) => ({ ...current, identifier: undefined })); }} phoneNumber={phoneNumber} />}
        <Field error={errors.password} label="Password" autoComplete="current-password" onChangeText={(value) => { setPassword(value); setErrors((current) => ({ ...current, password: undefined })); }} placeholder="Enter your password" secure value={password} />
        {requestError ? <Text accessibilityRole="alert" style={styles.requestError}>{requestError}</Text> : null}
        <PrimaryButton disabled={submitting} label={submitting ? 'Signing in…' : 'Sign in'} onPress={() => void signIn()} />
        {isDemoFeatureEnabled() ? <View style={styles.demo}><TertiaryButton label="Enter Demo App" onPress={() => { void enterDemoMode().then(() => router.replace('/demo')); }} /><Text style={styles.demoCopy}>Development-only local fixtures. Nothing is synced.</Text></View> : null}
        <TertiaryButton label="Forgot password?" onPress={() => router.push('/forgot-password')} />
        <View style={styles.divider}><View style={styles.rule} /><Text style={styles.or}>or continue with</Text><View style={styles.rule} /></View>
        <View style={styles.deferredRow}>
          <SecondaryButton accessibilityLabel="Google sign-in, unavailable during the pilot" label="Google" onPress={() => showDeferredMethod('Google')} />
          <SecondaryButton accessibilityLabel="Apple sign-in, unavailable during the pilot" label="Apple" onPress={() => showDeferredMethod('Apple')} />
        </View>
        <View style={styles.create}><Text style={styles.createText}>Don’t have an account?</Text><TertiaryButton label="Create account" onPress={() => router.push('/create-account')} /></View>
        <Text style={styles.legal}>By continuing, you agree to the Terms of Service and Privacy Policy.</Text>
      </View>
    </ScreenLayout>
  );
}

const styles = StyleSheet.create({
  content: { gap: 0, overflow: 'hidden', paddingBottom: spacing.xxl, paddingTop: spacing.lg, position: 'relative' },
  artLayer: { ...StyleSheet.absoluteFill, overflow: 'hidden' },
  foreground: { gap: spacing.lg, minWidth: 0, paddingTop: spacing.xl },
  logo: { alignSelf: 'center' },
  topBranch: { left: -spacing.md, position: 'absolute', top: spacing.xs },
  bottomBranch: { bottom: -spacing.xl, position: 'absolute', right: -spacing.md },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, marginTop: spacing.xl },
  subtitle: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 25 },
  requestError: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 },
  divider: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, marginVertical: spacing.xs },
  methodRow: { flexDirection: 'row', gap: spacing.sm, minWidth: 0 },
  methodOption: { flex: 1, minWidth: 0 },
  rule: { backgroundColor: colors.border.default, flex: 1, height: 1 },
  or: { color: colors.text.secondary, fontSize: fontSize.caption },
  deferredRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  create: { alignItems: 'center', flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', marginTop: spacing.sm },
  createText: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 20 },
  legal: { color: colors.text.secondary, fontSize: fontSize.caption, lineHeight: 18, textAlign: 'center' },
  demo: { alignItems: 'center', gap: spacing.xs, minWidth: 0 },
  demoCopy: { color: colors.text.secondary, fontSize: fontSize.caption, lineHeight: 18, textAlign: 'center' },
});
