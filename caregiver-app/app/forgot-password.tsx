import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { Image, StyleSheet, Text, View, useWindowDimensions } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, TertiaryButton } from '../src/components/AppUI';
import { Field } from '../src/components/Field';
import { validateEmail } from '../src/lib/authValidation';
import { passwordResetRequestMessage } from '../src/lib/authMessages';
import { requestPasswordResetV1 } from '../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, scaleSize, spacing } from '../src/theme';

export default function ForgotPasswordScreen() {
  const router = useRouter();
  const { width } = useWindowDimensions();
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

  const logoWidth = Math.min(width * 0.46, scaleSize(178));
  const envelopeWidth = Math.min(width * 0.64, scaleSize(248));
  const branchWidth = Math.min(width * 0.3, scaleSize(116));

  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <View pointerEvents="none" style={styles.artLayer}>
        <Image accessibilityElementsHidden importantForAccessibility="no-hide-descendants" resizeMode="contain" source={require('../assets/auth/botanical-right.png')} style={[styles.bottomBranch, { height: branchWidth * (470 / 250), width: branchWidth }]} />
      </View>
      <View style={styles.foreground}>
        <AppHeader onBack={() => router.back()} />
        <Image accessibilityElementsHidden importantForAccessibility="no-hide-descendants" resizeMode="contain" source={require('../assets/auth/reflexion-logo.png')} style={[styles.logo, { height: logoWidth * (260 / 450), width: logoWidth }]} />
        <Image accessibilityElementsHidden importantForAccessibility="no-hide-descendants" resizeMode="contain" source={require('../assets/auth/envelope-heart.png')} style={[styles.envelope, { height: envelopeWidth * (280 / 320), width: envelopeWidth }]} />
        <Text accessibilityRole="header" style={styles.title}>Forgot password?</Text>
        <Text style={styles.subtitle}>Enter your email and we’ll request a six-digit reset code.</Text>
        <Field error={error} label="Email" keyboardType="email-address" autoComplete="email" onChangeText={(value) => { setEmail(value); setError(''); }} placeholder="you@email.com" value={email} />
        <Text style={styles.note}>For your security, the same response is shown whether an account exists.</Text>
        {requestError ? <Text accessibilityRole="alert" style={styles.requestError}>{requestError}</Text> : null}
        <PrimaryButton disabled={working} label={working ? 'Requesting…' : 'Send code'} onPress={() => void sendCode()} />
        <TertiaryButton label="Back to sign in" onPress={() => router.replace('/sign-in')} />
      </View>
    </ScreenLayout>
  );
}

const styles = StyleSheet.create({
  content: { gap: 0, overflow: 'hidden', paddingBottom: spacing.xxl, paddingTop: 0, position: 'relative' },
  artLayer: { ...StyleSheet.absoluteFill, overflow: 'hidden' },
  foreground: { gap: spacing.lg, minWidth: 0 },
  logo: { alignSelf: 'center', marginTop: spacing.xs },
  envelope: { alignSelf: 'center', marginTop: spacing.sm },
  bottomBranch: { bottom: -spacing.xxl, position: 'absolute', right: -spacing.md },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, marginTop: spacing.xl },
  subtitle: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 },
  note: { color: colors.text.secondary, fontSize: fontSize.caption, lineHeight: 18 },
  requestError: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 },
});
