import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { Image, StyleSheet, Text, View, useWindowDimensions } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, TertiaryButton } from '../src/components/AppUI';
import { Field, PhoneField } from '../src/components/Field';
import { validateCreateAccount, type FieldErrors, normalizePhone } from '../src/lib/authValidation';
import { registrationMessage } from '../src/lib/authMessages';
import { clearPendingVerification } from '../src/lib/pendingVerification';
import { PasswordRequirements } from '../src/components/PasswordRequirements';
import { setV1Session } from '../src/lib/v1AuthSession';
import { registerCaregiverV1 } from '../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, scaleSize, spacing } from '../src/theme';

export default function CreateAccountScreen() {
  const router = useRouter();
  const { width } = useWindowDimensions();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [countryCode, setCountryCode] = useState('+65');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [password, setPassword] = useState('');
  const [repeatPassword, setRepeatPassword] = useState('');
  const [errors, setErrors] = useState<FieldErrors>({});
  const [requestError, setRequestError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const create = async () => {
    const nextErrors = validateCreateAccount({ name, email, countryCode, phoneNumber, password, repeatPassword });
    setErrors(nextErrors);
    setRequestError('');
    if (Object.keys(nextErrors).length) return;
    setSubmitting(true);
    try {
      const registration = await registerCaregiverV1({
        name: name.trim(),
        email: email.trim(),
        password,
        ...(phoneNumber.trim() ? { phoneNumber: normalizePhone(countryCode, phoneNumber) } : {}),
      });
      if (registration.state === 'authenticated' && registration.accessToken && registration.refreshToken && registration.actor) {
        await clearPendingVerification();
        await setV1Session({
          accessToken: registration.accessToken,
          refreshToken: registration.refreshToken,
          accessTokenExpiresAt: registration.accessTokenExpiresAt,
          refreshTokenExpiresAt: registration.refreshTokenExpiresAt,
          actor: registration.actor,
        });
        router.replace('/welcome');
        return;
      }
      // The current release does not require email verification after registration. Keep this branch
      // defensive for a misconfigured server, but never send a caregiver into the removed verification route.
      setRequestError('Your account was created. Sign in to continue setup.');
    } catch (cause) {
      setRequestError(registrationMessage(cause));
    } finally {
      setSubmitting(false);
    }
  };

  const clear = (key: keyof FieldErrors) => setErrors((current) => ({ ...current, [key]: undefined }));
  const logoWidth = Math.min(width * 0.47, scaleSize(180));
  const branchWidth = Math.min(width * 0.23, scaleSize(90));
  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <View pointerEvents="none" style={styles.artLayer}>
        <Image accessibilityElementsHidden importantForAccessibility="no-hide-descendants" resizeMode="contain" source={require('../assets/auth/botanical-left.png')} style={[styles.leftBranch, { height: branchWidth * (650 / 240), width: branchWidth }]} />
        <Image accessibilityElementsHidden importantForAccessibility="no-hide-descendants" resizeMode="contain" source={require('../assets/auth/botanical-right.png')} style={[styles.bottomBranch, { height: branchWidth * (470 / 250), width: branchWidth }]} />
      </View>
      <View style={styles.foreground}>
        <AppHeader onBack={() => router.back()} />
        <Image accessibilityElementsHidden importantForAccessibility="no-hide-descendants" resizeMode="contain" source={require('../assets/auth/reflexion-logo.png')} style={[styles.logo, { height: logoWidth * (260 / 450), width: logoWidth }]} />
        <Text accessibilityRole="header" style={styles.title}>Create your account</Text>
        <Text style={styles.subtitle}>Join Reflexion to support the people you love with ease.</Text>
        <Field error={errors.name} label="Preferred name" onChangeText={(value) => { setName(value); clear('name'); }} placeholder="How should we address you?" value={name} />
        <Field error={errors.email} label="Email" keyboardType="email-address" autoComplete="email" onChangeText={(value) => { setEmail(value); clear('email'); }} placeholder="you@email.com" value={email} />
        <PhoneField countryCode={countryCode} error={errors.phoneNumber} helperText="Optional. We keep the country code separate from your phone number." label="Phone number (optional)" onCountryCodeChange={setCountryCode} onPhoneNumberChange={(value) => { setPhoneNumber(value); clear('phoneNumber'); }} phoneNumber={phoneNumber} />
        <Field error={errors.password} label="Create password" onChangeText={(value) => { setPassword(value); clear('password'); }} placeholder="Create a secure password" secure value={password} />
        <PasswordRequirements password={password} repeatPassword={repeatPassword} />
        <Field error={errors.repeatPassword} label="Repeat password" onChangeText={(value) => { setRepeatPassword(value); clear('repeatPassword'); }} placeholder="Enter it again" secure value={repeatPassword} />
        {requestError ? <Text accessibilityRole="alert" style={styles.requestError}>{requestError}</Text> : null}
        <PrimaryButton disabled={submitting} label={submitting ? 'Creating account…' : 'Create account'} onPress={() => void create()} />
        <TertiaryButton label="Already have an account? Sign in" onPress={() => router.replace('/sign-in')} />
        <Text style={styles.legal}>By creating an account, you agree to the Terms of Service and Privacy Policy.</Text>
      </View>
    </ScreenLayout>
  );
}

const styles = StyleSheet.create({
  content: { gap: 0, overflow: 'hidden', paddingBottom: spacing.xxl, paddingTop: 0, position: 'relative' },
  artLayer: { ...StyleSheet.absoluteFill, overflow: 'hidden' },
  foreground: { gap: spacing.lg, minWidth: 0 },
  logo: { alignSelf: 'center', marginTop: spacing.xs },
  leftBranch: { left: -spacing.md, position: 'absolute', top: -spacing.xs },
  bottomBranch: { bottom: -spacing.xxl, position: 'absolute', right: -spacing.md },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, marginTop: spacing.xl },
  subtitle: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 },
  requestError: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 },
  legal: { color: colors.text.secondary, fontSize: fontSize.caption, lineHeight: 18, textAlign: 'center' },
});
