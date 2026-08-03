import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { StyleSheet, Text } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, TertiaryButton } from '../src/components/AppUI';
import { BrandLockup } from '../src/components/BrandLockup';
import { Field, PhoneField } from '../src/components/Field';
import { validateCreateAccount, type FieldErrors, normalizePhone } from '../src/lib/authValidation';
import { registrationMessage } from '../src/lib/authMessages';
import { clearPendingVerification, savePendingVerification } from '../src/lib/pendingVerification';
import { setV1Session } from '../src/lib/v1AuthSession';
import { registerCaregiverV1 } from '../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, spacing } from '../src/theme';

export default function CreateAccountScreen() {
  const router = useRouter();
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
      // Only the email and timestamp are persisted. Passwords never enter navigation params or storage.
      // Registration has already succeeded if this local write fails, so do not describe it as a server
      // failure or silently strand the caregiver on the form.
      const pendingEmail = registration.email || email.trim().toLowerCase();
      let pendingSaveFailed = false;
      try {
        await savePendingVerification(pendingEmail);
      } catch {
        pendingSaveFailed = true;
      }
      router.replace({ pathname: '/account-verification', params: { email: pendingEmail, ...(pendingSaveFailed ? { storageWarning: '1' } : {}) } });
    } catch (cause) {
      setRequestError(registrationMessage(cause));
    } finally {
      setSubmitting(false);
    }
  };

  const clear = (key: keyof FieldErrors) => setErrors((current) => ({ ...current, [key]: undefined }));
  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <AppHeader onBack={() => router.back()} />
      <BrandLockup compact />
      <Text accessibilityRole="header" style={styles.title}>Create your account</Text>
      <Text style={styles.subtitle}>Join Reflexion to support the people you love with ease.</Text>
      <Field error={errors.name} label="Preferred name" onChangeText={(value) => { setName(value); clear('name'); }} placeholder="How should we address you?" value={name} />
      <Field error={errors.email} label="Email" keyboardType="email-address" autoComplete="email" onChangeText={(value) => { setEmail(value); clear('email'); }} placeholder="you@email.com" value={email} />
      <PhoneField countryCode={countryCode} error={errors.phoneNumber} helperText="Optional. We keep the country code separate from your phone number." label="Phone number (optional)" onCountryCodeChange={setCountryCode} onPhoneNumberChange={(value) => { setPhoneNumber(value); clear('phoneNumber'); }} phoneNumber={phoneNumber} />
      <Field error={errors.password} helperText="Use at least 12 characters." label="Create password" onChangeText={(value) => { setPassword(value); clear('password'); }} placeholder="At least 12 characters" secure value={password} />
      <Field error={errors.repeatPassword} label="Repeat password" onChangeText={(value) => { setRepeatPassword(value); clear('repeatPassword'); }} placeholder="Enter it again" secure value={repeatPassword} />
      {requestError ? <Text accessibilityRole="alert" style={styles.requestError}>{requestError}</Text> : null}
      <PrimaryButton disabled={submitting} label={submitting ? 'Creating account…' : 'Create account'} onPress={() => void create()} />
      <TertiaryButton label="Already have an account? Sign in" onPress={() => router.replace('/sign-in')} />
      <Text style={styles.legal}>By creating an account, you agree to the Terms of Service and Privacy Policy.</Text>
    </ScreenLayout>
  );
}

const styles = StyleSheet.create({
  content: { gap: spacing.lg },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, marginTop: spacing.xl },
  subtitle: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 },
  requestError: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 },
  legal: { color: colors.text.secondary, fontSize: fontSize.caption, lineHeight: 18, textAlign: 'center' },
});
