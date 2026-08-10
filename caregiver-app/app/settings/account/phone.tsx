import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { ActivityIndicator, StyleSheet, Text } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, SecondaryButton } from '../../../src/components/AppUI';
import { Field, PhoneField } from '../../../src/components/Field';
import { validatePhone, normalizePhone, validateVerificationCode } from '../../../src/lib/authValidation';
import { phoneChangeMessage } from '../../../src/lib/authMessages';
import { confirmPhoneChangeV1, requestPhoneChangeV1 } from '../../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, spacing } from '../../../src/theme';

export default function PhoneScreen() {
  const router = useRouter();
  const [countryCode, setCountryCode] = useState('+65');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [code, setCode] = useState('');
  const [requested, setRequested] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const fullPhone = normalizePhone(countryCode, phoneNumber);

  const requestCode = async () => {
    const validation = validatePhone(countryCode, phoneNumber, true);
    setError(validation || ''); setNotice('');
    if (validation) return;
    setBusy(true);
    try { await requestPhoneChangeV1(fullPhone); setRequested(true); setNotice('The verification request is queued. Enter the six-digit code after the configured SMS provider accepts it.'); }
    catch (cause) { setError(phoneChangeMessage(cause)); }
    finally { setBusy(false); }
  };
  const confirm = async () => {
    const validation = validateVerificationCode(code);
    setError(validation || ''); setNotice('');
    if (validation) return;
    setBusy(true);
    try { await confirmPhoneChangeV1(fullPhone, code); setNotice('Your verified phone number has been saved.'); setRequested(false); }
    catch (cause) { setError(phoneChangeMessage(cause)); }
    finally { setBusy(false); }
  };
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Change phone" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Change phone number</Text><Text style={styles.copy}>We will verify the new number before it becomes your sign-in recovery contact.</Text><PhoneField countryCode={countryCode} error={!requested ? error : undefined} label="New phone number" onCountryCodeChange={setCountryCode} onPhoneNumberChange={(value) => { setPhoneNumber(value); setError(''); setNotice(''); }} phoneNumber={phoneNumber} />{requested ? <Field error={error} label="Verification code" keyboardType="number-pad" maxLength={6} onChangeText={(value) => { setCode(value.replace(/\D/g, '')); setError(''); setNotice(''); }} placeholder="123456" value={code} /> : null}{notice ? <Text accessibilityRole="alert" style={styles.success}>{notice}</Text> : null}{busy ? <ActivityIndicator color={colors.accent} /> : requested ? <><PrimaryButton label="Verify and save" onPress={() => void confirm()} /><SecondaryButton label="Request a new code" onPress={() => void requestCode()} /></> : <PrimaryButton label="Request verification code" onPress={() => void requestCode()} />}</ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, copy: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 }, success: { color: colors.accent, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 } });
