import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { ActivityIndicator, StyleSheet, Text } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout } from '../../../../src/components/AppUI';
import { Field } from '../../../../src/components/Field';
import { emailChangeMessage } from '../../../../src/lib/authMessages';
import { confirmEmailChangeV1 } from '../../../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, spacing } from '../../../../src/theme';

export default function VerifyEmailChange() {
  const router = useRouter();
  const [code, setCode] = useState('');
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState('');
  const confirm = async () => {
    if (!/^\d{6}$/.test(code)) { setError('Enter the six-digit code from your email.'); return; }
    setBusy(true); setError('');
    try { await confirmEmailChangeV1(code); setDone(true); }
    catch (cause) { setError(emailChangeMessage(cause)); }
    finally { setBusy(false); }
  };
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Verify email" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>{done ? 'Email updated' : 'Confirm your new email'}</Text><Text style={styles.copy}>{done ? 'Your sign-in email has been changed.' : 'Enter the six-digit code sent to your new email address.'}</Text>{error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}{!done ? <Field label="Verification code" keyboardType="number-pad" maxLength={6} onChangeText={(value) => { setCode(value.replace(/\D/g, '')); setError(''); }} placeholder="123456" value={code} /> : null}{done ? <PrimaryButton label="Back to account" onPress={() => router.replace('/settings/account')} /> : busy ? <ActivityIndicator color={colors.accent} /> : <PrimaryButton disabled={code.length !== 6} label="Verify and save" onPress={() => void confirm()} />}</ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, copy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 }, error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 } });
