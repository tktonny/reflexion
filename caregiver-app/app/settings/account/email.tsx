import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { ActivityIndicator, StyleSheet, Text } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout } from '../../../src/components/AppUI';
import { Field } from '../../../src/components/Field';
import { validateEmail } from '../../../src/lib/authValidation';
import { emailChangeMessage } from '../../../src/lib/authMessages';
import { requestEmailChangeV1 } from '../../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, spacing } from '../../../src/theme';

export default function EmailScreen() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [sending, setSending] = useState(false);
  const send = async () => {
    const validation = validateEmail(email);
    setError(validation || '');
    if (validation) return;
    setSending(true);
    try { await requestEmailChangeV1(email); router.push('/settings/account/email/verify'); }
    catch (cause) { setError(emailChangeMessage(cause)); }
    finally { setSending(false); }
  };
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Change email" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Change email</Text><Text style={styles.subtitle}>A six-digit confirmation code will be sent to your new address. Your sign-in email changes only after you confirm it.</Text><Field error={error} label="New email" keyboardType="email-address" autoComplete="email" onChangeText={(value) => { setEmail(value); setError(''); }} value={email} />{sending ? <ActivityIndicator color={colors.accent} /> : <PrimaryButton label="Request verification code" onPress={() => void send()} />}</ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, subtitle: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 } });
