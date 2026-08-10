import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useMemo, useState } from 'react';
import { StyleSheet, Text } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, TertiaryButton } from '../src/components/AppUI';
import { BrandLockup } from '../src/components/BrandLockup';
import { Field } from '../src/components/Field';
import { validateVerificationCode } from '../src/lib/authValidation';
import { passwordResetMessage, passwordResetRequestMessage } from '../src/lib/authMessages';
import { requestPasswordResetV1, verifyPasswordResetCodeV1 } from '../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, spacing } from '../src/theme';

export default function ResetVerificationScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ email?: string }>();
  const email = useMemo(() => Array.isArray(params.email) ? params.email[0] : params.email || '', [params.email]);
  const [code, setCode] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [working, setWorking] = useState(false);

  const verify = async () => {
    const validation = validateVerificationCode(code);
    if (!email) { setError('Return to the previous screen and enter your email address.'); return; }
    setError(validation || ''); setNotice('');
    if (validation) return;
    setWorking(true);
    try {
      const result = await verifyPasswordResetCodeV1(email, code);
      router.replace({ pathname: '/reset-password', params: { token: result.resetToken } });
    } catch (cause) {
      setError(passwordResetMessage(cause));
    } finally {
      setWorking(false);
    }
  };

  const resend = async () => {
    if (!email) { setError('Return to the previous screen and enter your email address.'); return; }
    setWorking(true); setError(''); setNotice('');
    try {
      await requestPasswordResetV1(email);
      setCode('');
      setNotice('A new reset-code request is queued. Check your inbox after the configured email provider accepts it.');
    } catch (cause) {
      setError(passwordResetRequestMessage(cause));
    } finally {
      setWorking(false);
    }
  };

  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <AppHeader onBack={() => router.back()} />
      <BrandLockup compact />
      <Text accessibilityRole="header" style={styles.title}>Enter verification code</Text>
      <Text style={styles.subtitle}>{email ? `A six-digit code was requested for ${email}.` : 'Enter the six-digit code from your email.'}</Text>
      <Field error={error} label="Six-digit code" keyboardType="number-pad" maxLength={6} onChangeText={(value) => { setCode(value.replace(/\D/g, '')); setError(''); setNotice(''); }} placeholder="000000" value={code} />
      <Text style={styles.note}>The code expires after 30 minutes.</Text>
      {notice ? <Text style={styles.notice}>{notice}</Text> : null}
      <PrimaryButton disabled={working} label={working ? 'Checking…' : 'Verify code'} onPress={() => void verify()} />
      <TertiaryButton disabled={working} label="Resend code" onPress={() => void resend()} />
    </ScreenLayout>
  );
}

const styles = StyleSheet.create({
  content: { gap: spacing.lg },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, marginTop: spacing.xl },
  subtitle: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 },
  note: { color: colors.text.secondary, fontSize: fontSize.caption, lineHeight: 18 },
  notice: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 },
});
