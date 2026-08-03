import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { StyleSheet, Text } from 'react-native';

import { BrandLockup } from '../src/components/BrandLockup';
import { Field } from '../src/components/Field';
import { PrimaryButton, ScreenLayout, TertiaryButton } from '../src/components/AppUI';
import { verificationMessage, verificationResendMessage } from '../src/lib/authMessages';
import { clearPendingVerification, loadPendingVerification, savePendingVerification } from '../src/lib/pendingVerification';
import { resendAccountVerificationV1, verifyAccountV1 } from '../src/lib/v1Caregiver';
import { setV1Session } from '../src/lib/v1AuthSession';
import { colors, fontFamily, fontSize, spacing } from '../src/theme';

export default function AccountVerificationScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ email?: string; code?: string; storageWarning?: string }>();
  const paramEmail = Array.isArray(params.email) ? params.email[0] : params.email || '';
  const paramCode = Array.isArray(params.code) ? params.code[0] : params.code || '';
  const storageWarning = Array.isArray(params.storageWarning) ? params.storageWarning[0] : params.storageWarning;
  const [email, setEmail] = useState(paramEmail);
  const [working, setWorking] = useState(false);
  const [code, setCode] = useState(paramCode);
  const [verified, setVerified] = useState(false);
  const [message, setMessage] = useState('');
  const [messageTone, setMessageTone] = useState<'error' | 'info' | 'success'>('error');

  useEffect(() => {
    if (storageWarning === '1') {
      setMessageTone('info');
      setMessage('Your account was created, but this device could not save the verification context. Enter the code from your email when it arrives.');
    }
  }, [storageWarning]);

  useEffect(() => {
    let active = true;
    void loadPendingVerification().then((pending) => {
      if (active && !paramEmail && pending?.email) setEmail(pending.email);
      if (active && paramEmail && pending?.email !== paramEmail) {
        void savePendingVerification(paramEmail).catch(() => {
          if (active) {
            setMessageTone('info');
            setMessage('This device could not save the verification context. Enter the code from your email when it arrives.');
          }
        });
      }
    });
    return () => { active = false; };
  }, [paramEmail]);

  const verify = async () => {
    if (!/^\d{6}$/.test(code)) {
      setMessageTone('error');
      setMessage('Enter the six-digit code from your verification email.');
      return;
    }
    setWorking(true);
    setMessage(''); setMessageTone('error');
    try {
      const account = await verifyAccountV1(email, code);
      await setV1Session({ accessToken: account.accessToken, refreshToken: account.refreshToken, actor: { userId: account.actor.userId, tenantId: account.actor.tenantId, name: account.actor.name || '', email: account.actor.email || email, roles: account.actor.roles || [] } });
      await clearPendingVerification();
      setVerified(true);
      setMessageTone('success');
      setMessage('Your account is verified. You can begin setup now.');
    } catch (cause) {
      setMessageTone('error');
      setMessage(verificationMessage(cause));
    } finally {
      setWorking(false);
    }
  };

  const resend = async () => {
    const pending = await loadPendingVerification();
    const target = email.trim() || pending?.email || '';
    if (!target) {
      setMessageTone('error');
      setMessage('Return to account creation and enter your email address.');
      return;
    }
    setWorking(true);
    setMessage(''); setMessageTone('error');
    try {
      await resendAccountVerificationV1(target);
      await savePendingVerification(target);
      // This is intentionally not “sent”: 202 means the server accepted the request for delivery. The
      // configured transactional provider may still reject it asynchronously.
      setMessageTone('info');
      setMessage('Your verification request is queued. Enter the six-digit code after the configured email provider accepts the message.');
    } catch (cause) {
      setMessageTone('error');
      setMessage(verificationResendMessage(cause));
    } finally {
      setWorking(false);
    }
  };

  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <BrandLockup />
      <Text accessibilityRole="header" style={styles.title}>{verified ? 'You’re verified' : 'Verify your account'}</Text>
      <Text style={styles.subtitle}>{email ? `Enter the six-digit code sent to ${email}. It expires after the period shown in the email.` : 'Enter the six-digit code from your verification email to continue setting up Reflexion.'}</Text>
      {message ? <Text accessibilityRole={messageTone === 'error' ? 'alert' : undefined} style={[styles.message, messageTone === 'success' && styles.success, messageTone === 'info' && styles.info]}>{message}</Text> : null}
      {!verified ? <Field label="Verification code" keyboardType="number-pad" maxLength={6} onChangeText={(value) => { setCode(value.replace(/\D/g, '')); setMessage(''); }} placeholder="123456" value={code} /> : null}
      <PrimaryButton disabled={working || (!verified && code.length !== 6)} label={verified ? 'Continue to setup' : working ? 'Checking…' : 'Verify account'} onPress={() => verified ? router.replace('/welcome') : void verify()} />
      <TertiaryButton disabled={working} label="Resend verification code" onPress={() => void resend()} />
      <TertiaryButton label="Change email address" onPress={() => router.back()} />
    </ScreenLayout>
  );
}

const styles = StyleSheet.create({
  content: { gap: spacing.lg, justifyContent: 'center', paddingTop: spacing.welcome },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.display, fontWeight: '500', lineHeight: 42, textAlign: 'center' },
  subtitle: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 25, textAlign: 'center' },
  message: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22, textAlign: 'center' },
  success: { color: colors.accent },
  info: { color: colors.text.secondary },
});
