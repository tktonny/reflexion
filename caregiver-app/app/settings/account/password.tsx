import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { ActivityIndicator, StyleSheet, Text } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout } from '../../../src/components/AppUI';
import { Field } from '../../../src/components/Field';
import { validateNewPassword, validatePasswordPair } from '../../../src/lib/authValidation';
import { MIN_PASSWORD_LENGTH, passwordResetMessage } from '../../../src/lib/authMessages';
import { changePasswordV1 } from '../../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, spacing } from '../../../src/theme';

export default function Password() {
  const router = useRouter();
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [repeat, setRepeat] = useState('');
  const [errors, setErrors] = useState<{ currentPassword?: string; password?: string; repeatPassword?: string }>({});
  const [requestError, setRequestError] = useState('');
  const [busy, setBusy] = useState(false);
  const save = async () => {
    const pairErrors = validatePasswordPair(next, repeat);
    const nextErrors = { ...pairErrors, ...(current ? {} : { currentPassword: 'Enter your current password.' }) };
    setErrors(nextErrors); setRequestError('');
    if (Object.keys(nextErrors).length || validateNewPassword(next)) return;
    setBusy(true);
    try { await changePasswordV1(current, next); setRequestError('Your password has been changed. Sign in again with the new password.'); }
    catch (cause) { setRequestError(passwordResetMessage(cause)); }
    finally { setBusy(false); }
  };
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Change password" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Change password</Text><Text style={styles.copy}>New passwords must be at least {MIN_PASSWORD_LENGTH} characters. Your current password is still accepted even if it was created under an older policy.</Text><Field error={errors.currentPassword} label="Current password" onChangeText={(value) => { setCurrent(value); setErrors((currentErrors) => ({ ...currentErrors, currentPassword: undefined })); }} secure value={current} /><Field error={errors.password} helperText={`At least ${MIN_PASSWORD_LENGTH} characters.`} label="New password" onChangeText={(value) => { setNext(value); setErrors((currentErrors) => ({ ...currentErrors, password: undefined })); }} placeholder={`At least ${MIN_PASSWORD_LENGTH} characters`} secure value={next} /><Field error={errors.repeatPassword} label="Repeat new password" onChangeText={(value) => { setRepeat(value); setErrors((currentErrors) => ({ ...currentErrors, repeatPassword: undefined })); }} secure value={repeat} />{requestError ? <Text accessibilityRole="alert" style={styles.message}>{requestError}</Text> : null}{busy ? <ActivityIndicator color={colors.accent} /> : <PrimaryButton label="Update password" onPress={() => void save()} />}</ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, copy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 }, message: { color: colors.accent, fontSize: fontSize.body, lineHeight: 22 } });
