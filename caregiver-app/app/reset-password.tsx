import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useMemo, useState } from 'react';
import { Alert, Image, StyleSheet, Text, View, useWindowDimensions } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, SurfaceCard, TertiaryButton } from '../src/components/AppUI';
import { Field } from '../src/components/Field';
import { PasswordRequirements } from '../src/components/PasswordRequirements';
import { passwordRequirementState, validatePasswordPair } from '../src/lib/authValidation';
import { passwordResetMessage } from '../src/lib/authMessages';
import { resetPasswordV1 } from '../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, scaleSize, spacing } from '../src/theme';

export default function ResetPasswordScreen() {
  const router = useRouter();
  const { width } = useWindowDimensions();
  const params = useLocalSearchParams<{ token?: string }>();
  const token = useMemo(() => Array.isArray(params.token) ? params.token[0] : params.token || '', [params.token]);
  const [password, setPassword] = useState('');
  const [repeat, setRepeat] = useState('');
  const [errors, setErrors] = useState<{ password?: string; repeatPassword?: string }>({});
  const [requestError, setRequestError] = useState('');
  const [working, setWorking] = useState(false);

  const reset = async () => {
    const nextErrors = validatePasswordPair(password, repeat);
    setErrors(nextErrors);
    setRequestError('');
    if (!token) { setRequestError('This reset link is missing. Request a new reset code and try again.'); return; }
    if (Object.keys(nextErrors).length) return;
    setWorking(true);
    try {
      await resetPasswordV1(token, password);
      Alert.alert('Password updated', 'Sign in with your new password.', [{ text: 'Sign in', onPress: () => router.replace('/sign-in') }]);
    } catch (cause) {
      setRequestError(passwordResetMessage(cause));
    } finally {
      setWorking(false);
    }
  };

  const branchWidth = Math.min(width * 0.3, scaleSize(116));
  const strengthState = passwordRequirementState(password);
  const strength = Object.values(strengthState).filter(Boolean).length;
  const strengthLabel = strength === 0 ? 'Not set' : strength >= 5 ? 'Strong' : strength >= 3 ? 'Medium' : 'Weak';

  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <View pointerEvents="none" style={styles.artLayer}>
        <Image accessibilityElementsHidden importantForAccessibility="no-hide-descendants" resizeMode="contain" source={require('../assets/auth/botanical-right.png')} style={[styles.bottomBranch, { height: branchWidth * (470 / 250), width: branchWidth }]} />
      </View>
      <View style={styles.foreground}>
        <AppHeader onBack={() => router.back()} />
        <Text accessibilityRole="header" style={styles.title}>Create new password</Text>
        <Text style={styles.subtitle}>Choose a new password. Active sessions will be signed out after the change.</Text>
        <Field error={errors.password} label="New password" onChangeText={(value) => { setPassword(value); setErrors((current) => ({ ...current, password: undefined })); }} placeholder="Create a secure password" secure value={password} />
        <View style={styles.strength}>
          <View style={styles.strengthLabel}><Text style={styles.strengthText}>Password strength</Text><Text style={[styles.strengthValue, strength >= 5 && styles.strengthStrong]}>{strengthLabel}</Text></View>
          <View accessibilityElementsHidden style={styles.strengthBars}>{Array.from({ length: 4 }, (_, index) => <View key={index} style={[styles.strengthBar, index < Math.ceil(strength / 1.25) && styles.strengthBarActive]} />)}</View>
        </View>
        <Field error={errors.repeatPassword} label="Repeat password" onChangeText={(value) => { setRepeat(value); setErrors((current) => ({ ...current, repeatPassword: undefined })); }} placeholder="Enter it again" secure value={repeat} />
        {requestError ? <Text accessibilityRole="alert" style={styles.requestError}>{requestError}</Text> : null}
        <PrimaryButton disabled={working} label={working ? 'Updating…' : 'Reset password'} onPress={() => void reset()} />
        <TertiaryButton label="Back to sign in" onPress={() => router.replace('/sign-in')} />
        <SurfaceCard style={styles.requirementsCard}><Text style={styles.requirementsTitle}>Password requirements</Text><PasswordRequirements password={password} repeatPassword={repeat} /></SurfaceCard>
      </View>
    </ScreenLayout>
  );
}

const styles = StyleSheet.create({
  content: { gap: 0, overflow: 'hidden', paddingBottom: spacing.xxl, paddingTop: 0, position: 'relative' },
  artLayer: { ...StyleSheet.absoluteFill, overflow: 'hidden' },
  foreground: { gap: spacing.lg, minWidth: 0 },
  bottomBranch: { bottom: -spacing.xxl, position: 'absolute', right: -spacing.md },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, marginTop: spacing.xl },
  subtitle: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 },
  requestError: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 },
  strength: { gap: spacing.sm, minWidth: 0 },
  strengthLabel: { alignItems: 'baseline', flexDirection: 'row', gap: spacing.sm, minWidth: 0 },
  strengthText: { color: colors.text.primary, fontSize: fontSize.body, lineHeight: 22 },
  strengthValue: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 },
  strengthStrong: { color: colors.status.green, fontWeight: '700' },
  strengthBars: { flexDirection: 'row', gap: spacing.xs, minWidth: 0 },
  strengthBar: { backgroundColor: colors.border.strong, borderRadius: 3, flex: 1, height: scaleSize(5), minWidth: 0 },
  strengthBarActive: { backgroundColor: colors.status.green },
  requirementsCard: { gap: spacing.md },
  requirementsTitle: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 24 },
});
