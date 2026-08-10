import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useMemo, useState } from 'react';
import { Image, StyleSheet, Text, TextInput, View, useWindowDimensions } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, TertiaryButton } from '../src/components/AppUI';
import { validateVerificationCode } from '../src/lib/authValidation';
import { passwordResetMessage, passwordResetRequestMessage } from '../src/lib/authMessages';
import { requestPasswordResetV1, verifyPasswordResetCodeV1 } from '../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, scaleSize, spacing } from '../src/theme';

export default function ResetVerificationScreen() {
  const router = useRouter();
  const { width } = useWindowDimensions();
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

  const logoWidth = Math.min(width * 0.46, scaleSize(178));
  const branchWidth = Math.min(width * 0.2, scaleSize(78));
  const bearPlantWidth = Math.min(width * 0.7, scaleSize(270));

  const updateCode = (value: string) => {
    setCode(value.replace(/\D/g, ''));
    setError('');
    setNotice('');
  };

  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <View pointerEvents="none" style={styles.artLayer}>
        <Image accessibilityElementsHidden importantForAccessibility="no-hide-descendants" resizeMode="contain" source={require('../assets/auth/botanical-left.png')} style={[styles.leftBranch, { height: branchWidth * (650 / 240), width: branchWidth }]} />
      </View>
      <View style={styles.foreground}>
        <AppHeader onBack={() => router.back()} />
        <Image accessibilityElementsHidden importantForAccessibility="no-hide-descendants" resizeMode="contain" source={require('../assets/auth/reflexion-logo.png')} style={[styles.logo, { height: logoWidth * (260 / 450), width: logoWidth }]} />
        <Text accessibilityRole="header" style={styles.title}>Enter verification code</Text>
        <Text style={styles.subtitle}>{email ? `A six-digit code was requested for ${email}.` : 'Enter the six-digit code from your email.'}</Text>
        <View style={styles.codeEntry}>
          <View pointerEvents="none" style={styles.codeBoxes}>
            {Array.from({ length: 6 }, (_, index) => <View key={index} style={styles.codeBox}><Text style={styles.codeText}>{code[index] || ''}</Text></View>)}
          </View>
          <TextInput accessibilityLabel="Six-digit code" autoCapitalize="none" keyboardType="number-pad" maxLength={6} onChangeText={updateCode} style={styles.codeInput} value={code} />
        </View>
        {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
        <Text style={styles.note}>The code expires after 30 minutes.</Text>
        {notice ? <Text style={styles.notice}>{notice}</Text> : null}
        <PrimaryButton disabled={working} label={working ? 'Checking…' : 'Verify code'} onPress={() => void verify()} />
        <TertiaryButton disabled={working} label="Resend code" onPress={() => void resend()} />
        <View pointerEvents="none" style={styles.bearPlantWrap}><Image accessibilityElementsHidden importantForAccessibility="no-hide-descendants" resizeMode="contain" source={require('../assets/auth/bear-plant.png')} style={{ height: bearPlantWidth * (300 / 340), width: bearPlantWidth }} /></View>
      </View>
    </ScreenLayout>
  );
}

const styles = StyleSheet.create({
  content: { gap: 0, overflow: 'hidden', paddingBottom: spacing.xxl, paddingTop: 0, position: 'relative' },
  artLayer: { ...StyleSheet.absoluteFill, overflow: 'hidden' },
  foreground: { gap: spacing.lg, minWidth: 0 },
  logo: { alignSelf: 'center', marginTop: spacing.xs },
  leftBranch: { left: -spacing.xxl, position: 'absolute', top: scaleSize(80) },
  bearPlantWrap: { alignItems: 'center', minWidth: 0, width: '100%' },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, marginTop: spacing.sm },
  subtitle: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 },
  note: { color: colors.text.secondary, fontSize: fontSize.caption, lineHeight: 18 },
  notice: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 },
  error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 },
  codeEntry: { minHeight: scaleSize(62), minWidth: 0, position: 'relative' },
  codeBoxes: { flexDirection: 'row', gap: spacing.sm, minWidth: 0 },
  codeBox: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.md, borderWidth: 1, flex: 1, height: scaleSize(62), justifyContent: 'center', minWidth: 0 },
  codeText: { color: colors.text.primary, fontSize: fontSize.display, lineHeight: scaleSize(42), textAlign: 'center' },
  codeInput: { ...StyleSheet.absoluteFill, color: 'transparent', opacity: 0.02, padding: 0, textAlign: 'center' },
});
