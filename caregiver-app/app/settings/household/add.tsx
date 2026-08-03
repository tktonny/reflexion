import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, SecondaryButton } from '../../../src/components/AppUI';
import { Field, PhoneField } from '../../../src/components/Field';
import { normalizePhone, validatePhone } from '../../../src/lib/authValidation';
import { createPatientV1 } from '../../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, spacing } from '../../../src/theme';

export default function AddLovedOneScreen() {
  const router = useRouter();
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState<'female' | 'male' | 'other'>('female');
  const [language, setLanguage] = useState<'en' | 'zh'>('en');
  const [countryCode, setCountryCode] = useState('+65');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [error, setError] = useState('');
  const [fieldError, setFieldError] = useState('');
  const [saving, setSaving] = useState(false);

  const save = async () => {
    if (!name.trim()) { setFieldError('Enter the name they like to be called.'); return; }
    const parsedAge = age.trim() ? Number(age) : null;
    if (parsedAge !== null && (!Number.isInteger(parsedAge) || parsedAge < 1 || parsedAge > 130)) { setFieldError('Enter an age between 1 and 130.'); return; }
    if (phoneNumber.trim()) { const phoneError = validatePhone(countryCode, phoneNumber); if (phoneError) { setFieldError(phoneError); return; } }
    setFieldError(''); setError(''); setSaving(true);
    try {
      await createPatientV1({ displayName: name.trim(), preferredLanguage: language, timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'Asia/Singapore', relationshipType: 'other', profile: { age: parsedAge, gender, phoneNumber: phoneNumber.trim() ? normalizePhone(countryCode, phoneNumber) : null } });
      router.replace('/settings/household');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'We could not save this loved one. Check your connection and try again.'); }
    finally { setSaving(false); }
  };

  return <ScreenLayout><AppHeader title="Add loved one" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Add a loved one</Text><Text style={styles.copy}>Tell us a few details so updates and messages stay connected to the right person.</Text><Field error={fieldError && !fieldError.toLowerCase().includes('phone') ? fieldError : undefined} label="Name they like to be called" onChangeText={(value) => { setName(value); setFieldError(''); }} placeholder="e.g. Mum" value={name} /><Field label="Age (optional)" keyboardType="number-pad" onChangeText={(value) => { setAge(value.replace(/\D/g, '')); setFieldError(''); }} placeholder="82" value={age} /><Text style={styles.label}>Gender</Text><View style={styles.options}>{(['female', 'male', 'other'] as const).map((value) => <SecondaryButton key={value} label={gender === value ? `✓ ${value}` : value} onPress={() => setGender(value)} />)}</View><Text style={styles.label}>Preferred language</Text><View style={styles.options}><SecondaryButton label={language === 'en' ? '✓ English' : 'English'} onPress={() => setLanguage('en')} /><SecondaryButton label={language === 'zh' ? '✓ 中文' : '中文'} onPress={() => setLanguage('zh')} /></View><PhoneField countryCode={countryCode} error={fieldError.toLowerCase().includes('phone') ? fieldError : undefined} helperText="Optional. The country code stays separate from the phone number." label="Phone number (optional)" onCountryCodeChange={(value) => { setCountryCode(value); setFieldError(''); }} onPhoneNumberChange={(value) => { setPhoneNumber(value); setFieldError(''); }} phoneNumber={phoneNumber} />{error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}{saving ? <ActivityIndicator color={colors.accent} /> : <PrimaryButton label="Save loved one" onPress={() => void save()} />}</ScreenLayout>;
}

const styles = StyleSheet.create({ title: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, copy: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 }, label: { color: colors.text.primary, fontSize: fontSize.body, fontWeight: '700', marginTop: spacing.sm }, options: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm }, error: { color: colors.error.text, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 } });
