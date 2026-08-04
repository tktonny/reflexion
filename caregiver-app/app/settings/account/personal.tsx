import { useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout } from '../../../src/components/AppUI';
import { Field } from '../../../src/components/Field';
import { getCaregiverProfileV1, updateCaregiverProfileV1 } from '../../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, spacing } from '../../../src/theme';

export default function PersonalScreen() {
  const router = useRouter();
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  useEffect(() => { void getCaregiverProfileV1().then((profile) => setName(profile.name)).catch(() => setError('We could not load your personal information. Check your connection and try again.')); }, []);
  const save = async () => {
    if (!name.trim()) { setError('Enter your name.'); return; }
    setSaving(true); setError('');
    try { await updateCaregiverProfileV1({ name: name.trim() }); setError('Your personal information has been updated.'); }
    catch { setError('We could not save your personal information. Check your connection and try again.'); }
    finally { setSaving(false); }
  };
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Personal information" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Edit personal information</Text><Text style={styles.subtitle}>Use the name Reflexion should show in the caregiver app.</Text><Field error={error === 'Enter your name.' ? error : undefined} label="Preferred name" onChangeText={(value) => { setName(value); setError(''); }} value={name} />{error && error !== 'Enter your name.' ? <Text accessibilityRole="alert" style={styles.message}>{error}</Text> : null}{saving ? <ActivityIndicator color={colors.accent} /> : <PrimaryButton label="Save changes" onPress={() => void save()} />}</ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, subtitle: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 }, message: { color: colors.accent, fontSize: fontSize.body, lineHeight: 22 } });
