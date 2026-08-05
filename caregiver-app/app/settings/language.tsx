import { useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text } from 'react-native';

import { useCaregiver } from '../../src/architecture/CaregiverContext';
import { AppHeader, ChoiceCard, PrimaryButton, ScreenLayout } from '../../src/components/AppUI';
import { getCaregiverProfileV1, updateCaregiverProfileV1 } from '../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, spacing } from '../../src/theme';

export default function Language() {
  const router = useRouter();
  const { language, setLanguage } = useCaregiver();
  const [selected, setSelected] = useState<'en' | 'zh'>(language);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => { void getCaregiverProfileV1().then((profile) => { setSelected(profile.appLanguage); setLanguage(profile.appLanguage); }).catch(() => setError('We could not load your language preference. Check your connection and try again.')); }, []);
  const save = async () => {
    setBusy(true); setError('');
    try { await updateCaregiverProfileV1({ appLanguage: selected }); setLanguage(selected); setError(selected === 'en' ? 'Reflexion is now using English.' : 'Reflexion is now using Chinese where translations are available.'); }
    catch { setError('We could not save your language. Check your connection and try again.'); }
    finally { setBusy(false); }
  };
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="App language" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>App language</Text><Text style={styles.copy}>Your selection updates the app immediately and is saved for the next time you open Reflexion.</Text>{error ? <Text accessibilityRole="alert" style={[styles.message, error.startsWith('Reflexion') && styles.success]}>{error}</Text> : null}<ChoiceCard icon="check-circle" title="English" description="English" selected={selected === 'en'} onPress={() => setSelected('en')} /><ChoiceCard icon="circle" title="中文" description="Chinese" selected={selected === 'zh'} onPress={() => setSelected('zh')} />{busy ? <ActivityIndicator color={colors.accent} /> : <PrimaryButton label="Save language" onPress={() => void save()} />}</ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, copy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 }, message: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 }, success: { color: colors.accent } });
