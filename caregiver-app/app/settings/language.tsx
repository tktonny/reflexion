import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, SecondaryButton } from '../../src/components/AppUI';
import { useCaregiver } from '../../src/architecture/CaregiverContext';
import { MotionPressable, MotionToggle } from '../../src/components/Motion';
import { loadCaregiverSettings, putCarePlanV1, updatePatientV1, type CaregiverSettings } from '../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';

const LANGUAGES = ['English', '中文', 'Malay', 'Tamil'];
const TEXT_SIZES = ['Small', 'Medium', 'Large'];
const PACES = ['Slow', 'Normal', 'Fast'];
const VOICES = ['Warm female', 'Warm male', 'Neutral'];

export default function LanguageAccessibilityScreen() {
  const router = useRouter();
  const { setSetupStatus } = useCaregiver();
  const [settings, setSettings] = useState<CaregiverSettings | null>(null);
  const [preferredLanguage, setPreferredLanguage] = useState('English');
  const [secondaryLanguage, setSecondaryLanguage] = useState('None');
  const [timezone, setTimezone] = useState('Asia/Singapore');
  const [textSize, setTextSize] = useState('Medium');
  const [captions, setCaptions] = useState(true);
  const [speakingPace, setSpeakingPace] = useState('Normal');
  const [voice, setVoice] = useState('Warm female');
  const [volume, setVolume] = useState(70);
  const [hearingSupport, setHearingSupport] = useState(true);
  const [highContrast, setHighContrast] = useState(false);
  const [simplifiedInterface, setSimplifiedInterface] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  const lovedOneName = settings?.patients[0]?.displayName || 'your loved one';
  const forLovedOne = (label: string) => `${label} for ${lovedOneName}`;
  const onMirror = (label: string) => lovedOneName === 'your loved one' ? `${label} on your loved one's Mirror` : `${label} on ${lovedOneName}'s Mirror`;

  const load = useCallback(async () => {
    setLoading(true); setError('');
    try {
      const next = await loadCaregiverSettings();
      setSettings(next);
      const patient = next.patients[0];
      if (patient) {
        setPreferredLanguage(patient.preferredLanguage || 'English');
        setTimezone(patient.timezone || 'Asia/Singapore');
        const prefs = patient.carePlan?.communicationPreferences || {};
        setSecondaryLanguage(String(prefs.secondaryLanguage || 'None'));
        setTextSize(String(prefs.textSize || 'Medium'));
        setCaptions(prefs.captions !== false);
        setSpeakingPace(String(prefs.speechSpeed || 'normal').replace(/^./, (letter) => letter.toUpperCase()));
        setVoice(String(prefs.assistantVoice || 'Warm female'));
        setVolume(Number(prefs.volume ?? 70));
        setHearingSupport(prefs.hearingSupport !== false);
        setHighContrast(Boolean(prefs.highContrast));
        setSimplifiedInterface(Boolean(prefs.simplifiedInterface));
      }
    } catch { setError('We could not load language and accessibility settings. Check your connection and try again.'); }
    finally { setLoading(false); }
  }, []);
  useFocusEffect(useCallback(() => { void load(); }, [load]));

  const save = async () => {
    const patient = settings?.patients[0];
    if (!patient) { setError('Add a loved one before saving these settings.'); return; }
    setSaving(true); setError('');
    try {
      await updatePatientV1(patient.patientId, patient.version, {
        preferredLanguage,
        timezone,
        profile: { speechSpeed: speakingPace.toLowerCase() as 'slow' | 'normal' | 'fast' },
      });
      const currentPlan = patient.carePlan;
      await putCarePlanV1(patient.patientId, currentPlan?.version || 0, {
        dailyRoutine: currentPlan?.dailyRoutine || {},
        communicationPreferences: {
          ...(currentPlan?.communicationPreferences || {}),
          secondaryLanguage,
          textSize,
          captions,
          speechSpeed: speakingPace.toLowerCase(),
          assistantVoice: voice,
          volume,
          hearingSupport,
          highContrast,
          simplifiedInterface,
        },
        safetyNotes: currentPlan?.safetyNotes || null,
      });
      setSetupStatus('language-accessibility', 'complete');
      router.back();
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'We could not save these settings. Check your connection and try again.'); }
    finally { setSaving(false); }
  };

  return <ScreenLayout contentContainerStyle={styles.content}>
    <AppHeader title="Language & accessibility" onBack={() => router.back()} />
    <Text style={styles.eyebrow}>For your loved one</Text>
    <Text accessibilityRole="header" style={styles.title}>Language & accessibility</Text>
    <Text style={styles.subtitle}>Personalise how Reflexion speaks and looks. These choices are shared with the paired Mirror.</Text>
    {loading ? <ActivityIndicator color={colors.accent} /> : null}
    {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
    <Text style={styles.section}>Region & time</Text>
    <View style={styles.card}><Text style={styles.label}>Country / region</Text><Text style={styles.value}>Singapore</Text><Text style={styles.label}>Time zone</Text><Text style={styles.value}>{timezone}</Text></View>
    <Text style={styles.section}>Languages</Text>
    <View style={styles.card}><Text style={styles.label}>{forLovedOne('Spoken language')}</Text><View style={styles.options}>{LANGUAGES.map((item) => <Option key={item} label={item} selected={preferredLanguage === item} onPress={() => setPreferredLanguage(item)} />)}</View><Text style={styles.label}>{forLovedOne('Secondary language (optional)')}</Text><View style={styles.options}>{['None', ...LANGUAGES.filter((item) => item !== preferredLanguage)].map((item) => <Option key={item} label={item} selected={secondaryLanguage === item} onPress={() => setSecondaryLanguage(item)} />)}</View></View>
    <Text style={styles.section}>Accessibility</Text>
    <View style={styles.card}><Text style={styles.label}>{forLovedOne('Text size')}</Text><View style={styles.options}>{TEXT_SIZES.map((item) => <Option key={item} label={item} selected={textSize === item} onPress={() => setTextSize(item)} />)}</View><ToggleRow label={onMirror('Captions')} value={captions} onChange={setCaptions} /><Text style={styles.label}>{forLovedOne('Voice speed')}</Text><View style={styles.options}>{PACES.map((item) => <Option key={item} label={item} selected={speakingPace === item} onPress={() => setSpeakingPace(item)} />)}</View><Text style={styles.label}>{forLovedOne('Voice')}</Text><View style={styles.options}>{VOICES.map((item) => <Option key={item} label={item} selected={voice === item} onPress={() => setVoice(item)} />)}</View><Text style={styles.label}>{onMirror(`Reminder volume · ${volume}%`)}</Text><View style={styles.volumeRow}><MotionPressable accessibilityLabel={`Decrease ${onMirror('reminder volume')}`} haptic="selection" onPress={() => setVolume((value) => Math.max(0, value - 10))} style={styles.volumeButton}><Feather color={colors.accent} name="volume-1" size={18} /></MotionPressable><View style={styles.volumeTrack}><View style={[styles.volumeFill, { width: `${volume}%` }]} /></View><MotionPressable accessibilityLabel={`Increase ${onMirror('reminder volume')}`} haptic="selection" onPress={() => setVolume((value) => Math.min(100, value + 10))} style={styles.volumeButton}><Feather color={colors.accent} name="volume-2" size={18} /></MotionPressable></View><ToggleRow label={forLovedOne('Hearing support')} value={hearingSupport} onChange={setHearingSupport} /><ToggleRow label={onMirror('High contrast')} value={highContrast} onChange={setHighContrast} /><ToggleRow label={onMirror('Simplified interface')} value={simplifiedInterface} onChange={setSimplifiedInterface} /></View>
    <SecondaryButton label="Preview voice" onPress={() => router.push('/settings/voice-preview')} />
    {saving ? <ActivityIndicator color={colors.accent} /> : <PrimaryButton label="Save settings" onPress={() => void save()} />}
  </ScreenLayout>;
}

function Option({ label, selected, onPress }: { label: string; selected: boolean; onPress: () => void }) { return <MotionPressable accessibilityRole="button" accessibilityState={{ selected }} feedback="card" haptic="selection" onPress={onPress} style={[styles.option, selected && styles.optionSelected]}><Text style={[styles.optionText, selected && styles.optionTextSelected]}>{label}</Text></MotionPressable>; }
function ToggleRow({ label, value, onChange }: { label: string; value: boolean; onChange: (value: boolean) => void }) { return <View style={styles.toggle}><Text style={styles.toggleLabel}>{label}</Text><MotionToggle label={label} value={value} onValueChange={onChange} /></View>; }

const styles = StyleSheet.create({
  content: { gap: spacing.lg, minWidth: 0 }, eyebrow: { color: colors.accent, fontSize: fontSize.caption, fontWeight: '700', marginTop: spacing.sm }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, minWidth: 0 }, subtitle: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 24, minWidth: 0 }, section: { color: colors.text.primary, fontSize: fontSize.body, fontWeight: '700', marginTop: spacing.md }, card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, minWidth: 0, padding: spacing.lg }, label: { color: colors.text.primary, fontSize: fontSize.body, fontWeight: '700', minWidth: 0 }, value: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, minWidth: 0 }, options: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, minWidth: 0 }, option: { borderColor: colors.border.default, borderRadius: radius.pill, borderWidth: 1, minHeight: 42, justifyContent: 'center', paddingHorizontal: spacing.md }, optionSelected: { backgroundColor: colors.accent, borderColor: colors.accent }, optionText: { color: colors.text.primary, fontSize: fontSize.body, minWidth: 0 }, optionTextSelected: { color: colors.text.onAccent, fontWeight: '700' }, toggle: { alignItems: 'center', borderTopColor: colors.border.subtle, borderTopWidth: 1, flexDirection: 'row', justifyContent: 'space-between', minHeight: 52, minWidth: 0, paddingTop: spacing.sm }, toggleLabel: { color: colors.text.primary, flex: 1, fontSize: fontSize.bodyLarge, minWidth: 0 }, volumeRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, minWidth: 0 }, volumeButton: { alignItems: 'center', flexShrink: 0, justifyContent: 'center', minHeight: 44, width: 36 }, volumeTrack: { backgroundColor: '#DCE9E2', borderRadius: 4, flex: 1, height: 8, minWidth: 0, overflow: 'hidden' }, volumeFill: { backgroundColor: colors.accent, borderRadius: 4, height: '100%' }, error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 },
});
