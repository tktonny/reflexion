import { useFocusEffect, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, Alert, StyleSheet, Text, TextInput, View } from 'react-native';
import { AppHeader, ChoiceCard, PrimaryButton, ScreenLayout, SecondaryButton, SettingsRow, TertiaryButton } from '../../src/components/AppUI';
import { useCaregiver } from '../../src/architecture/CaregiverContext';
import { createRoutineV1, endRoutineV1, listRoutinesV1, loadCaregiverHome, updateRoutineV1, type CaregiverHome, type V1Routine } from '../../src/lib/v1Caregiver';
import { colors, contentColumn, fontFamily, fontSize, radius, spacing } from '../../src/theme';

const CATEGORIES: V1Routine['category'][] = ['medication', 'meals', 'hydration', 'medical-appointments', 'exercise', 'family-events', 'custom-other'];
const CATEGORY_LABELS: Record<V1Routine['category'], string> = { medication: 'Medication', meals: 'Meals', hydration: 'Hydration', 'medical-appointments': 'Medical appointments', exercise: 'Exercise', 'family-events': 'Family events', 'custom-other': 'Custom / Other' };
const POLICIES: V1Routine['notificationPolicy'][] = ['do-not-notify', 'after-one-missed-or-unclear-response', 'daily-summary'];
const POLICY_LABELS: Record<V1Routine['notificationPolicy'], string> = { 'do-not-notify': 'Do not notify me', 'after-one-missed-or-unclear-response': 'Notify me after one missed or unclear response', 'daily-summary': 'Include it in my daily summary' };

export default function RoutineManagementScreen() {
  const router = useRouter();
  const { setSetupStatus } = useCaregiver();
  const [home, setHome] = useState<CaregiverHome | null>(null);
  const [patientId, setPatientId] = useState('');
  const [routines, setRoutines] = useState<V1Routine[]>([]);
  const [editing, setEditing] = useState<V1Routine | null>(null);
  const [name, setName] = useState('');
  const [category, setCategory] = useState<V1Routine['category']>('medication');
  const [time, setTime] = useState('08:00');
  const [policy, setPolicy] = useState<V1Routine['notificationPolicy']>('after-one-missed-or-unclear-response');
  const [notes, setNotes] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    setLoading(true); setError('');
    try {
      const nextHome = await loadCaregiverHome();
      setHome(nextHome);
      const selected = patientId || nextHome.patients[0]?.patientId || '';
      setPatientId(selected);
      setRoutines(selected ? await listRoutinesV1(selected) : []);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Routines could not be loaded.');
    } finally { setLoading(false); }
  }, [patientId]);
  useFocusEffect(useCallback(() => { void refresh(); }, [refresh]));

  const selectPatient = async (nextId: string) => {
    setPatientId(nextId); setEditing(null); setError('');
    try { setRoutines(await listRoutinesV1(nextId)); } catch (cause) { setError(cause instanceof Error ? cause.message : 'Routines could not be loaded.'); }
  };

  const beginAdd = () => { setEditing(null); setName(''); setCategory('medication'); setTime('08:00'); setPolicy('after-one-missed-or-unclear-response'); setNotes(''); };
  const beginEdit = (routine: V1Routine) => { setEditing(routine); setName(routine.name); setCategory(routine.category); setTime(routine.schedule.times[0] || '08:00'); setPolicy(routine.notificationPolicy); setNotes(routine.notes || ''); };
  const save = async () => {
    if (!patientId || !name.trim()) { setError('Choose a loved one and add a routine name.'); return; }
    if (!/^\d{2}:\d{2}$/.test(time) || Number(time.slice(0, 2)) > 23 || Number(time.slice(3)) > 59) { setError('Use a time in 24-hour HH:mm format.'); return; }
    setSaving(true); setError('');
    try {
      if (editing) await updateRoutineV1(editing, { name: name.trim(), category, schedule: { ...editing.schedule, times: [time] }, notificationPolicy: policy, notes: notes.trim() || null });
      else await createRoutineV1(patientId, { name: name.trim(), category, schedule: { timezone: home?.patients.find((person) => person.patientId === patientId)?.timezone || 'Asia/Singapore', times: [time], recurrence: 'daily' }, notificationPolicy: policy, notes: notes.trim() || undefined });
      setSetupStatus('routines', 'complete');
      beginAdd();
      setRoutines(await listRoutinesV1(patientId));
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'The routine could not be saved.'); }
    finally { setSaving(false); }
  };
  const toggle = async (routine: V1Routine) => {
    setSaving(true); setError('');
    try { await updateRoutineV1(routine, { status: routine.status === 'paused' ? 'active' : 'paused' }); setRoutines(await listRoutinesV1(patientId)); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'The routine status could not be changed.'); }
    finally { setSaving(false); }
  };
  const end = (routine: V1Routine) => Alert.alert('Delete future reminders?', 'Past responses remain in the activity record. Future reminders for this routine will stop.', [{ text: 'Keep routine', style: 'cancel' }, { text: 'Delete', style: 'destructive', onPress: async () => { setSaving(true); try { await endRoutineV1(routine.routineId); setRoutines(await listRoutinesV1(patientId)); if (editing?.routineId === routine.routineId) beginAdd(); } catch (cause) { setError(cause instanceof Error ? cause.message : 'The routine could not be deleted.'); } finally { setSaving(false); } } }]);

  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Routines" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Routine management</Text><Text style={styles.subtitle}>Create gentle daily prompts. The Mirror reports responses; it never claims that a routine happened without a response.</Text>
    {home && home.patients.length > 1 ? <View style={styles.patientPicker}><Text style={styles.label}>For</Text>{home.patients.map((person) => <SecondaryButton key={person.patientId} label={person.displayName} onPress={() => void selectPatient(person.patientId)} />)}</View> : null}
    {loading ? <ActivityIndicator color={colors.accent} /> : null}{error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
    {!loading && patientId ? <View style={styles.list}>{routines.map((routine) => <View key={routine.routineId} style={styles.routine}><SettingsRow icon="clock" label={routine.name} value={`${CATEGORY_LABELS[routine.category]} · ${routine.schedule.times.join(', ')} · ${routine.status === 'paused' ? 'Paused' : 'Active'}`} onPress={() => beginEdit(routine)} /><View style={styles.routineActions}><TertiaryButton label={routine.status === 'paused' ? 'Resume' : 'Pause'} onPress={() => void toggle(routine)} /><TertiaryButton label="Delete" onPress={() => end(routine)} /></View></View>)}</View> : null}
    {!loading && !routines.length && patientId ? <View style={styles.empty}><Text style={styles.emptyTitle}>No routines yet</Text><Text style={styles.emptyCopy}>Add the first reminder for this loved one.</Text></View> : null}
    <Text style={styles.section}>{editing ? 'Edit routine' : 'Add routine'}</Text><View style={styles.form}><Text style={styles.label}>Routine name</Text><TextInput accessibilityLabel="Routine name" onChangeText={setName} placeholder="Morning medication" placeholderTextColor={colors.placeholder} style={styles.input} value={name} /><Text style={styles.label}>Category</Text><View style={styles.categoryGrid}>{CATEGORIES.map((item) => <ChoiceCard key={item} icon="circle" title={CATEGORY_LABELS[item]} description="Daily prompt" selected={category === item} onPress={() => setCategory(item)} />)}</View><Text style={styles.label}>Time</Text><TextInput accessibilityLabel="Routine time" autoCapitalize="none" keyboardType="numbers-and-punctuation" onChangeText={setTime} placeholder="08:00" placeholderTextColor={colors.placeholder} style={styles.input} value={time} /><Text style={styles.label}>Caregiver notifications</Text>{POLICIES.map((item) => <ChoiceCard key={item} icon="bell" title={POLICY_LABELS[item]} description={item === 'do-not-notify' ? 'No notification for this routine.' : item === 'daily-summary' ? 'Include the response in your daily summary.' : 'Notify after one missed or unclear response.'} selected={policy === item} onPress={() => setPolicy(item)} />)}<Text style={styles.label}>Notes (optional)</Text><TextInput accessibilityLabel="Routine notes" multiline onChangeText={setNotes} placeholder="A short, familiar reminder" placeholderTextColor={colors.placeholder} style={[styles.input, styles.notes]} value={notes} /></View><PrimaryButton disabled={saving || !patientId} label={saving ? 'Saving…' : editing ? 'Save changes' : 'Save routine'} onPress={() => void save()} />{editing ? <SecondaryButton label="Cancel editing" onPress={beginAdd} /> : null}<TertiaryButton label="Set up later" onPress={() => router.back()} /></ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, subtitle: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 }, patientPicker: { gap: spacing.sm }, label: { color: colors.text.primary, fontSize: fontSize.body, fontWeight: '700', lineHeight: 20, marginTop: spacing.sm }, error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 21 }, list: { gap: spacing.md }, routine: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, overflow: 'hidden' }, routineActions: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.lg, justifyContent: 'flex-end', paddingHorizontal: spacing.lg, paddingVertical: spacing.xs }, empty: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, gap: spacing.sm, padding: spacing.xl }, emptyTitle: { color: colors.text.primary, fontSize: fontSize.heading, fontWeight: '700' }, emptyCopy: { color: colors.text.secondary, fontSize: fontSize.body }, section: { color: colors.text.primary, fontSize: fontSize.heading, fontWeight: '700', marginTop: spacing.lg }, form: { gap: spacing.sm }, input: { backgroundColor: colors.surface.card, borderColor: colors.border.strong, borderRadius: radius.md, borderWidth: 1, color: colors.text.primary, fontSize: fontSize.bodyLarge, minHeight: 52, paddingHorizontal: spacing.lg, paddingVertical: spacing.md }, notes: { minHeight: 92, paddingTop: spacing.lg, textAlignVertical: 'top' }, categoryGrid: { gap: spacing.sm },
});
