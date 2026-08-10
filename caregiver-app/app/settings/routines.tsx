import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useCallback, useRef, useState } from 'react';
import { ActivityIndicator, Alert, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';

import { useCaregiver } from '../../src/architecture/CaregiverContext';
import { AppHeader, ChoiceCard, PrimaryButton, ScreenLayout, SecondaryButton, SelectionButton, SettingsRow, TertiaryButton } from '../../src/components/AppUI';
import { MotionPressable } from '../../src/components/Motion';
import { createRoutineV1, endRoutineV1, listRoutinesV1, loadCaregiverHome, updateRoutineV1, type CaregiverHome, type V1Routine, type V1RoutineNotificationPolicy } from '../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';

const CATEGORIES: V1Routine['category'][] = ['medication', 'meals', 'hydration', 'medical-appointments', 'exercise', 'family-events', 'custom-other'];
const CATEGORY_LABELS: Record<V1Routine['category'], string> = { medication: 'Medication', meals: 'Meals', hydration: 'Hydration', 'medical-appointments': 'Medical appointments', exercise: 'Exercise', 'family-events': 'Family events', 'custom-other': 'Custom / Other' };
const POLICIES: V1RoutineNotificationPolicy[] = ['do-not-notify', 'after-one-missed-or-unclear-response', 'daily-summary'];
const POLICY_LABELS: Record<V1RoutineNotificationPolicy, string> = { 'do-not-notify': 'Do not notify me', 'after-one-missed-or-unclear-response': 'Notify me after one missed or unclear response', 'daily-summary': 'Include it in my daily summary' };
const DEFAULT_POLICIES: V1RoutineNotificationPolicy[] = ['after-one-missed-or-unclear-response'];
const ALL_DAYS = [0, 1, 2, 3, 4, 5, 6];
const DAY_OPTIONS = [{ value: 1, label: 'Mon' }, { value: 2, label: 'Tue' }, { value: 3, label: 'Wed' }, { value: 4, label: 'Thu' }, { value: 5, label: 'Fri' }, { value: 6, label: 'Sat' }, { value: 0, label: 'Sun' }];
const DEFAULT_TIMES = ['08:00', '12:00', '18:00', '20:00', '21:00', '22:00'];
type Frequency = 'daily' | 'weekly';

function legacyPolicy(policies: V1RoutineNotificationPolicy[]) {
  return policies.find((policy) => policy !== 'do-not-notify') || 'do-not-notify';
}

function policiesFor(routine: V1Routine) {
  return routine.notificationPolicies?.length ? routine.notificationPolicies : [routine.notificationPolicy];
}

function validTime(value: string) {
  return /^(?:[01]\d|2[0-3]):[0-5]\d$/.test(value);
}

function scheduleLabel(schedule: V1Routine['schedule']) {
  const frequency = schedule.recurrence === 'weekly' ? `${schedule.daysOfWeek?.length || 0} days/week` : 'Every day';
  const count = schedule.times.length;
  return `${frequency} · ${count} ${count === 1 ? 'time' : 'times'}/day`;
}

export default function RoutineManagementScreen() {
  const router = useRouter();
  const { setSetupStatus } = useCaregiver();
  const scrollRef = useRef<ScrollView>(null);
  const [formTop, setFormTop] = useState(0);
  const [home, setHome] = useState<CaregiverHome | null>(null);
  const [patientId, setPatientId] = useState('');
  const [routines, setRoutines] = useState<V1Routine[]>([]);
  const [editing, setEditing] = useState<V1Routine | null>(null);
  const [name, setName] = useState('');
  const [category, setCategory] = useState<V1Routine['category']>('medication');
  const [times, setTimes] = useState(['08:00']);
  const [frequency, setFrequency] = useState<Frequency>('daily');
  const [daysOfWeek, setDaysOfWeek] = useState<number[]>(ALL_DAYS);
  const [policies, setPolicies] = useState<V1RoutineNotificationPolicy[]>(DEFAULT_POLICIES);
  const [startsOn, setStartsOn] = useState('');
  const [endsOn, setEndsOn] = useState('');
  const [spokenReminder, setSpokenReminder] = useState('');
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

  const beginAdd = () => {
    setEditing(null); setName(''); setCategory('medication'); setTimes(['08:00']); setFrequency('daily'); setDaysOfWeek([...ALL_DAYS]); setPolicies([...DEFAULT_POLICIES]); setStartsOn(''); setEndsOn(''); setSpokenReminder('');
  };

  const beginEdit = (routine: V1Routine) => {
    const weekly = routine.schedule.recurrence === 'weekly' || Array.isArray(routine.schedule.daysOfWeek);
    setEditing(routine); setName(routine.name); setCategory(routine.category); setTimes(routine.schedule.times.length ? routine.schedule.times : ['08:00']); setFrequency(weekly ? 'weekly' : 'daily'); setDaysOfWeek(routine.schedule.daysOfWeek?.length ? [...routine.schedule.daysOfWeek] : [...ALL_DAYS]); setPolicies(policiesFor(routine)); setStartsOn(routine.schedule.startsOn || ''); setEndsOn(routine.schedule.endsOn || ''); setSpokenReminder(routine.spokenReminder || routine.notes || '');
    scrollRef.current?.scrollTo({ y: formTop, animated: true });
  };

  const addTime = () => {
    if (times.length >= DEFAULT_TIMES.length) return;
    setTimes((current) => [...current, DEFAULT_TIMES.find((candidate) => !current.includes(candidate)) || '08:00']);
  };

  const removeTime = (index: number) => setTimes((current) => current.length === 1 ? current : current.filter((_, currentIndex) => currentIndex !== index));
  const toggleDay = (day: number) => setDaysOfWeek((current) => current.includes(day) ? current.filter((item) => item !== day) : [...current, day].sort((a, b) => a - b));

  const togglePolicy = (policy: V1RoutineNotificationPolicy) => {
    setPolicies((current) => {
      if (policy === 'do-not-notify') return ['do-not-notify'];
      const withoutNone = current.filter((item) => item !== 'do-not-notify');
      const next = withoutNone.includes(policy) ? withoutNone.filter((item) => item !== policy) : [...withoutNone, policy];
      return next.length ? next : ['do-not-notify'];
    });
  };

  const save = async (addAnother = false) => {
    if (!patientId || !name.trim()) { setError('Choose a loved one and add a routine name.'); return; }
    const normalizedTimes = [...new Set(times.map((item) => item.trim()))].sort();
    if (!normalizedTimes.length || normalizedTimes.some((item) => !validTime(item))) { setError('Use times in 24-hour HH:mm format.'); return; }
    if (frequency === 'weekly' && !daysOfWeek.length) { setError('Choose at least one day of the week.'); return; }
    if (startsOn && !/^\d{4}-\d{2}-\d{2}$/.test(startsOn)) { setError('Use a start date in YYYY-MM-DD format.'); return; }
    if (endsOn && !/^\d{4}-\d{2}-\d{2}$/.test(endsOn)) { setError('Use an end date in YYYY-MM-DD format.'); return; }
    if (startsOn && endsOn && endsOn < startsOn) { setError('The end date must be on or after the start date.'); return; }
    setSaving(true); setError('');
    try {
      const notificationPolicy = legacyPolicy(policies);
      const schedule = {
        timezone: editing?.schedule.timezone || home?.patients.find((person) => person.patientId === patientId)?.timezone || 'Asia/Singapore',
        times: normalizedTimes,
        recurrence: frequency,
        ...(frequency === 'weekly' ? { daysOfWeek: [...daysOfWeek].sort((a, b) => a - b) } : {}),
        ...(startsOn ? { startsOn } : {}),
        ...(endsOn ? { endsOn } : {}),
      };
      if (editing) {
        await updateRoutineV1(editing, { name: name.trim(), category, schedule, notificationPolicy, notificationPolicies: policies, spokenReminder: spokenReminder.trim() || null, notes: editing.notes });
      } else {
        await createRoutineV1(patientId, { name: name.trim(), category, schedule, notificationPolicy, notificationPolicies: policies, spokenReminder: spokenReminder.trim() || undefined });
      }
      setSetupStatus('routines', 'complete');
      setRoutines(await listRoutinesV1(patientId));
      beginAdd();
      scrollRef.current?.scrollTo({ y: addAnother ? Math.max(0, formTop - spacing.md) : 0, animated: true });
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'The routine could not be saved.'); }
    finally { setSaving(false); }
  };

  const end = (routine: V1Routine) => Alert.alert('Delete future reminders?', 'Past responses remain in the activity record. Future reminders for this routine will stop.', [{ text: 'Keep routine', style: 'cancel' }, { text: 'Delete', style: 'destructive', onPress: async () => { setSaving(true); try { await endRoutineV1(routine.routineId); setRoutines(await listRoutinesV1(patientId)); if (editing?.routineId === routine.routineId) beginAdd(); } catch (cause) { setError(cause instanceof Error ? cause.message : 'The routine could not be deleted.'); } finally { setSaving(false); } } }]);

  return <ScreenLayout scrollRef={scrollRef} contentContainerStyle={styles.content}>
    <AppHeader title="Routines" onBack={() => router.back()} />
    <Text accessibilityRole="header" style={styles.title}>Routine management</Text>
    <Text style={styles.subtitle}>Create gentle daily prompts. The Mirror reports responses; it never claims that a routine happened without a response.</Text>
    {home && home.patients.length > 1 ? <View style={styles.patientPicker}><Text style={styles.label}>For</Text>{home.patients.map((person) => <SelectionButton key={person.patientId} label={person.displayName} onPress={() => void selectPatient(person.patientId)} selected={patientId === person.patientId} />)}</View> : null}
    {loading ? <ActivityIndicator color={colors.accent} /> : null}{error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
    {!loading && patientId ? <View style={styles.list}>{routines.map((routine) => <View key={routine.routineId} style={styles.routine}><SettingsRow icon="clock" label={routine.name} value={`${CATEGORY_LABELS[routine.category]} · ${scheduleLabel(routine.schedule)}`} onPress={() => beginEdit(routine)} /><View style={styles.routineActions}><TertiaryButton label="Delete" onPress={() => end(routine)} /></View></View>)}</View> : null}
    {!loading && !routines.length && patientId ? <View style={styles.empty}><Text style={styles.emptyTitle}>No routines yet</Text><Text style={styles.emptyCopy}>Add the first reminder for this loved one.</Text></View> : null}
    <View onLayout={(event) => setFormTop(event.nativeEvent.layout.y)} style={styles.formSection}>
      <Text style={styles.section}>{editing ? 'Edit routine' : 'Add routine'}</Text>
      <View style={styles.form}>
        <Text style={styles.label}>Routine name</Text>
        <TextInput accessibilityLabel="Routine name" onChangeText={setName} placeholder="Morning medication" placeholderTextColor={colors.placeholder} style={styles.input} value={name} />
        <Text style={styles.label}>Category</Text>
        <View style={styles.categoryGrid}>{CATEGORIES.map((item) => <ChoiceCard key={item} icon="circle" title={CATEGORY_LABELS[item]} description="Daily prompt" selected={category === item} onPress={() => setCategory(item)} />)}</View>
        <Text style={styles.label}>Frequency</Text>
        <SelectionButton label="Every day · 7 days/week" onPress={() => setFrequency('daily')} selected={frequency === 'daily'} />
        <SelectionButton label={`Selected days · ${daysOfWeek.length} days/week`} onPress={() => setFrequency('weekly')} selected={frequency === 'weekly'} />
        {frequency === 'weekly' ? <View style={styles.dayGrid}>{DAY_OPTIONS.map((day) => <DayButton key={day.value} label={day.label} selected={daysOfWeek.includes(day.value)} onPress={() => toggleDay(day.value)} />)}</View> : null}
        <Text style={styles.label}>Times each day</Text>
        {times.map((item, index) => <View key={`${index}-${item}`} style={styles.timeRow}><TextInput accessibilityLabel={`Routine time ${index + 1}`} autoCapitalize="none" keyboardType="numbers-and-punctuation" onChangeText={(value) => setTimes((current) => current.map((currentValue, currentIndex) => currentIndex === index ? value : currentValue))} placeholder="08:00" placeholderTextColor={colors.placeholder} style={[styles.input, styles.timeInput]} value={item} />{times.length > 1 ? <MotionPressable accessibilityLabel={`Remove time ${index + 1}`} haptic="selection" onPress={() => removeTime(index)} style={styles.removeTime}><Feather color={colors.accent} name="x" size={20} /></MotionPressable> : null}</View>)}
        <SecondaryButton disabled={times.length >= DEFAULT_TIMES.length} label="Add another time" onPress={addTime} />
        <Text style={styles.label}>Caregiver notifications</Text>
        {POLICIES.map((item) => <ChoiceCard key={item} icon="bell" title={POLICY_LABELS[item]} description={item === 'do-not-notify' ? 'No notification for this routine.' : item === 'daily-summary' ? 'Include the response in your daily summary.' : 'Notify after one missed or unclear response.'} selected={policies.includes(item)} onPress={() => togglePolicy(item)} />)}
        <Text style={styles.policyNote}>The two notification choices can be selected together.</Text>
        <Text style={styles.label}>Start date (optional)</Text>
        <TextInput accessibilityLabel="Routine start date" keyboardType="numbers-and-punctuation" onChangeText={setStartsOn} placeholder="YYYY-MM-DD" placeholderTextColor={colors.placeholder} style={styles.input} value={startsOn} />
        <Text style={styles.label}>End date (optional)</Text>
        <TextInput accessibilityLabel="Routine end date" keyboardType="numbers-and-punctuation" onChangeText={setEndsOn} placeholder="YYYY-MM-DD" placeholderTextColor={colors.placeholder} style={styles.input} value={endsOn} />
        <Text style={styles.label}>Spoken reminder</Text>
        <TextInput accessibilityLabel="Spoken reminder" multiline onChangeText={setSpokenReminder} placeholder="A short, familiar reminder" placeholderTextColor={colors.placeholder} style={[styles.input, styles.notes]} value={spokenReminder} />
      </View>
      <PrimaryButton disabled={saving || !patientId} label={saving ? 'Saving…' : 'Save routine'} onPress={() => void save(false)} />
      {!editing ? <SecondaryButton disabled={saving} label="Save and add another routine" onPress={() => void save(true)} /> : <SecondaryButton label="Cancel editing" onPress={beginAdd} />}
    </View>
    <TertiaryButton label="Set up later" onPress={() => router.back()} />
  </ScreenLayout>;
}

function DayButton({ label, selected, onPress }: { label: string; selected: boolean; onPress: () => void }) {
  return <MotionPressable accessibilityLabel={`${label}, ${selected ? 'selected' : 'not selected'}`} accessibilityRole="button" accessibilityState={{ selected }} feedback="card" haptic="selection" onPress={onPress} style={[styles.dayButton, selected && styles.dayButtonSelected]}><Text style={[styles.dayButtonText, selected && styles.dayButtonTextSelected]}>{label}</Text>{selected ? <Feather color={colors.accent} name="check" size={16} /> : null}</MotionPressable>;
}

const styles = StyleSheet.create({
  content: { gap: spacing.lg, minWidth: 0 },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg, minWidth: 0 },
  subtitle: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22, minWidth: 0 },
  patientPicker: { gap: spacing.sm, minWidth: 0 },
  label: { color: colors.text.primary, fontSize: fontSize.body, fontWeight: '700', lineHeight: 20, marginTop: spacing.sm, minWidth: 0 },
  error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 21, minWidth: 0 },
  list: { gap: spacing.md, minWidth: 0 },
  routine: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, minWidth: 0, overflow: 'hidden' },
  routineActions: { alignItems: 'center', borderTopColor: colors.border.subtle, borderTopWidth: 1, flexDirection: 'row', flexWrap: 'wrap', gap: spacing.lg, justifyContent: 'flex-end', minWidth: 0, paddingHorizontal: spacing.lg, paddingVertical: spacing.xs },
  empty: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, gap: spacing.sm, minWidth: 0, padding: spacing.xl },
  emptyTitle: { color: colors.text.primary, fontSize: fontSize.heading, fontWeight: '700', textAlign: 'center' },
  emptyCopy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 21, textAlign: 'center' },
  formSection: { gap: spacing.lg, minWidth: 0 },
  section: { color: colors.text.primary, fontSize: fontSize.heading, fontWeight: '700', marginTop: spacing.lg, minWidth: 0 },
  form: { gap: spacing.sm, minWidth: 0 },
  input: { backgroundColor: colors.surface.card, borderColor: colors.border.strong, borderRadius: radius.md, borderWidth: 1, color: colors.text.primary, fontSize: fontSize.bodyLarge, minHeight: 52, minWidth: 0, paddingHorizontal: spacing.lg, paddingVertical: spacing.md },
  notes: { minHeight: 92, paddingTop: spacing.lg, textAlignVertical: 'top' },
  categoryGrid: { gap: spacing.sm, minWidth: 0 },
  dayGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, minWidth: 0 },
  dayButton: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.md, borderWidth: 1, flexDirection: 'row', gap: spacing.xs, justifyContent: 'center', minHeight: 46, minWidth: 62, paddingHorizontal: spacing.md },
  dayButtonSelected: { backgroundColor: '#E7F3F0', borderColor: colors.accent, borderWidth: 1.5 },
  dayButtonText: { color: colors.text.primary, fontSize: fontSize.body, fontWeight: '700' },
  dayButtonTextSelected: { color: colors.accent },
  timeRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, minWidth: 0 },
  timeInput: { flex: 1 },
  removeTime: { alignItems: 'center', justifyContent: 'center', minHeight: 44, width: 44 },
  policyNote: { color: colors.text.secondary, fontSize: fontSize.caption, lineHeight: 18, minWidth: 0 },
});
