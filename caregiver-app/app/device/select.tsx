import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { useCaregiver } from '../../src/architecture/CaregiverContext';
import { AppHeader, PrimaryButton, ScreenLayout, TertiaryButton } from '../../src/components/AppUI';
import { MotionPressable } from '../../src/components/Motion';
import { listPatientRecordsV1, type V1PatientRecord } from '../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';

const DEVICE_TYPES = [
  { id: 'mirror', title: 'Mirror', description: 'Reflexion Mirror', icon: 'monitor' as const },
  { id: 'bear', title: 'Bear', description: 'Reflexion Bear', icon: 'heart' as const },
  { id: 'app', title: 'App', description: 'Older-adult app', icon: 'smartphone' as const },
  { id: 'other', title: 'Other supported device', description: 'A supported Reflexion device', icon: 'box' as const },
];

export default function DeviceSelectionScreen() {
  const router = useRouter();
  const { setSetupStatus } = useCaregiver();
  const [people, setPeople] = useState<V1PatientRecord[]>([]);
  const [selectedPerson, setSelectedPerson] = useState('');
  const [selectedDevice, setSelectedDevice] = useState('mirror');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const next = await listPatientRecordsV1();
      setPeople(next);
      setSelectedPerson((current) => current || next[0]?.patientId || '');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Loved ones could not be loaded. Check your connection and try again.');
    } finally {
      setLoading(false);
    }
  }, []);
  useFocusEffect(useCallback(() => { void load(); }, [load]));

  return <ScreenLayout contentContainerStyle={styles.content}>
    <AppHeader title="Caregiver setup" onBack={() => router.back()} />
    <View style={styles.hero}><Text accessibilityRole="header" style={styles.title}>Caregiver setup</Text><Text style={styles.subtitle}>Choose who the device is for and how you’ll connect.</Text></View>
    <Text style={styles.sectionTitle}>Who is this device for?</Text>
    {loading ? <ActivityIndicator color={colors.accent} /> : null}
    {error ? <><Text accessibilityRole="alert" style={styles.error}>{error}</Text><PrimaryButton label="Try again" onPress={() => void load()} /></> : null}
    {!loading && !error ? <View style={styles.people}>{people.map((person) => {
      const selected = selectedPerson === person.patientId;
      return <MotionPressable key={person.patientId} accessibilityLabel={person.displayName} accessibilityRole="button" accessibilityState={{ selected }} feedback="card" haptic="selection" onPress={() => setSelectedPerson(person.patientId)} style={[styles.person, selected && styles.personSelected]}>
        <View style={styles.avatar}><Text style={styles.avatarText}>{person.displayName.slice(0, 1).toUpperCase()}</Text></View>
        <Text style={styles.personName}>{person.displayName}</Text>
        <View style={[styles.radio, selected && styles.radioSelected]}>{selected ? <Feather color={colors.text.onAccent} name="check" size={16} /> : null}</View>
      </MotionPressable>;
    })}</View> : null}
    {!loading && !error && !people.length ? <Text style={styles.empty}>Add a loved one before choosing a device.</Text> : null}
    <Text style={styles.sectionTitle}>Select device type</Text>
    <View style={styles.deviceGrid}>{DEVICE_TYPES.map((device) => {
      const selected = selectedDevice === device.id;
      return <MotionPressable key={device.id} accessibilityLabel={`${device.title}. ${device.description}`} accessibilityRole="button" accessibilityState={{ selected }} feedback="card" haptic="selection" onPress={() => setSelectedDevice(device.id)} style={[styles.deviceCard, selected && styles.deviceSelected]}>
        <View style={[styles.deviceCheck, selected && styles.deviceCheckSelected]}>{selected ? <Feather color={colors.text.onAccent} name="check" size={15} /> : null}</View>
        <Feather color={selected ? colors.accent : colors.textDecorative} name={device.icon} size={30} />
        <Text style={styles.deviceTitle}>{device.title}</Text>
        <Text style={styles.deviceDescription}>{device.description}</Text>
      </MotionPressable>;
    })}</View>
    {selectedDevice !== 'mirror' ? <View style={styles.note}><Feather color={colors.accent} name="info" size={18} /><Text style={styles.noteText}>Physical Reflexion Mirror pairing is currently available for this setup.</Text></View> : null}
    <PrimaryButton disabled={!selectedPerson} label="Continue" onPress={() => { if (!selectedPerson) return; setSetupStatus('pair-device', 'in-progress'); router.push(`/device/${selectedPerson}/pairing`); }} />
    <TertiaryButton label="Set up later" onPress={() => router.replace('/(tabs)')} />
  </ScreenLayout>;
}

const styles = StyleSheet.create({
  content: { gap: spacing.lg, minWidth: 0 },
  hero: { gap: spacing.sm, minWidth: 0 },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, minWidth: 0 },
  subtitle: { color: colors.text.secondary, fontFamily: fontFamily.ui, fontSize: fontSize.bodyLarge, lineHeight: 24, minWidth: 0 },
  sectionTitle: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.heading, fontWeight: '500', lineHeight: 28, marginTop: spacing.md, minWidth: 0 },
  people: { gap: spacing.md, minWidth: 0 },
  person: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.lg, borderWidth: 1, flexDirection: 'row', gap: spacing.md, minHeight: 76, minWidth: 0, paddingHorizontal: spacing.lg, paddingVertical: spacing.md },
  personSelected: { backgroundColor: '#F1F8F4', borderColor: colors.accent, borderWidth: 1.5 },
  avatar: { alignItems: 'center', backgroundColor: '#E7F0EA', borderRadius: 24, flexShrink: 0, height: 48, justifyContent: 'center', width: 48 },
  avatarText: { color: colors.accent, fontFamily: fontFamily.ui, fontSize: fontSize.bodyLarge, fontWeight: '600' },
  personName: { color: colors.text.primary, flex: 1, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.bodyLarge, fontWeight: '600', lineHeight: 23, minWidth: 0 },
  radio: { alignItems: 'center', borderColor: colors.border.strong, borderRadius: 999, borderWidth: 2, flexShrink: 0, height: 30, justifyContent: 'center', width: 30 },
  radioSelected: { backgroundColor: colors.accent, borderColor: colors.accent },
  deviceGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.md, minWidth: 0 },
  deviceCard: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.lg, borderWidth: 1, flexBasis: '47%', flexGrow: 1, flexShrink: 1, minHeight: 148, minWidth: 0, padding: spacing.lg, position: 'relative' },
  deviceSelected: { backgroundColor: '#F1F8F4', borderColor: colors.accent, borderWidth: 1.5 },
  deviceCheck: { alignItems: 'center', borderColor: colors.border.strong, borderRadius: 999, borderWidth: 2, height: 26, justifyContent: 'center', position: 'absolute', right: spacing.md, top: spacing.md, width: 26 },
  deviceCheckSelected: { backgroundColor: colors.accent, borderColor: colors.accent },
  deviceTitle: { color: colors.text.primary, fontFamily: fontFamily.ui, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 22, marginTop: spacing.md, minWidth: 0, textAlign: 'center' },
  deviceDescription: { color: colors.text.secondary, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 20, marginTop: spacing.xs, minWidth: 0, textAlign: 'center' },
  note: { alignItems: 'flex-start', backgroundColor: '#F1F6F2', borderRadius: radius.lg, flexDirection: 'row', gap: spacing.sm, minWidth: 0, padding: spacing.md },
  noteText: { color: colors.text.secondary, flex: 1, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 20, minWidth: 0 },
  empty: { color: colors.text.secondary, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 22 },
  error: { color: colors.error.text, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 22, minWidth: 0 },
});
