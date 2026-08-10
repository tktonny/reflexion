import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { useCaregiver } from '../../src/architecture/CaregiverContext';
import { AppHeader, PrimaryButton, ScreenLayout, TertiaryButton } from '../../src/components/AppUI';
import { MotionPressable } from '../../src/components/Motion';
import { listPatientRecordsV1, type V1PatientRecord } from '../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';

export default function HouseholdReviewScreen() {
  const router = useRouter();
  const { setSetupStatus } = useCaregiver();
  const [people, setPeople] = useState<V1PatientRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const load = useCallback(async () => {
    setLoading(true); setError('');
    try { setPeople(await listPatientRecordsV1()); } catch { setError('We could not load your household. Check your connection and try again.'); } finally { setLoading(false); }
  }, []);
  useFocusEffect(useCallback(() => { void load(); }, [load]));
  return <ScreenLayout contentContainerStyle={styles.content}>
    <AppHeader title="Review household" onBack={() => router.back()} />
    <Text accessibilityRole="header" style={styles.title}>Review household</Text>
    <Text style={styles.subtitle}>Here are the loved ones in your household. You can edit or add another person.</Text>
    {loading ? <ActivityIndicator color={colors.accent} /> : null}
    {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
    {people.map((person) => <View key={person.patientId} style={styles.card}><View style={styles.person}><View style={styles.avatar}><Text style={styles.avatarText}>{person.displayName.slice(0, 1).toUpperCase()}</Text></View><View style={styles.personCopy}><Text style={styles.name}>{person.displayName}</Text><Text style={styles.meta}>{person.profile.age ? `${person.profile.age} years` : 'Age not added'}</Text></View><Feather color={colors.textDecorative} name="chevron-right" size={20} /></View><View style={styles.actions}><MotionPressable accessibilityRole="button" onPress={() => router.push(`/settings/household/${person.patientId}`)} style={styles.action}><Text style={styles.actionText}>Edit</Text></MotionPressable><MotionPressable accessibilityRole="button" onPress={() => router.push(`/device/${person.patientId}`)} style={styles.action}><Text style={styles.actionText}>Pair device</Text></MotionPressable></View></View>)}
    {!loading && !people.length ? <View style={styles.empty}><Text style={styles.emptyTitle}>No loved ones added yet</Text><Text style={styles.emptyCopy}>Add a loved one to continue household setup.</Text></View> : null}
    <MotionPressable accessibilityRole="button" haptic="selection" onPress={() => router.push('/setup/household')} style={styles.add}><Feather color={colors.accent} name="plus" size={18} /><Text style={styles.addText}>Add another loved one</Text></MotionPressable>
    <PrimaryButton label="Continue setup" onPress={() => { setSetupStatus('household', people.length ? 'complete' : 'in-progress'); router.replace('/setup'); }} />
    <TertiaryButton label="Set up later" onPress={() => router.replace('/(tabs)')} />
  </ScreenLayout>;
}

const styles = StyleSheet.create({
  content: { gap: spacing.lg, minWidth: 0 }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, minWidth: 0 }, subtitle: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 24, minWidth: 0 }, card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, minWidth: 0, padding: spacing.lg }, person: { alignItems: 'center', flexDirection: 'row', gap: spacing.md, minWidth: 0 }, avatar: { alignItems: 'center', backgroundColor: '#E7F0EA', borderRadius: 28, flexShrink: 0, height: 56, justifyContent: 'center', width: 56 }, avatarText: { color: colors.accent, fontFamily: fontFamily.display, fontSize: 24 }, personCopy: { flex: 1, minWidth: 0 }, name: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.heading, minWidth: 0 }, meta: { color: colors.text.secondary, fontSize: fontSize.body, marginTop: 2, minWidth: 0 }, actions: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, marginTop: spacing.lg, minWidth: 0 }, action: { alignItems: 'center', borderColor: colors.border.default, borderRadius: radius.md, borderWidth: 1, flex: 1, minHeight: 44, minWidth: 120, justifyContent: 'center' }, actionText: { color: colors.accent, fontSize: fontSize.body, fontWeight: '700' }, add: { alignItems: 'center', borderColor: colors.border.default, borderRadius: radius.md, borderStyle: 'dashed', borderWidth: 1, flexDirection: 'row', gap: spacing.sm, justifyContent: 'center', minHeight: 54, minWidth: 0 }, addText: { color: colors.accent, fontSize: fontSize.body, fontWeight: '700' }, empty: { alignItems: 'center', backgroundColor: colors.surface.card, borderRadius: radius.xl, gap: spacing.sm, minWidth: 0, padding: spacing.xl }, emptyTitle: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.heading, textAlign: 'center' }, emptyCopy: { color: colors.text.secondary, fontSize: fontSize.body, textAlign: 'center' }, error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 },
});
