import { useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { AppHeader, ChoiceCard, PrimaryButton, ScreenLayout } from '../../src/components/AppUI';
import { Field } from '../../src/components/Field';
import { createAwayPeriodV1, listPatientRecordsV1, type V1PatientRecord } from '../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';

export default function AwayModeScreen() {
  const router = useRouter();
  const [people, setPeople] = useState<V1PatientRecord[]>([]);
  const [patientId, setPatientId] = useState('');
  const [startsOn, setStartsOn] = useState(new Date().toISOString().slice(0, 10));
  const [endsOn, setEndsOn] = useState(new Date(Date.now() + 7 * 86_400_000).toISOString().slice(0, 10));
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [saved, setSaved] = useState(false);

  useEffect(() => { void listPatientRecordsV1().then((next) => { setPeople(next); setPatientId(next[0]?.patientId || ''); }).catch(() => setError('We could not load loved ones. Check your connection and try again.')).finally(() => setLoading(false)); }, []);

  const save = async () => {
    if (!patientId) { setError('Choose a loved one first.'); return; }
    if (!/^\d{4}-\d{2}-\d{2}$/.test(startsOn) || !/^\d{4}-\d{2}-\d{2}$/.test(endsOn) || endsOn < startsOn) { setError('Enter a valid date range in YYYY-MM-DD format.'); return; }
    setBusy(true); setError('');
    try { await createAwayPeriodV1(patientId, { startsOn, endsOn, timezone: people.find((person) => person.patientId === patientId)?.timezone || 'Asia/Singapore', reason: reason.trim() || undefined }); setSaved(true); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Away Mode could not be saved.'); }
    finally { setBusy(false); }
  };

  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Away Mode" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Mark as away</Text><Text style={styles.subtitle}>Pause missed-interaction reminders for a date range. This changes notification interpretation; it does not delete sessions or messages.</Text>{loading ? <ActivityIndicator color={colors.accent} /> : null}{error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}{saved ? <View style={styles.success}><Text style={styles.successTitle}>Away period saved</Text><Text style={styles.copy}>The selected loved one’s dates are recorded. Existing data remains available.</Text></View> : null}{!loading ? <><Text style={styles.label}>For</Text>{people.map((person) => <ChoiceCard key={person.patientId} icon="user" title={person.displayName} description="Apply the away period to this loved one." selected={patientId === person.patientId} onPress={() => setPatientId(person.patientId)} />)}<Field label="Starts on" autoCapitalize="none" onChangeText={setStartsOn} placeholder="YYYY-MM-DD" value={startsOn} /><Field label="Ends on" autoCapitalize="none" onChangeText={setEndsOn} placeholder="YYYY-MM-DD" value={endsOn} /><Field label="Reason (optional)" multiline onChangeText={setReason} placeholder="Travel, respite or another reason" value={reason} />{busy ? <ActivityIndicator color={colors.accent} /> : <PrimaryButton label="Save away period" onPress={() => void save()} />}</> : null}</ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, subtitle: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 }, label: { color: colors.text.primary, fontSize: fontSize.body, fontWeight: '700' }, error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 }, success: { backgroundColor: colors.status.greenBg, borderColor: '#CFE4D0', borderRadius: radius.xl, borderWidth: 1, gap: spacing.sm, padding: spacing.lg }, successTitle: { color: colors.status.green, fontSize: fontSize.bodyLarge, fontWeight: '700' }, copy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 } });
