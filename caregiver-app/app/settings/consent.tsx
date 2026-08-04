import { useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Alert, StyleSheet, Text, View } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, SecondaryButton } from '../../src/components/AppUI';
import { getConsentStateV1, listPatientRecordsV1, withdrawCheckInConsentV1, type V1ConsentState, type V1PatientRecord } from '../../src/lib/v1Caregiver';
import { colors, contentColumn, fontFamily, fontSize, radius, spacing } from '../../src/theme';

export default function ConsentSettings() {
  const router = useRouter();
  const [patient, setPatient] = useState<V1PatientRecord | null>(null);
  const [state, setState] = useState<V1ConsentState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    void listPatientRecordsV1().then((people) => {
      const first = people[0] || null;
      setPatient(first);
      return first ? getConsentStateV1(first.patientId) : null;
    }).then((next) => { if (next) setState(next); }).catch((cause) => setError(cause instanceof Error ? cause.message : 'Could not load consent status.'));
  }, []);

  const withdraw = () => {
    if (!patient) return;
    Alert.alert('Withdraw consent?', 'New daily check-ins will stop until consent is provided again. Existing records are not deleted by this action.', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Withdraw', style: 'destructive', onPress: () => void (async () => {
        setBusy(true);
        try { await withdrawCheckInConsentV1(patient.patientId); setState(await getConsentStateV1(patient.patientId)); }
        catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not update consent.'); }
        finally { setBusy(false); }
      })() },
    ]);
  };

  const currentStatus = state?.consents.find((consent) => consent.purpose === state.requiredPurposes[0]);
  const status = currentStatus?.status === 'granted' && !currentStatus.withdrawnAt
    ? 'accepted'
    : currentStatus?.status === 'withdrawn' || currentStatus?.withdrawnAt
      ? 'withdrawn'
      : currentStatus?.status === 'declined'
        ? 'declined'
        : 'pending';

  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Consent & control" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Older-Adult Consent & Control</Text><Text style={styles.copy}>Consent for home conversations is separate from optional research. Reflexion never uses this status to make a health conclusion.</Text>{error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}{!patient || !state ? <ActivityIndicator color={colors.accent} /> : <><View style={styles.card}><Text style={styles.label}>{patient.displayName}</Text><Text style={styles.status}>{status === 'accepted' ? 'Consent is accepted' : status === 'declined' ? 'Consent was declined' : status === 'withdrawn' ? 'Consent was withdrawn' : 'Consent is pending'}</Text><Text style={styles.copy}>Required purpose: home conversations and routine support.</Text></View>{busy ? <ActivityIndicator color={colors.accent} /> : status === 'accepted' ? <SecondaryButton label="Withdraw consent" onPress={withdraw} /> : <Text style={styles.help}>Ask your loved one to review the consent on the Mirror or with the care team. A caregiver cannot override their choice.</Text>}</>}</ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, copy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 }, error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 }, card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.lg, borderWidth: 1, gap: spacing.sm, padding: spacing.lg }, label: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '700' }, status: { color: colors.accent, fontSize: fontSize.bodyLarge, fontWeight: '700' }, help: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 } });
