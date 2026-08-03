import { useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Alert, StyleSheet, Text, View } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, SecondaryButton } from '../../src/components/AppUI';
import { getConsentStateV1, grantResearchParticipationV1, listPatientRecordsV1, RESEARCH_CONSENT_PURPOSE, withdrawResearchParticipationV1, type V1ConsentState, type V1PatientRecord } from '../../src/lib/v1Caregiver';
import { colors, contentColumn, fontFamily, fontSize, radius, spacing } from '../../src/theme';

export default function ResearchSettings() {
  const router = useRouter();
  const [patient, setPatient] = useState<V1PatientRecord | null>(null);
  const [state, setState] = useState<V1ConsentState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { void listPatientRecordsV1().then((people) => { const first = people[0] || null; setPatient(first); return first ? getConsentStateV1(first.patientId) : null; }).then((next) => { if (next) setState(next); }).catch((cause) => setError(cause instanceof Error ? cause.message : 'Could not load research preference.')); }, []);
  const current = state?.consents.find((consent) => consent.purpose === RESEARCH_CONSENT_PURPOSE && consent.status === 'granted' && !consent.withdrawnAt);
  const update = async (grant: boolean) => { if (!patient) return; setBusy(true); try { if (grant) await grantResearchParticipationV1(patient.patientId); else await withdrawResearchParticipationV1(patient.patientId); setState(await getConsentStateV1(patient.patientId)); Alert.alert(grant ? 'Research interest saved' : 'Research preference saved', grant ? 'You will be contacted before any optional study begins.' : 'Your care experience is unchanged.'); } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not save research preference.'); } finally { setBusy(false); } };
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Research" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Research participation</Text><Text style={styles.copy}>Research is optional and separate from Reflexion care. Choosing not to participate does not change the care your loved one receives.</Text>{error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}{!patient || !state ? <ActivityIndicator color={colors.accent} /> : <><View style={styles.card}><Text style={styles.label}>Current choice</Text><Text style={styles.copy}>{current ? 'Interested in optional research' : 'Not participating'}</Text><Text style={styles.copy}>A future study may ask to use de-identified interaction records. No identifying information is shared with researchers, and you can change your choice at any time.</Text></View>{busy ? <ActivityIndicator color={colors.accent} /> : current ? <SecondaryButton label="Withdraw research participation" onPress={() => void update(false)} /> : <PrimaryButton label="I’m interested" onPress={() => void update(true)} />}</>}</ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, copy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 }, error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 }, card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.lg, borderWidth: 1, gap: spacing.md, padding: spacing.lg }, label: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '700' } });
