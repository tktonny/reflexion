import { Feather } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Alert, StyleSheet, Text, View } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, SecondaryButton, TertiaryButton } from '../components/AppUI';
import { BotanicalSprig } from '../components/Illustrations';
import { declineResearchParticipationV1, getConsentStateV1, grantResearchParticipationV1, listPatientRecordsV1, RESEARCH_CONSENT_PURPOSE, withdrawResearchParticipationV1, type V1ConsentState, type V1PatientRecord } from '../lib/v1Caregiver';
import { V1ApiError } from '../lib/v1Client';
import { colors, fontFamily, fontSize, radius, spacing } from '../theme';

type ResearchData = { patient: V1PatientRecord | null; state: V1ConsentState | null };

function useResearchData(): ResearchData & { loading: boolean; error: string; reload: () => void } {
  const [patient, setPatient] = useState<V1PatientRecord | null>(null);
  const [state, setState] = useState<V1ConsentState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [reloadKey, setReloadKey] = useState(0);
  const reload = useCallback(() => setReloadKey((current) => current + 1), []);
  useEffect(() => {
    setLoading(true);
    setError('');
    void listPatientRecordsV1().then((people) => { const next = people[0] || null; setPatient(next); return next ? getConsentStateV1(next.patientId) : null; }).then((next) => { if (next) setState(next); }).catch((cause) => setError(cause instanceof V1ApiError ? cause.message : 'Research status could not be loaded.')).finally(() => setLoading(false));
  }, [reloadKey]);
  return { patient, state, loading, error, reload };
}

function researchStatus(state: V1ConsentState | null) {
  const consent = state?.consents.find((item) => item.purpose === RESEARCH_CONSENT_PURPOSE);
  if (!consent) return 'Not invited';
  if (consent?.status === 'granted' && !consent.withdrawnAt) return 'Consented';
  if (consent?.status === 'withdrawn' || consent?.withdrawnAt) return 'Withdrawn';
  if (consent?.status === 'declined') return 'Declined';
  return 'Invitation pending';
}

function StatusCard({ patient, state }: { patient: V1PatientRecord | null; state: V1ConsentState | null }) {
  const current = researchStatus(state);
  return <View style={styles.statusCard}><View style={styles.statusCopy}><Text style={styles.cardLabel}>Current status</Text><Text style={[styles.status, current === 'Not invited' || current === 'Invitation pending' ? styles.statusPending : null]}>{current}</Text><Text style={styles.cardCopy}>{patient ? current === 'Not invited' ? `No active study invitation is available for ${patient.displayName} in this account.` : `Research participation status for ${patient.displayName}.` : 'Add a loved one before reviewing research participation.'}</Text></View><View style={styles.statusMark}><Feather color={current === 'Not invited' || current === 'Invitation pending' ? colors.alertAccent.laterThanUsual : colors.accent} name={current === 'Consented' ? 'check-circle' : 'book-open'} size={30} /></View></View>;
}

function PartnerHeader({ compact = false }: { compact?: boolean }) {
  return <View accessibilityLabel="Reflexion research collaboration with NUS Yong Loo Lin School of Medicine and NUHS" style={[styles.partnerHeader, compact && styles.partnerHeaderCompact]}>
    <View style={styles.partnerTop}>
      <View style={styles.partnerBrand}><View style={styles.reflexionRow}><Feather color={colors.accent} name="feather" size={20} /><Text style={styles.partnerName}>Reflexion</Text></View><Text style={styles.partnerCaption}>Research collaboration</Text></View>
      <View style={styles.partnerMarks}><View style={styles.institutionRow}><Text style={styles.nusMark}>NUS</Text><Text style={styles.institutionText}>Yong Loo Lin School of Medicine</Text></View><View style={styles.institutionRow}><Text style={styles.nuhsMark}>NUHS</Text><Text style={styles.institutionText}>National University Health System</Text></View></View>
    </View>
  </View>;
}

function OptionalCard() {
  return <View style={styles.optionalCard}><View style={styles.optionalHeader}><Text style={styles.cardLabel}>Participation is completely optional</Text><Feather color={colors.accent} name="heart" size={22} /></View>{['It will not affect Reflexion’s features.', 'You can change your mind anytime.', 'Research is separate from normal Reflexion use.'].map((item) => <View key={item} style={styles.bullet}><Feather color={colors.accent} name="check-circle" size={19} /><Text style={styles.cardCopy}>{item}</Text></View>)}</View>;
}

export function ResearchOverviewScreen() {
  const router = useRouter();
  const { patient, state, loading, error, reload } = useResearchData();
  const current = researchStatus(state);
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Research participation" onBack={() => router.back()} /><PartnerHeader /><Text accessibilityRole="header" style={styles.title}>Help shape the future{ '\n' }of healthy ageing</Text><Text style={styles.subtitle}>Reflexion is working with NUS Yong Loo Lin School of Medicine and NUHS to understand how everyday conversations may support research into healthy ageing and cognitive health.</Text>{loading ? <ActivityIndicator color={colors.accent} /> : null}{error ? <><Text accessibilityRole="alert" style={styles.error}>{error}</Text><SecondaryButton label="Try again" onPress={reload} /></> : null}{!loading && !error ? <><OptionalCard /><StatusCard patient={patient} state={state} /><Text style={styles.sectionTitle}>About the study</Text><View style={styles.aboutRow}><Feather color={colors.accent} name="message-circle" size={24} /><Text style={styles.cardCopy}>The collaboration is interested in everyday conversations and routines over time.</Text></View><View style={styles.aboutRow}><Feather color={colors.accent} name="shield" size={24} /><Text style={styles.cardCopy}>Your choice is optional and separate from ordinary Reflexion use.</Text></View><PrimaryButton label="Learn more about the study" onPress={() => router.push('/research/study')} />{current === 'Not invited' ? <SecondaryButton label="View status in Settings" disabled={!patient} onPress={() => router.replace('/settings/research')} /> : current === 'Declined' || current === 'Withdrawn' ? <SecondaryButton label="View participation status" onPress={() => router.replace('/settings/research')} /> : <SecondaryButton label="Review participation choice" disabled={!patient} onPress={() => router.push('/research/study')} />}<TertiaryButton label="Decide later" onPress={() => router.back()} /></> : null}</ScreenLayout>;
}

export function ResearchStudyScreen() {
  const router = useRouter();
  const { patient, state, loading, error, reload } = useResearchData();
  const [actionError, setActionError] = useState('');
  const current = researchStatus(state);
  const chooseToParticipate = () => {
    if (!patient || current !== 'Invitation pending') return;
    Alert.alert('Record participation?', `Confirm this choice with ${patient.displayName}. Their choice belongs to them, even when you are helping on this phone.`, [
      { text: 'Go back', style: 'cancel' },
      { text: 'Record participation', onPress: () => { void grantResearchParticipationV1(patient.patientId).then(() => router.replace({ pathname: '/research/confirmation', params: { status: 'Consented' } })).catch((cause) => setActionError(cause instanceof V1ApiError ? cause.message : 'Research participation could not be recorded.')); } },
    ]);
  };
  const decline = () => {
    if (!patient || current !== 'Invitation pending') return;
    Alert.alert('Record no participation?', `Confirm this choice with ${patient.displayName}. Their choice belongs to them, even when you are helping on this phone.`, [
      { text: 'Go back', style: 'cancel' },
      { text: 'Record decline', style: 'destructive', onPress: () => { void declineResearchParticipationV1(patient.patientId).then(() => router.replace({ pathname: '/research/confirmation', params: { status: 'Declined' } })).catch((cause) => setActionError(cause instanceof V1ApiError ? cause.message : 'Research choice could not be recorded.')); } },
    ]);
  };
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Research participation" onBack={() => router.back()} /><PartnerHeader compact /><Text accessibilityRole="header" style={styles.title}>Research participation</Text><Text style={styles.subtitle}>Reflexion is working with NUS Yong Loo Lin School of Medicine and NUHS. Optional research is separate from normal Reflexion use, and choosing not to participate does not change the product.</Text>{loading ? <ActivityIndicator color={colors.accent} /> : null}{error || actionError ? <Text accessibilityRole="alert" style={styles.error}>{error || actionError}</Text> : null}{error ? <SecondaryButton label="Try again" onPress={reload} /> : null}{!loading && !error ? <><View style={styles.studyCard}><Text style={styles.cardCopy}>This page provides the current collaboration information available in Reflexion. A choice belongs to the loved one, and the caregiver can help explain it.</Text><OptionalCard /></View><StatusCard patient={patient} state={state} />{current === 'Invitation pending' ? <><PrimaryButton disabled={!patient} label="Choose to participate" onPress={chooseToParticipate} /><SecondaryButton disabled={!patient} label="Do not participate" onPress={decline} /></> : null}<SecondaryButton label="Decide later" onPress={() => router.replace('/research/overview')} /></> : null}</ScreenLayout>;
}

export function ResearchConfirmationScreen() {
  const router = useRouter();
  const { patient, state, loading, error, reload } = useResearchData();
  const { status } = useLocalSearchParams<{ status?: string }>();
  const recordedStatus = status || researchStatus(state);
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Research participation" onBack={() => router.back()} /><PartnerHeader compact /><Text accessibilityRole="header" style={styles.title}>{recordedStatus === 'Consented' ? 'Participation recorded' : 'Research choice recorded'}</Text><Text style={styles.subtitle}>Optional research is separate from ordinary Reflexion use. Your loved one’s choice can be reviewed in Settings.</Text>{loading ? <ActivityIndicator color={colors.accent} /> : null}{error ? <><Text accessibilityRole="alert" style={styles.error}>{error}</Text><SecondaryButton label="Try again" onPress={reload} /></> : null}{!loading && !error ? <><View style={styles.statusCard}><Text style={styles.cardLabel}>{patient?.displayName || 'Loved one'}</Text><Text style={styles.status}>{recordedStatus}</Text><Text style={styles.cardCopy}>No change to ordinary Reflexion use is made by optional research participation.</Text></View><PrimaryButton label="Back to Settings" onPress={() => router.replace('/settings/research')} /></> : null}</ScreenLayout>;
}

export function ResearchStatusScreen() {
  const router = useRouter();
  const { patient, state, loading, error, reload } = useResearchData();
  const current = researchStatus(state);
  const withdraw = () => {
    if (!patient || current !== 'Consented') return;
    Alert.alert('Withdraw research participation?', 'This does not change ordinary Reflexion use. Confirm this choice with your loved one before withdrawing.', [{ text: 'Cancel', style: 'cancel' }, { text: 'Withdraw', style: 'destructive', onPress: () => { void withdrawResearchParticipationV1(patient.patientId).then(() => router.replace('/settings/research')).catch(() => undefined); } }]);
  };
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Research participation" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Research participation</Text><Text style={styles.subtitle}>Optional research is separate from ordinary Reflexion use. Choosing not to participate does not change the product.</Text>{loading ? <ActivityIndicator color={colors.accent} /> : null}{error ? <><Text accessibilityRole="alert" style={styles.error}>{error}</Text><SecondaryButton label="Try again" onPress={reload} /></> : null}{!loading && !error ? <><StatusCard patient={patient} state={state} /><View style={styles.studyCard}><Text style={styles.cardLabel}>Reflexion Healthy Ageing Study</Text><Text style={styles.cardCopy}>Reflexion is working with NUS Yong Loo Lin School of Medicine and NUHS on research into healthy ageing and cognitive health.</Text></View><PrimaryButton label="Learn more about the study" onPress={() => router.push('/research/study')} />{current === 'Consented' ? <SecondaryButton label="Withdraw / leave study" onPress={withdraw} /> : null}</> : null}</ScreenLayout>;
}

const styles = StyleSheet.create({
  content: { gap: spacing.lg, minWidth: 0 },
  partnerHeader: { backgroundColor: '#F7FBF8', borderColor: '#DCEBE1', borderRadius: radius.xl, borderWidth: 1, minHeight: 142, minWidth: 0, padding: spacing.lg },
  partnerHeaderCompact: { minHeight: 126 },
  partnerTop: { alignItems: 'flex-start', flexDirection: 'row', gap: spacing.lg, minWidth: 0 },
  partnerBrand: { flex: 1, gap: spacing.sm, minWidth: 0 },
  partnerMarks: { flex: 1.25, gap: spacing.sm, minWidth: 0 },
  reflexionRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.xs, minWidth: 0 },
  partnerName: { color: '#0D514C', flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.heading, fontWeight: '700', lineHeight: 26, minWidth: 0 },
  partnerCaption: { color: colors.text.secondary, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.caption, lineHeight: 18, minWidth: 0 },
  institutionRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, minWidth: 0 },
  nusMark: { color: '#174A84', fontFamily: fontFamily.ui, fontSize: fontSize.bodyLarge, fontWeight: '800', minWidth: 48 },
  nuhsMark: { color: '#0B6A68', fontFamily: fontFamily.ui, fontSize: fontSize.bodyLarge, fontWeight: '800', minWidth: 48 },
  institutionText: { color: colors.text.secondary, flex: 1, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.caption, lineHeight: 18, minWidth: 0 },
  title: { color: '#0D514C', fontFamily: fontFamily.display, fontSize: fontSize.display, lineHeight: 42, minWidth: 0 },
  subtitle: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 25, minWidth: 0 },
  error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22, minWidth: 0 },
  optionalCard: { backgroundColor: '#F4FAF5', borderColor: '#D3E6D7', borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, minWidth: 0, padding: spacing.xl },
  optionalHeader: { alignItems: 'flex-start', flexDirection: 'row', gap: spacing.md, justifyContent: 'space-between', minWidth: 0 },
  bullet: { alignItems: 'flex-start', flexDirection: 'row', gap: spacing.md, minWidth: 0 },
  statusCard: { backgroundColor: colors.surface.card, borderColor: '#CDE2D6', borderRadius: radius.xl, borderWidth: 1, gap: spacing.sm, minWidth: 0, padding: spacing.xl, paddingRight: spacing.xxl, position: 'relative' },
  statusCopy: { gap: spacing.sm, minWidth: 0 },
  statusMark: { position: 'absolute', right: spacing.xl, top: spacing.xl },
  cardLabel: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 23, minWidth: 0 },
  status: { color: colors.accent, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.heading, fontWeight: '700', lineHeight: 28, minWidth: 0 },
  statusPending: { color: colors.alertAccent.laterThanUsual },
  cardCopy: { color: colors.text.secondary, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 22, minWidth: 0 },
  sectionTitle: { color: colors.text.primary, fontFamily: fontFamily.ui, fontSize: fontSize.heading, fontWeight: '700', lineHeight: 27, marginTop: spacing.sm, minWidth: 0 },
  aboutRow: { alignItems: 'flex-start', flexDirection: 'row', gap: spacing.md, minWidth: 0 },
  studyCard: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.lg, minWidth: 0, padding: spacing.xl },
});
