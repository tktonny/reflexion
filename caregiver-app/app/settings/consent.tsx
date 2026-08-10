import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, Alert, StyleSheet, Text, View } from 'react-native';

import { useCaregiver } from '../../src/architecture/CaregiverContext';
import { AppHeader, ChoiceCard, PrimaryButton, ScreenLayout, SecondaryButton, SelectionButton, TertiaryButton } from '../../src/components/AppUI';
import { BotanicalSprig } from '../../src/components/Illustrations';
import { CHECKIN_CONSENT_PURPOSE, getCarePlanV1, getConsentStateV1, listPatientRecordsV1, putCarePlanV1, recordCheckInConsentV1, type V1CarePlan, type V1ConsentState, type V1PatientRecord } from '../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';

type ConsentChoice = 'granted' | 'declined' | 'help';
type ProductControl = 'active' | 'paused';

function consentStatus(state: V1ConsentState | null) {
  const record = state?.consents.find((consent) => consent.purpose === CHECKIN_CONSENT_PURPOSE);
  if (record?.status === 'granted' && !record.withdrawnAt) return { key: 'accepted' as const, label: 'Accepted', record };
  if (record?.status === 'withdrawn' || record?.withdrawnAt) return { key: 'withdrawn' as const, label: 'Withdrawn', record };
  if (record?.status === 'declined') return { key: 'declined' as const, label: 'Declined', record };
  return { key: 'pending' as const, label: 'Pending', record };
}

function displayDate(value: string | null | undefined) {
  if (!value) return '';
  return new Intl.DateTimeFormat('en-SG', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value));
}

export default function ConsentSettings() {
  const router = useRouter();
  const { lovedOneId: lovedOneIdParam } = useLocalSearchParams<{ lovedOneId?: string }>();
  const lovedOneId = typeof lovedOneIdParam === 'string' ? lovedOneIdParam : undefined;
  const { setSetupStatus } = useCaregiver();
  const [patient, setPatient] = useState<V1PatientRecord | null>(null);
  const [state, setState] = useState<V1ConsentState | null>(null);
  const [carePlan, setCarePlan] = useState<V1CarePlan | null>(null);
  const [productControl, setProductControl] = useState<ProductControl>('active');
  const [reviewOpen, setReviewOpen] = useState(false);
  const [choice, setChoice] = useState<ConsentChoice | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    setError('');
    if (!lovedOneId) {
      setPatient(null);
      setState(null);
      setCarePlan(null);
      setError('Open consent from a loved-one profile so the correct person can be selected.');
      return;
    }

    const people = await listPatientRecordsV1();
    const selected = people.find((person) => person.patientId === lovedOneId) || null;
    setPatient(selected);
    if (!selected) {
      setState(null);
      setCarePlan(null);
      setError('This loved one is unavailable. Refresh and try again.');
      return;
    }
    const [next, nextPlan] = await Promise.all([getConsentStateV1(selected.patientId), getCarePlanV1(selected.patientId)]);
    setState(next);
    setCarePlan(nextPlan);
    const configuredControl = nextPlan?.communicationPreferences?.productControl;
    setProductControl(configuredControl === 'paused' ? 'paused' : 'active');
    setSetupStatus('consent-control', consentStatus(next).key === 'pending' ? 'in-progress' : 'complete');
  }, [lovedOneId, setSetupStatus]);

  useEffect(() => { void refresh().catch((cause) => setError(cause instanceof Error ? cause.message : 'Could not load consent status.')); }, [refresh]);

  const current = useMemo(() => consentStatus(state), [state]);

  const updateChoice = async (next: Exclude<ConsentChoice, 'help'>) => {
    if (!patient) return;
    const nextLabel = next === 'granted' ? 'accept' : 'decline';
    const description = next === 'granted'
      ? `${patient.displayName} is choosing to accept home conversations and routine support.`
      : `${patient.displayName} is choosing not to accept home conversations and routine support.`;
    Alert.alert('Record their choice?', `${description} Confirm only after reviewing it with them.`, [
      { text: 'Go back', style: 'cancel' },
      { text: `Record ${nextLabel}`, onPress: () => { void (async () => {
        setBusy(true); setError(''); setMessage('');
        try {
          await recordCheckInConsentV1(patient.patientId, next);
          await refresh();
          setReviewOpen(false); setChoice(null);
          setMessage(next === 'granted' ? 'Consent was recorded as accepted.' : 'Consent was recorded as declined.');
        } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not record their consent choice.'); }
        finally { setBusy(false); }
      })(); } },
    ]);
  };

  const withdraw = () => {
    if (!patient) return;
    Alert.alert('Withdraw consent?', 'Confirm this with your loved one. New daily check-ins will stop until consent is accepted again. Existing records are not deleted by this action.', [
      { text: 'Go back', style: 'cancel' },
      { text: 'Withdraw', style: 'destructive', onPress: () => { void (async () => {
        setBusy(true); setError(''); setMessage('');
        try { await recordCheckInConsentV1(patient.patientId, 'withdrawn'); await refresh(); setMessage('Consent was recorded as withdrawn.'); }
        catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not withdraw consent.'); }
        finally { setBusy(false); }
      })(); } },
    ]);
  };

  const selectChoice = (next: ConsentChoice) => {
    setChoice(next);
    if (next === 'help') {
      setReviewOpen(false); setChoice(null); setMessage('No choice was recorded. You can return here when your loved one is ready or needs help.');
    }
  };

  const saveProductControl = async (next: ProductControl) => {
    if (!patient || next === productControl) return;
    setBusy(true); setError(''); setMessage('');
    try {
      const currentPlan = carePlan || await getCarePlanV1(patient.patientId);
      const communicationPreferences = {
        ...(currentPlan?.communicationPreferences || {}),
        productControl: next,
      };
      const saved = await putCarePlanV1(patient.patientId, currentPlan?.version || 0, {
        dailyRoutine: currentPlan?.dailyRoutine || {},
        communicationPreferences,
        safetyNotes: currentPlan?.safetyNotes || null,
      });
      setCarePlan(saved);
      setProductControl(next);
      setMessage(next === 'paused' ? 'Reflexion conversations are paused for now.' : 'Reflexion conversations are active again.');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Product control could not be updated.');
    } finally { setBusy(false); }
  };

  return <ScreenLayout contentContainerStyle={styles.content}>
    <AppHeader title="Consent & control" onBack={() => router.back()} />
    <View style={styles.heroMark}><BotanicalSprig size={70} /><Text style={styles.heroHeart}>♡</Text></View>
    <Text accessibilityRole="header" style={styles.title}>Your choice, always</Text>
    <Text style={styles.copy}>{patient ? `${patient.displayName} is in control. You can help explain the information and record their choice here or on their Reflexion Mirror.` : 'Your loved one is in control. You can help explain the information and record their choice here or on their Reflexion Mirror.'}</Text>
    {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
    {message ? <View style={styles.messageCard}><View style={styles.messageRow}><Text style={styles.messageIcon}>✓</Text><View style={styles.messageCopy}><Text style={styles.messageTitle}>{message.startsWith('No choice') ? 'Consent remains pending' : 'Choice recorded'}</Text><Text style={styles.messageText}>{message}</Text></View></View></View> : null}
    {!patient || !state ? (error ? null : <ActivityIndicator color={colors.accent} />) : <>
      <View style={styles.statusCard}><View style={styles.statusCopy}><Text style={styles.cardLabel}>{patient.displayName}</Text><Text style={styles.statusLabel}>Current status</Text><Text style={[styles.status, current.key === 'accepted' && styles.statusAccepted, current.key === 'declined' && styles.statusDeclined, current.key === 'withdrawn' && styles.statusWithdrawn]}>{current.label}</Text>{current.record?.signedAt || current.record?.withdrawnAt ? <Text style={styles.timestamp}>{displayDate(current.record.signedAt || current.record.withdrawnAt)}</Text> : <Text style={styles.timestamp}>No choice recorded yet</Text>}<Text style={styles.cardCopy}>Required purpose: home conversations and routine support.</Text><SecondaryButton label="View consent details" onPress={() => setDetailsOpen((open) => !open)} /></View></View>
      <View style={styles.controlCard}><Text style={styles.cardLabel}>Product control</Text><Text style={styles.cardCopy}>Pausing is temporary and does not withdraw {patient.displayName}’s consent. {patient.displayName} can also stop an active conversation from the Mirror.</Text><SelectionButton label="Active · conversations and routine support can run" onPress={() => void saveProductControl('active')} selected={productControl === 'active'} /><SelectionButton label="Paused · temporarily stop ordinary support" onPress={() => void saveProductControl('paused')} selected={productControl === 'paused'} /></View>
      {detailsOpen ? <View style={styles.detailsCard}><Text style={styles.cardLabel}>What this is for</Text><View style={styles.detailRow}><Text style={styles.detailTitle}>Home conversations</Text><Text style={styles.cardCopy}>Support recorded conversations and session summaries.</Text></View><View style={styles.detailRow}><Text style={styles.detailTitle}>Routine support</Text><Text style={styles.cardCopy}>Support gentle reminders and record responses.</Text></View><View style={styles.detailRow}><Text style={styles.detailTitle}>Separate choices</Text><Text style={styles.cardCopy}>Product consent is separate from optional research participation.</Text></View></View> : null}
      {reviewOpen ? <View style={styles.reviewCard}><Text style={styles.cardLabel}>Review with {patient.displayName}</Text><Text style={styles.cardCopy}>The choice belongs to your loved one. Choose an option only after explaining what it means and checking what they want.</Text><ChoiceCard icon="check-circle" title="Accept" description="Allow home conversations and routine support." selected={choice === 'granted'} onPress={() => selectChoice('granted')} /><ChoiceCard icon="x-circle" title="Decline" description="Do not allow home conversations and routine support." selected={choice === 'declined'} onPress={() => selectChoice('declined')} /><ChoiceCard icon="help-circle" title="Needs help / decide later" description="Keep consent pending and record no choice." selected={choice === 'help'} onPress={() => selectChoice('help')} />{choice && choice !== 'help' ? <PrimaryButton disabled={busy} label={busy ? 'Recording…' : 'Confirm choice'} onPress={() => void updateChoice(choice)} /> : null}<TertiaryButton label="Cancel" onPress={() => { setReviewOpen(false); setChoice(null); }} /></View> : null}
      {!reviewOpen ? <PrimaryButton label="Review or update consent" onPress={() => { setMessage(''); setReviewOpen(true); }} /> : null}
      {current.key === 'accepted' && !reviewOpen ? <SecondaryButton disabled={busy} label="Withdraw consent" onPress={withdraw} /> : null}
      {!reviewOpen ? <TertiaryButton label="Learn more" onPress={() => setDetailsOpen((open) => !open)} /> : null}
    </>}
  </ScreenLayout>;
}

const styles = StyleSheet.create({
  content: { gap: spacing.lg, minWidth: 0 },
  heroMark: { alignItems: 'flex-start', flexDirection: 'row', height: 74, minWidth: 0, paddingTop: spacing.sm },
  heroHeart: { color: '#A9CBA3', fontSize: 46, lineHeight: 54, marginLeft: spacing.sm },
  title: { color: '#0D514C', fontFamily: fontFamily.display, fontSize: fontSize.display, lineHeight: 42, minWidth: 0 },
  copy: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 25, minWidth: 0 },
  error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22, minWidth: 0 },
  messageCard: { backgroundColor: '#EAF4E7', borderColor: '#CDE2C8', borderRadius: radius.lg, borderWidth: 1, minWidth: 0, padding: spacing.lg },
  messageRow: { alignItems: 'flex-start', flexDirection: 'row', gap: spacing.md, minWidth: 0 },
  messageIcon: { color: colors.status.green, fontSize: fontSize.heading, fontWeight: '800', lineHeight: 27 },
  messageCopy: { flex: 1, gap: spacing.xs, minWidth: 0 },
  messageTitle: { color: colors.status.green, flexShrink: 1, fontSize: fontSize.bodyLarge, fontWeight: '800', lineHeight: 22, minWidth: 0 },
  messageText: { color: colors.status.green, flexShrink: 1, fontSize: fontSize.body, lineHeight: 21, minWidth: 0 },
  statusCard: { backgroundColor: colors.surface.card, borderColor: '#CDE2D6', borderRadius: radius.xl, borderWidth: 1, minWidth: 0, padding: spacing.xl },
  controlCard: { backgroundColor: '#F7FBF8', borderColor: '#D8E8DC', borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, minWidth: 0, padding: spacing.xl },
  statusCopy: { gap: spacing.sm, minWidth: 0 },
  cardLabel: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 23, minWidth: 0 },
  statusLabel: { color: colors.text.secondary, fontSize: fontSize.caption, fontWeight: '700', lineHeight: 18, marginTop: spacing.sm },
  status: { color: '#B26A00', flexShrink: 1, fontSize: fontSize.heading, fontWeight: '700', lineHeight: 27, minWidth: 0 },
  statusAccepted: { color: colors.accent },
  statusDeclined: { color: colors.status.red },
  statusWithdrawn: { color: colors.text.secondary },
  timestamp: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 21 },
  cardCopy: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22, minWidth: 0 },
  detailsCard: { backgroundColor: '#F7FBF8', borderColor: '#D8E8DC', borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, minWidth: 0, padding: spacing.xl },
  detailRow: { borderTopColor: '#D8E8DC', borderTopWidth: 1, gap: spacing.xs, minWidth: 0, paddingTop: spacing.md },
  detailTitle: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 22, minWidth: 0 },
  reviewCard: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, minWidth: 0, padding: spacing.xl },
});
