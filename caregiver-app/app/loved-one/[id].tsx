import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, Linking, StyleSheet, Text, useWindowDimensions, View } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, SecondaryButton, SurfaceCard } from '../../src/components/AppUI';
import { LovedAvatar, LovedBotanical, LovedChip, LovedIconCircle } from '../../src/components/LovedOneVisuals';
import { MotionPressable } from '../../src/components/Motion';
import { loadCaregiverHome, type CaregiverHomePatient } from '../../src/lib/v1Caregiver';
import { usePatientStatusV1 } from '../../src/lib/v1Client';
import { formatLastInteraction, getConversationsTodayText, getObjectiveInteractionState, getTechnicalNote } from '../../src/lib/v1Status';
import { colors, fontFamily, fontSize, radius, spacing, typography } from '../../src/theme';

function ExploreRow({ icon, label, onPress, wide = false, fullWidth = false }: { icon: 'message-circle' | 'bar-chart-2' | 'trending-up' | 'clock' | 'download'; label: string; onPress: () => void; wide?: boolean; fullWidth?: boolean }) {
  return <MotionPressable accessibilityLabel={label} accessibilityRole="button" feedback="card" onPress={onPress} style={[styles.exploreRow, wide && !fullWidth && styles.exploreRowWide]}>
    <View style={styles.exploreIcon}><Feather color={colors.accent} name={icon} size={20} /></View>
    <Text style={styles.exploreLabel}>{label}</Text>
    <Feather color={colors.textDecorative} name="chevron-right" size={20} />
  </MotionPressable>;
}

export default function LovedOneDashboardScreen() {
  const router = useRouter();
  const { id } = useLocalSearchParams<{ id: string }>();
  const { width } = useWindowDimensions();
  const [person, setPerson] = useState<CaregiverHomePatient | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const status = usePatientStatusV1(id);

  const refresh = useCallback(async () => {
    if (!id) return;
    setLoading(true);
    setError('');
    try {
      const home = await loadCaregiverHome();
      setPerson(home.patients.find((item) => item.patientId === id) || null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'This loved one could not be loaded.');
    } finally {
      setLoading(false);
    }
  }, [id]);
  useFocusEffect(useCallback(() => { void refresh(); }, [refresh]));

  if (loading) return <ScreenLayout scroll={false} contentContainerStyle={styles.loading}><ActivityIndicator color={colors.accent} /></ScreenLayout>;
  if (error || !person) return <ScreenLayout><AppHeader title="Loved one" onBack={() => router.back()} /><Text style={styles.title}>This loved one is unavailable</Text><Text style={styles.subtitle}>{error || 'Refresh the household list and try again.'}</Text><PrimaryButton label="Back to Home" onPress={() => router.replace('/(tabs)')} /></ScreenLayout>;

  const facts = status.data;
  const hasDevice = Boolean(person.deviceId);
  const state = person.deviceTechnicalState === 'possible_issue' ? 'device-may-be-offline' : getObjectiveInteractionState(facts, hasDevice);
  const technicalNote = person.deviceTechnicalState === 'possible_issue' ? getTechnicalNote('possible_issue') : facts ? getTechnicalNote(facts.technicalState) : null;
  const nextStep = !hasDevice ? 'Pair the Mirror to start receiving factual updates.' : state === 'needs-your-attention' ? 'Review the latest session or contact your loved one using the usual channel.' : 'Leave a familiar message or review the latest session.';
  const wideExplore = width >= 540;

  const dateLabel = new Intl.DateTimeFormat('en-SG', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' }).format(new Date());
  const summaryCopy = getConversationsTodayText(facts) || 'No conversations have been recorded today.';
  const statusCopy = state === 'interaction-recorded-today' ? 'Interacted today' : state === 'needs-your-attention' ? 'Needs your attention' : state === 'device-may-be-offline' ? 'Device may be offline' : 'No interaction yet today';

  return <ScreenLayout contentContainerStyle={styles.content}>
    <AppHeader onBack={() => router.back()} />
    <View style={styles.pageHero}><View style={styles.heroCopy}><Text accessibilityRole="header" style={styles.title}>{person.displayName}</Text><Text style={styles.subtitle}>Today · {dateLabel}</Text></View><LovedBotanical style={styles.botanical} /></View>
    <SurfaceCard style={styles.heroCard}><View style={styles.heroCardTop}><LovedAvatar name={person.displayName} photoUrl={person.profile.photoUrl} size={112} /><View style={styles.heroFacts}><View style={styles.statusLine}><LovedIconCircle icon="check-circle" size={28} /><Text style={styles.statusLabel}>{statusCopy}</Text></View><View style={styles.factRow}><LovedIconCircle icon="calendar" size={38} /><Text style={styles.fact}>Last interaction: {facts?.lastInteractionAt ? formatLastInteraction(facts.lastInteractionAt) : 'No check-in yet.'}</Text></View><View style={styles.factRow}><LovedIconCircle icon="clock" size={38} /><Text style={styles.fact}>{summaryCopy}</Text></View></View><Feather color={colors.text.primary} name="chevron-right" size={25} /></View><View style={styles.deviceRow}><Feather color={person.deviceTechnicalState === 'ok' ? colors.status.green : colors.status.grey} name="monitor" size={18} /><Text style={styles.deviceText}>Device status · {hasDevice ? person.deviceTechnicalState === 'ok' ? `Online${person.lastHeartbeatAt ? ` · ${formatLastInteraction(person.lastHeartbeatAt)}` : ''}` : person.deviceTechnicalState === 'possible_issue' ? 'May be offline' : 'Unknown' : 'Not paired'}</Text></View>{technicalNote ? <Text style={styles.technicalNote}>{technicalNote}</Text> : null}</SurfaceCard>
    <Text style={styles.section}>Today’s summary</Text>
    <SurfaceCard style={styles.contentCard}><View style={styles.cardHeading}><LovedIconCircle icon="eye" size={42} /><Text style={styles.cardHeadingText}>Observed</Text></View><Text style={styles.body}>Recorded interaction information is available for review. {summaryCopy}</Text></SurfaceCard>
    <SurfaceCard style={styles.contentCard}><View style={styles.cardHeading}><LovedIconCircle icon="tag" size={42} /><Text style={styles.cardHeadingText}>Topics we talked about</Text></View><View style={styles.chips}><LovedChip>Morning routine</LovedChip><LovedChip>Family</LovedChip><LovedChip>Past memories</LovedChip><LovedChip>Gardening</LovedChip></View></SurfaceCard>
    <SurfaceCard style={styles.contentCard}><View style={styles.cardHeading}><LovedIconCircle icon="calendar" size={42} /><Text style={styles.cardHeadingText}>Routine activity</Text><LovedIconCircle icon="check" size={42} /></View><Text style={styles.body}>Review the routines recorded for today in the activity history.</Text></SurfaceCard>
    <SurfaceCard style={styles.contentCard}><View style={styles.cardHeading}><LovedIconCircle icon="bar-chart-2" size={42} /><Text style={styles.cardHeadingText}>Compared with usual</Text><LovedChip>About the same</LovedChip></View><Text style={styles.body}>This screen reports recorded activity and does not infer wellbeing.</Text></SurfaceCard>
    <SurfaceCard style={styles.contentCard}><View style={styles.cardHeading}><LovedIconCircle icon="feather" size={42} /><Text style={styles.cardHeadingText}>Recommended next step</Text></View><View style={styles.recommendRow}><Text style={[styles.body, styles.recommendText]}>{nextStep}</Text><PrimaryButton label="Start a chat" onPress={() => router.push(hasDevice ? `/chat/${person.patientId}/compose` : `/device/${person.patientId}/pairing`)} /></View></SurfaceCard>
    {person.needsConsent ? <View style={styles.consentCard}><Feather color={colors.status.amber} name="info" size={20} /><View style={styles.consentCopy}><Text style={styles.consentTitle}>Consent is still pending</Text><Text style={styles.consentText}>Some conversation information remains unavailable until your loved one reviews the product consent.</Text><MotionPressable accessibilityRole="button" haptic="selection" onPress={() => router.push({ pathname: '/settings/consent', params: { lovedOneId: person.patientId } })} style={styles.inlineAction}><Text style={styles.inlineActionText}>Review consent status</Text><Feather color={colors.accent} name="chevron-right" size={19} /></MotionPressable></View></View> : null}
    <SurfaceCard style={styles.contentCard}><View style={styles.cardHeading}><LovedIconCircle icon="info" size={42} /><Text style={styles.cardHeadingText}>Limitations</Text><Feather color={colors.text.primary} name="chevron-right" size={21} /></View><Text style={styles.body}>Reflexion is not a medical device and does not provide medical advice, diagnosis or treatment.</Text></SurfaceCard>
    <View style={styles.actionRow}><View style={styles.primaryWrap}><PrimaryButton icon={hasDevice ? 'message-circle' : 'monitor'} label={hasDevice ? 'Leave a message' : 'Pair device'} onPress={() => router.push(hasDevice ? `/chat/${person.patientId}/compose` : `/device/${person.patientId}/pairing`)} /></View>{person.profile.phoneNumber ? <View style={styles.callWrap}><SecondaryButton icon="phone" label="Call" onPress={() => { void Linking.openURL(`tel:${person.profile.phoneNumber}`).catch(() => undefined); }} /></View> : null}</View>
    <Text accessibilityRole="header" style={styles.section}>Explore</Text>
    <View style={[styles.exploreGrid, wideExplore && styles.exploreGridWide]}>
      <ExploreRow wide={wideExplore} icon="message-circle" label="Sessions" onPress={() => router.push(`/loved-one/${person.patientId}/sessions`)} />
      <ExploreRow wide={wideExplore} icon="bar-chart-2" label="Weekly summary" onPress={() => router.push(`/loved-one/${person.patientId}/weekly-summary`)} />
      <ExploreRow wide={wideExplore} icon="trending-up" label="Trends" onPress={() => router.push(`/loved-one/${person.patientId}/trends`)} />
      <ExploreRow wide={wideExplore} icon="clock" label="History" onPress={() => router.push(`/loved-one/${person.patientId}/history`)} />
      <View style={styles.exploreFull}><ExploreRow wide={wideExplore} fullWidth icon="download" label="Export summaries" onPress={() => router.push(`/loved-one/${person.patientId}/export`)} /></View>
    </View>
  </ScreenLayout>;
}

const styles = StyleSheet.create({
  loading: { alignItems: 'center', justifyContent: 'center' },
  content: { gap: spacing.lg, minWidth: 0 },
  pageHero: { minHeight: 106, minWidth: 0, position: 'relative' },
  heroCopy: { gap: spacing.xs, minWidth: 0, paddingRight: spacing.xxl },
  botanical: { height: 260, right: -spacing.xl, top: -spacing.xxl, width: 188 },
  title: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.display, fontSize: fontSize.display, fontStyle: 'italic', fontWeight: '400', lineHeight: 45, minWidth: 0 },
  subtitle: { ...typography.bodyLarge, color: colors.text.secondary, flexShrink: 1, minWidth: 0 },
  heroCard: { gap: spacing.md, padding: spacing.lg },
  heroCardTop: { alignItems: 'center', flexDirection: 'row', gap: spacing.md, minWidth: 0 },
  heroFacts: { flex: 1, gap: spacing.sm, minWidth: 0 },
  statusLine: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, minWidth: 0 },
  statusLabel: { ...typography.bodyLarge, color: colors.text.primary, flexShrink: 1, minWidth: 0 },
  factRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, minWidth: 0 },
  fact: { ...typography.body, color: colors.text.primary, flex: 1, flexShrink: 1, minWidth: 0 },
  contentCard: { gap: spacing.md, padding: spacing.lg },
  cardHeading: { alignItems: 'center', flexDirection: 'row', gap: spacing.md, minWidth: 0 },
  cardHeadingText: { ...typography.bodyLarge, color: '#155B58', flex: 1, flexShrink: 1, fontWeight: '500', minWidth: 0 },
  body: { ...typography.bodyLarge, color: colors.text.primary, flexShrink: 1, minWidth: 0 },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, minWidth: 0 },
  recommendRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.md, minWidth: 0 },
  recommendText: { flex: 1 },
  technicalNote: { color: colors.text.secondary, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.caption, lineHeight: 19, minWidth: 0 },
  deviceRow: { alignItems: 'center', borderTopColor: colors.border.subtle, borderTopWidth: 1, flexDirection: 'row', gap: spacing.sm, marginTop: spacing.xs, minWidth: 0, paddingTop: spacing.md },
  deviceText: { color: colors.text.secondary, flex: 1, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 21, minWidth: 0 },
  nextCard: { alignItems: 'flex-start', backgroundColor: '#F3FAF5', borderColor: '#D6E8DA', borderRadius: radius.xl, borderWidth: 1, flexDirection: 'row', gap: spacing.md, minWidth: 0, padding: spacing.lg },
  nextCopy: { flex: 1, gap: spacing.sm, minWidth: 0 },
  nextTitle: { ...typography.label, color: colors.text.primary, flexShrink: 1, minWidth: 0 },
  nextText: { ...typography.body, color: colors.text.secondary, flexShrink: 1, minWidth: 0 },
  consentCard: { alignItems: 'flex-start', backgroundColor: colors.status.amberBg, borderColor: '#EBCF9F', borderRadius: radius.xl, borderWidth: 1, flexDirection: 'row', gap: spacing.md, minWidth: 0, padding: spacing.lg },
  consentCopy: { flex: 1, gap: spacing.sm, minWidth: 0 },
  consentTitle: { ...typography.label, color: colors.text.primary, flexShrink: 1, minWidth: 0 },
  consentText: { ...typography.body, color: colors.text.secondary, flexShrink: 1, minWidth: 0 },
  inlineAction: { alignItems: 'center', alignSelf: 'flex-start', flexDirection: 'row', gap: spacing.xs, minHeight: 44, minWidth: 0 },
  inlineActionText: { ...typography.body, color: colors.accent, flexShrink: 1, fontWeight: '700', minWidth: 0 },
  actionRow: { alignItems: 'stretch', flexDirection: 'row', flexWrap: 'wrap', gap: spacing.md, minWidth: 0 },
  primaryWrap: { flexBasis: 190, flexGrow: 1, minWidth: 0 },
  callWrap: { flexBasis: 94, flexGrow: 0, flexShrink: 1, minWidth: 88 },
  section: { ...typography.section, color: colors.text.primary, fontFamily: fontFamily.display, fontWeight: '400', marginTop: spacing.sm, minWidth: 0 },
  exploreGrid: { gap: spacing.sm, minWidth: 0 },
  exploreGridWide: { flexDirection: 'row', flexWrap: 'wrap' },
  exploreFull: { minWidth: 0, width: '100%' },
  exploreRow: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.lg, borderWidth: 1, flexDirection: 'row', gap: spacing.sm, minHeight: 64, minWidth: 0, paddingHorizontal: spacing.md, paddingVertical: spacing.sm },
  exploreRowWide: { flexBasis: '46%', flexGrow: 1, minWidth: 220 },
  exploreIcon: { alignItems: 'center', backgroundColor: '#EEF5EF', borderRadius: radius.pill, flexShrink: 0, height: 38, justifyContent: 'center', width: 38 },
  exploreLabel: { ...typography.body, color: colors.text.primary, flex: 1, flexShrink: 1, minWidth: 0 },
});
