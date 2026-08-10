import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { AppHeader, PrimaryButton, ScreenLayout, SurfaceCard } from '../../../../src/components/AppUI';
import { LovedAvatar, LovedBotanical, LovedChip, LovedIconCircle } from '../../../../src/components/LovedOneVisuals';
import { getSessionProcessingStatusV1, getSessionV1, type V1SessionDetail, type V1SessionProcessingStatus } from '../../../../src/lib/v1Caregiver';
import { cardShadow, colors, fontFamily, fontSize, radius, spacing, typography } from '../../../../src/theme';

export default function SessionDetailScreen() {
  const router = useRouter();
  const { id, sessionId } = useLocalSearchParams<{ id: string; sessionId: string }>();
  const [session, setSession] = useState<V1SessionDetail | null>(null);
  const [processing, setProcessing] = useState<V1SessionProcessingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    if (!id || !sessionId) return;
    setLoading(true);
    setError('');
    try {
      const [nextSession, nextProcessing] = await Promise.all([
        getSessionV1(id, sessionId),
        getSessionProcessingStatusV1(sessionId).catch(() => null),
      ]);
      setSession(nextSession);
      setProcessing(nextProcessing);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'This session could not be loaded.');
    } finally {
      setLoading(false);
    }
  }, [id, sessionId]);

  useFocusEffect(useCallback(() => {
    void refresh();
    const timer = setInterval(() => { void refresh(); }, 15_000);
    return () => clearInterval(timer);
  }, [refresh]));

  if (loading && !session) return <ScreenLayout scroll={false} contentContainerStyle={styles.loading}><ActivityIndicator color={colors.accent} /></ScreenLayout>;
  if (error && !session) return <ScreenLayout><AppHeader title="Session detail" onBack={() => router.back()} /><Text style={styles.title}>Session unavailable</Text><Text style={styles.error}>{error}</Text><PrimaryButton label="Try again" onPress={() => void refresh()} /></ScreenLayout>;
  if (!session) return null;

  const at = session.createdAt ? new Date(session.createdAt) : null;
  const date = at && !Number.isNaN(at.getTime()) ? new Intl.DateTimeFormat('en-SG', { dateStyle: 'full', timeStyle: 'short' }).format(at) : 'Time unavailable';
  const processingLabel = processing ? processingStateLabel(processing) : sessionStateLabel(session.state);
  const sharedLines = session.logs.filter((log) => log.role === 'user' || log.role === 'loved-one');
  return <ScreenLayout>
    <AppHeader onBack={() => router.back()} />
    <View style={styles.pageHero}><Text accessibilityRole="header" style={styles.headerTitle}>Session detail</Text><LovedBotanical style={styles.botanical} /></View>
    <View style={styles.intro}><LovedAvatar name={session.patientName} size={112} /><View style={styles.introCopy}><Text style={styles.personName}>{session.patientName}</Text><View style={styles.metaRow}><LovedIconCircle icon="calendar" size={34} /><Text style={styles.meta}>{date}</Text></View><View style={styles.metaRow}><LovedIconCircle icon="clock" size={34} /><Text style={styles.meta}>{formatDuration(session.duration)}</Text></View><View style={styles.statusLine}><LovedIconCircle icon={processing?.state === 'failed' ? 'alert-circle' : 'check-circle'} size={32} /><Text style={styles.statusLabel}>{processingLabel}</Text></View></View></View>
    <SurfaceCard style={styles.card}><Text style={styles.cardTitle}>Summary</Text><Text style={styles.body}>{session.logs.length ? `${session.patientName} shared updates during this recorded conversation.` : 'No transcript is available for this session.'}</Text><Text style={styles.cardTitle}>Topics</Text><View style={styles.chips}><LovedChip>Daily routine</LovedChip><LovedChip>Social connection</LovedChip><LovedChip>Meals</LovedChip><LovedChip>Planning ahead</LovedChip></View></SurfaceCard>
    <SurfaceCard style={styles.card}><Text style={styles.cardTitle}>What {session.patientName} shared</Text>{sharedLines.length ? sharedLines.slice(0, 4).map((log, index) => <Text key={`${session.id}-shared-${index}`} style={styles.bullet}>• {log.sentence}</Text>) : <Text style={styles.body}>The Mirror did not upload spoken text for this session.</Text>}</SurfaceCard>
    <SurfaceCard style={styles.card}><View style={styles.detailRow}><LovedIconCircle icon="clipboard" size={54} /><View style={styles.detailCopy}><Text style={styles.cardTitle}>Routines discussed</Text><Text style={styles.body}>Session exchanges and any recorded routine responses are shown in the conversation.</Text></View></View></SurfaceCard>
    <SurfaceCard style={styles.card}><View style={styles.detailRow}><LovedIconCircle icon="shield" size={54} /><View style={styles.detailCopy}><Text style={styles.cardTitle}>Information provenance</Text><Text style={styles.body}>This update is based on this recorded conversation.</Text><Text style={styles.note}>Reflexion does not access medical records or other sources.</Text></View></View></SurfaceCard>
    <PrimaryButton icon="message-circle" label="View full conversation" onPress={() => router.push(`/loved-one/${id}/sessions/${session.id}/conversation`)} />
    <SurfaceCard style={styles.feedbackCard}><Text style={styles.cardTitle}>Was this update useful?</Text><Text style={styles.note}>Your feedback helps improve summaries.</Text><View style={styles.feedbackButtons}><View style={styles.feedbackButton}><Text style={styles.feedbackButtonText}>Yes</Text></View><View style={styles.feedbackButton}><Text style={styles.feedbackButtonText}>Not really</Text></View></View></SurfaceCard>
    <View style={styles.limitations}><Feather color={colors.text.secondary} name="lock" size={16} /><Text style={styles.note}>Limitations: This summary is not a diagnosis or medical advice.</Text></View>
    <PrimaryButton label="Back to sessions" onPress={() => router.replace(`/loved-one/${id}/sessions`)} />
  </ScreenLayout>;
}

function processingStateLabel(status: V1SessionProcessingStatus) {
  if (status.state === 'processing') return 'Processing conversation';
  if (status.state === 'queued') return 'Queued for processing';
  if (status.state === 'failed') return status.retryable ? 'Processing needs a retry' : 'Processing unavailable';
  if (status.state === 'completed') return 'Processing complete';
  return 'Session received';
}

function sessionStateLabel(state: string | null) {
  if (state === 'processing' || state === 'ingesting') return 'Processing conversation';
  if (state === 'processing_failed') return 'Processing unavailable';
  if (state === 'abandoned') return 'Session not completed';
  return 'Session received';
}

function formatDuration(seconds: number) { const total = Math.max(0, Math.round(seconds)); const minutes = Math.floor(total / 60); const remainder = total % 60; return minutes ? `${minutes} min${remainder ? ` ${remainder} sec` : ''}` : `${remainder} sec`; }

const styles = StyleSheet.create({
  loading: { alignItems: 'center', justifyContent: 'center' },
  pageHero: { minHeight: 62, minWidth: 0, position: 'relative' },
  headerTitle: { ...typography.label, color: colors.text.primary, textAlign: 'center' },
  title: { ...typography.display, color: colors.text.primary },
  botanical: { height: 242, right: -spacing.xl, top: -spacing.xxl, width: 170 },
  intro: { alignItems: 'center', flexDirection: 'row', gap: spacing.lg, minWidth: 0 },
  introCopy: { flex: 1, gap: spacing.sm, minWidth: 0 },
  personName: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.display, fontStyle: 'italic', fontWeight: '400', lineHeight: 43, minWidth: 0 },
  metaRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, minWidth: 0 },
  meta: { ...typography.body, color: colors.text.primary, flexShrink: 1, minWidth: 0 },
  statusLine: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, minWidth: 0 },
  statusLabel: { ...typography.body, color: colors.accent, flexShrink: 1, fontWeight: '600', minWidth: 0 },
  card: { ...cardShadow, gap: spacing.md, padding: spacing.lg },
  cardTitle: { ...typography.section, color: colors.text.primary, flexShrink: 1, minWidth: 0 },
  body: { ...typography.bodyLarge, color: colors.text.primary, flexShrink: 1, minWidth: 0 },
  note: { ...typography.body, color: colors.text.secondary, flexShrink: 1, minWidth: 0 },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, minWidth: 0 },
  bullet: { ...typography.bodyLarge, color: colors.text.primary, flexShrink: 1, minWidth: 0 },
  detailRow: { alignItems: 'flex-start', flexDirection: 'row', gap: spacing.lg, minWidth: 0 },
  detailCopy: { flex: 1, gap: spacing.sm, minWidth: 0 },
  feedbackCard: { gap: spacing.sm },
  feedbackButtons: { alignItems: 'center', flexDirection: 'row', gap: spacing.md, justifyContent: 'flex-end', minWidth: 0 },
  feedbackButton: { alignItems: 'center', borderColor: colors.accent, borderRadius: radius.md, borderWidth: 1, minHeight: 44, justifyContent: 'center', minWidth: 84, paddingHorizontal: spacing.md },
  feedbackButtonText: { ...typography.body, color: colors.accent, fontWeight: '600' },
  limitations: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, justifyContent: 'center', minWidth: 0 },
  turn: { borderRadius: radius.xl, maxWidth: '92%', minWidth: 0, padding: spacing.lg }, turnPatient: { alignSelf: 'flex-end', backgroundColor: '#E7F3F0' }, turnAssistant: { alignSelf: 'flex-start', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderWidth: 1 }, turnRole: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.caption, fontWeight: '700', marginBottom: 4, minWidth: 0 }, turnText: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, lineHeight: 23, minWidth: 0 },
  empty: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, minWidth: 0, padding: spacing.xl }, emptyTitle: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.heading, fontWeight: '700', textAlign: 'center' }, emptyCopy: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22, textAlign: 'center' }, error: { color: colors.error.text, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22, minWidth: 0 },
});
