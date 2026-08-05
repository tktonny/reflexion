import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { AppHeader, PrimaryButton, ProvenanceSection, ScreenLayout } from '../../../../src/components/AppUI';
import { getSessionProcessingStatusV1, getSessionV1, type V1SessionDetail, type V1SessionProcessingStatus } from '../../../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, spacing } from '../../../../src/theme';

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
  return <ScreenLayout>
    <AppHeader title={session.patientName} onBack={() => router.back()} />
    <Text accessibilityRole="header" style={styles.title}>Session detail</Text>
    <Text style={styles.subtitle}>{date} · {formatDuration(session.duration)}</Text>
    <View style={styles.statusCard}><Feather color={colors.accent} name={processing?.state === 'failed' ? 'alert-circle' : 'check-circle'} size={22} /><View style={styles.statusCopy}><Text style={styles.statusTitle}>{processingLabel}</Text><Text style={styles.statusDetail}>This status describes the recording and processing pipeline, not a conclusion about {session.patientName}.</Text></View></View>
    <View style={styles.card}><ProvenanceSection label="Observed">{session.exchanges ? `${session.exchanges} response${session.exchanges === 1 ? '' : 's'} recorded.` : 'No responses were recorded.'}</ProvenanceSection><ProvenanceSection label={`What ${session.patientName} shared`}>{session.logs.length ? `${session.logs.filter((log) => log.role === 'user').length} spoken response${session.logs.filter((log) => log.role === 'user').length === 1 ? '' : 's'} are available below.` : 'No transcript is available for this session.'}</ProvenanceSection><ProvenanceSection label="Limitations">The transcript is not a medical record. Reflexion does not infer mood, wellbeing, diagnoses or whether an activity happened.</ProvenanceSection></View>
    <Text style={styles.section}>Conversation</Text>
    {session.logs.length ? session.logs.map((log, index) => <View key={`${session.id}-${index}`} style={[styles.turn, log.role === 'user' ? styles.turnPatient : styles.turnAssistant]}><Text style={styles.turnRole}>{log.role === 'user' ? session.patientName : 'Reflexion'}</Text><Text style={styles.turnText}>{log.sentence}</Text></View>) : <View style={styles.empty}><Text style={styles.emptyTitle}>No transcript available</Text><Text style={styles.emptyCopy}>The Mirror did not upload spoken text for this session.</Text></View>}
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
  title: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', marginTop: spacing.lg }, subtitle: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 },
  statusCard: { alignItems: 'flex-start', backgroundColor: '#EEF7F0', borderColor: '#CFE4D0', borderRadius: radius.xl, borderWidth: 1, flexDirection: 'row', gap: spacing.md, padding: spacing.lg }, statusCopy: { flex: 1, flexShrink: 1 }, statusTitle: { color: colors.status.green, flexShrink: 1, fontSize: fontSize.bodyLarge, fontWeight: '700' }, statusDetail: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.caption, lineHeight: 18, marginTop: 4 },
  card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, paddingHorizontal: spacing.xl }, section: { color: colors.text.primary, fontSize: fontSize.heading, fontWeight: '700', marginTop: spacing.md },
  turn: { borderRadius: radius.xl, maxWidth: '92%', padding: spacing.lg }, turnPatient: { alignSelf: 'flex-end', backgroundColor: '#E7F3F0' }, turnAssistant: { alignSelf: 'flex-start', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderWidth: 1 }, turnRole: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.caption, fontWeight: '700', marginBottom: 4 }, turnText: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, lineHeight: 23 },
  empty: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, padding: spacing.xl }, emptyTitle: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.heading, fontWeight: '700', textAlign: 'center' }, emptyCopy: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22, textAlign: 'center' }, error: { color: colors.error.text, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 },
});
