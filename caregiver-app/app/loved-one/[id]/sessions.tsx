import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { AppHeader, PrimaryButton, ScreenLayout } from '../../../src/components/AppUI';
import { MotionFadeIn, MotionPressable } from '../../../src/components/Motion';
import { listSessionsV1, type V1SessionDetail, type V1SessionFeed } from '../../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, spacing } from '../../../src/theme';

export default function LovedOneSessionsScreen() {
  const router = useRouter();
  const { id } = useLocalSearchParams<{ id: string }>();
  const [feed, setFeed] = useState<V1SessionFeed | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    if (!id) return;
    setLoading(true);
    setError('');
    try {
      setFeed(await listSessionsV1(id, { limit: 20 }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Sessions could not be loaded.');
    } finally {
      setLoading(false);
    }
  }, [id]);

  useFocusEffect(useCallback(() => { void refresh(); }, [refresh]));

  const loadMore = async () => {
    if (!id || !feed?.nextBefore || loadingMore) return;
    setLoadingMore(true);
    try {
      const next = await listSessionsV1(id, { limit: 20, before: feed.nextBefore });
      setFeed({ ...next, sessions: [...feed.sessions, ...next.sessions] });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Older sessions could not be loaded.');
    } finally {
      setLoadingMore(false);
    }
  };

  const name = feed?.patientName || 'Loved one';
  return (
    <ScreenLayout>
        <AppHeader title={name} onBack={() => router.back()} />
        <Text accessibilityRole="header" style={styles.title}>Sessions</Text>
        <Text style={styles.subtitle}>A chronological record of conversations received from the paired Mirror.</Text>
        {loading ? <ActivityIndicator color={colors.accent} /> : null}
        {error ? <View style={styles.errorCard}><Text style={styles.error}>{error}</Text><PrimaryButton label="Try again" onPress={() => void refresh()} /></View> : null}
        {!loading && !error && !feed?.sessions.length ? (
          <View style={styles.empty}><Feather color={colors.textDecorative} name="message-circle" size={32} /><Text style={styles.emptyTitle}>No sessions recorded yet</Text><Text style={styles.emptyCopy}>When the Mirror records a conversation, it will appear here with its transcript and processing status.</Text></View>
        ) : null}
        {feed?.sessions.map((session, index) => <MotionFadeIn key={session.id} delay={Math.min(index, 4) * 18} playKey={session.id}><SessionRow session={session} onPress={() => router.push(`/loved-one/${id}/sessions/${session.id}`)} /></MotionFadeIn>)}
        {feed?.nextBefore ? <PrimaryButton disabled={loadingMore} label={loadingMore ? 'Loading…' : 'Load older sessions'} onPress={() => void loadMore()} /> : null}
    </ScreenLayout>
  );
}

function SessionRow({ session, onPress }: { session: V1SessionDetail; onPress: () => void }) {
  const at = session.createdAt ? new Date(session.createdAt) : null;
  const date = at && !Number.isNaN(at.getTime())
    ? new Intl.DateTimeFormat('en-SG', { dateStyle: 'medium', timeStyle: 'short' }).format(at)
    : 'Time unavailable';
  const duration = session.duration > 0 ? formatDuration(session.duration) : 'Duration unavailable';
  return <MotionPressable accessibilityRole="button" accessibilityLabel={`Conversation on ${date}`} feedback="card" onPress={onPress} style={styles.row}>
    <View style={styles.rowIcon}><Feather color={colors.accent} name="message-circle" size={20} /></View>
    <View style={styles.rowCopy}><Text style={styles.rowTitle}>{session.type === 'daily_checkin' ? 'Daily check-in' : 'Conversation'}</Text><Text style={styles.rowMeta}>{date} · {duration}</Text><Text style={styles.rowState}>{sessionStateLabel(session.state, session.logs.length > 0)}</Text></View>
    <Feather color={colors.textDecorative} name="chevron-right" size={20} />
  </MotionPressable>;
}

function formatDuration(seconds: number) {
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  return minutes ? `${minutes} min${remainder ? ` ${remainder} sec` : ''}` : `${remainder} sec`;
}

function sessionStateLabel(state: string | null, hasTranscript: boolean) {
  if (state === 'processing' || state === 'ingesting') return 'Processing summary';
  if (state === 'processing_failed') return 'Processing unavailable';
  if (state === 'abandoned') return 'Not completed';
  if (hasTranscript) return 'Transcript available';
  return 'Transcript not available';
}

const styles = StyleSheet.create({
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', marginTop: spacing.lg },
  subtitle: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22, minWidth: 0 },
  row: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, flexDirection: 'row', gap: spacing.md, minHeight: 88, minWidth: 0, padding: spacing.lg },
  rowIcon: { alignItems: 'center', backgroundColor: '#EEF3E9', borderRadius: radius.pill, flexShrink: 0, height: 42, justifyContent: 'center', width: 42 },
  rowCopy: { flex: 1, flexShrink: 1, minWidth: 0 }, rowTitle: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, fontWeight: '700', minWidth: 0 }, rowMeta: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, marginTop: 3, minWidth: 0 }, rowState: { color: colors.accent, flexShrink: 1, fontSize: fontSize.caption, fontWeight: '700', marginTop: 4, minWidth: 0 },
  empty: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, minWidth: 0, padding: spacing.xl }, emptyTitle: { color: colors.text.primary, fontSize: fontSize.heading, fontWeight: '700', textAlign: 'center' }, emptyCopy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22, textAlign: 'center' },
  errorCard: { backgroundColor: colors.error.surface, borderColor: colors.error.border, borderRadius: radius.lg, borderWidth: 1, gap: spacing.md, minWidth: 0, padding: spacing.lg }, error: { color: colors.error.text, flexShrink: 1, fontSize: fontSize.body, lineHeight: 21, minWidth: 0 },
});
