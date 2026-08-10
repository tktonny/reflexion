import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, SurfaceCard } from '../../../../../src/components/AppUI';
import { LovedAvatar, LovedBotanical, LovedIconCircle } from '../../../../../src/components/LovedOneVisuals';
import { getSessionV1, type V1SessionDetail } from '../../../../../src/lib/v1Caregiver';
import { cardShadow, colors, fontFamily, fontSize, spacing, typography } from '../../../../../src/theme';

export default function FullConversationScreen() {
  const router = useRouter();
  const { id, sessionId } = useLocalSearchParams<{ id: string; sessionId: string }>();
  const [session, setSession] = useState<V1SessionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    if (!id || !sessionId) return;
    setLoading(true);
    try {
      setSession(await getSessionV1(id, sessionId));
      setError('');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The conversation could not be loaded.');
    } finally {
      setLoading(false);
    }
  }, [id, sessionId]);
  useFocusEffect(useCallback(() => { void refresh(); }, [refresh]));

  if (loading && !session) return <ScreenLayout scroll={false} contentContainerStyle={styles.loading}><ActivityIndicator color={colors.accent} /></ScreenLayout>;
  if (!session) return <ScreenLayout><AppHeader title="Full conversation" onBack={() => router.back()} /><Text style={styles.title}>Conversation unavailable</Text><Text style={styles.error}>{error || 'No transcript is available for this session.'}</Text><PrimaryButton label="Back to session" onPress={() => router.back()} /></ScreenLayout>;

  const at = session.createdAt ? new Date(session.createdAt) : null;
  const date = at && !Number.isNaN(at.getTime()) ? new Intl.DateTimeFormat('en-SG', { dateStyle: 'full', timeStyle: 'short' }).format(at) : 'Time unavailable';
  const timeLabel = at && !Number.isNaN(at.getTime()) ? new Intl.DateTimeFormat('en-SG', { hour: 'numeric', minute: '2-digit' }).format(at) : 'Time unavailable';
  return <ScreenLayout contentContainerStyle={styles.content}>
    <AppHeader onBack={() => router.back()} />
    <View style={styles.pageHero}><View style={styles.titleCopy}><Text accessibilityRole="header" style={styles.title}>Conversation with</Text><Text style={styles.personName}>{session.patientName}</Text></View><LovedBotanical style={styles.botanical} /></View>
    <SurfaceCard style={styles.metaCard}><View style={styles.metaRow}><LovedIconCircle icon="calendar" size={38} /><Text style={styles.metaText}>{date}</Text></View><View style={styles.metaRow}><LovedIconCircle icon="clock" size={38} /><Text style={styles.metaText}>{timeLabel} · {formatDuration(session.duration)}</Text></View></SurfaceCard>
    {session.logs.length ? <SurfaceCard style={styles.transcriptCard}>{session.logs.map((log, index) => { const isLovedOne = log.role === 'user' || log.role === 'loved-one'; return <View key={`${session.id}-${index}`} style={styles.messageRow}><View style={styles.messageAvatar}>{isLovedOne ? <LovedAvatar name={session.patientName} size={50} /> : <LovedIconCircle icon="feather" size={54} />}</View><View style={styles.messageCopy}><View style={styles.messageHeader}><Text style={[styles.sender, isLovedOne && styles.senderLoved]}>{isLovedOne ? session.patientName : 'Reflexion'}</Text><Text style={styles.messageTime}>{timeLabel}</Text></View><Text style={styles.messageText}>{log.sentence}</Text></View></View>; })}</SurfaceCard> : <SurfaceCard style={styles.empty}><Text style={styles.emptyTitle}>No transcript available</Text><Text style={styles.emptyCopy}>The Mirror did not upload spoken text for this session.</Text></SurfaceCard>}
    <View style={styles.footerNote}><Feather color={colors.text.secondary} name="info" size={16} /><Text style={styles.note}>This is the full transcript of your conversation.</Text></View>
  </ScreenLayout>;
}

function formatDuration(seconds: number) { const total = Math.max(0, Math.round(seconds)); const minutes = Math.floor(total / 60); const remainder = total % 60; return minutes ? `${minutes} min${remainder ? ` ${remainder} sec` : ''}` : `${remainder} sec`; }

const styles = StyleSheet.create({
  loading: { alignItems: 'center', justifyContent: 'center' },
  content: { gap: spacing.lg },
  pageHero: { minHeight: 102, minWidth: 0, position: 'relative' },
  titleCopy: { gap: spacing.xs, minWidth: 0, paddingRight: spacing.xxl },
  title: { ...typography.display, color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.display, fontWeight: '400', lineHeight: 44 },
  personName: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.display, fontSize: fontSize.display, fontStyle: 'italic', lineHeight: 43, minWidth: 0 },
  botanical: { height: 270, right: -spacing.xl, top: -spacing.xxl, width: 180 },
  metaCard: { alignItems: 'center', flexDirection: 'row', flexWrap: 'wrap', gap: spacing.lg, justifyContent: 'space-between', padding: spacing.lg },
  metaRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, minWidth: 0 },
  metaText: { ...typography.bodyLarge, color: colors.text.primary, flexShrink: 1, minWidth: 0 },
  transcriptCard: { ...cardShadow, gap: 0, overflow: 'hidden', paddingHorizontal: spacing.lg, paddingVertical: 0 },
  messageRow: { alignItems: 'flex-start', borderBottomColor: colors.border.subtle, borderBottomWidth: 1, flexDirection: 'row', gap: spacing.md, minWidth: 0, paddingVertical: spacing.lg },
  messageAvatar: { flexShrink: 0 },
  messageCopy: { flex: 1, gap: spacing.xs, minWidth: 0 },
  messageHeader: { alignItems: 'baseline', flexDirection: 'row', gap: spacing.sm, justifyContent: 'space-between', minWidth: 0 },
  sender: { ...typography.bodyLarge, color: colors.text.primary, flexShrink: 1, fontWeight: '700', minWidth: 0 },
  senderLoved: { fontFamily: fontFamily.display, fontStyle: 'italic', fontWeight: '400' },
  messageTime: { ...typography.caption, color: colors.text.secondary, flexShrink: 0 },
  messageText: { ...typography.bodyLarge, color: colors.text.primary, flexShrink: 1, minWidth: 0 },
  footerNote: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, justifyContent: 'center', minWidth: 0 },
  note: { ...typography.body, color: colors.text.secondary, flexShrink: 1, minWidth: 0 },
  error: { color: colors.error.text, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 },
  empty: { alignItems: 'center', gap: spacing.md, padding: spacing.xl },
  emptyTitle: { color: colors.text.primary, fontSize: fontSize.heading, fontWeight: '700', textAlign: 'center' },
  emptyCopy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22, textAlign: 'center' },
});
