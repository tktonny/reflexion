import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { AppHeader, PrimaryButton, ProvenanceSection, ScreenLayout } from '../../../../../src/components/AppUI';
import { getSessionV1, type V1SessionDetail } from '../../../../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, spacing } from '../../../../../src/theme';

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
  return <ScreenLayout contentContainerStyle={styles.content}>
    <AppHeader title="Full conversation" onBack={() => router.back()} />
    <Text accessibilityRole="header" style={styles.title}>Full conversation</Text>
    <Text style={styles.subtitle}>{session.patientName} · {date}</Text>
    <View style={styles.card}><ProvenanceSection label="What this record contains">The transcript below is the conversation text received from the paired Mirror.</ProvenanceSection><ProvenanceSection label="Limitations">This transcript is not a medical record. It may be incomplete when audio or text was not received.</ProvenanceSection></View>
    {session.logs.length ? session.logs.map((log, index) => <View key={`${session.id}-${index}`} style={[styles.turn, log.role === 'user' ? styles.turnPatient : styles.turnAssistant]}><Text style={styles.turnRole}>{log.role === 'user' ? session.patientName : 'Reflexion'}</Text><Text style={styles.turnText}>{log.sentence}</Text></View>) : <View style={styles.empty}><Text style={styles.emptyTitle}>No transcript available</Text><Text style={styles.emptyCopy}>The Mirror did not upload spoken text for this session.</Text></View>}
  </ScreenLayout>;
}

const styles = StyleSheet.create({
  loading: { alignItems: 'center', justifyContent: 'center' },
  content: { gap: spacing.lg },
  title: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, marginTop: spacing.lg },
  subtitle: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 },
  error: { color: colors.error.text, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 },
  card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, overflow: 'hidden', paddingHorizontal: spacing.lg },
  turn: { borderRadius: radius.xl, maxWidth: '92%', padding: spacing.lg },
  turnPatient: { alignSelf: 'flex-end', backgroundColor: '#E7F3F0' },
  turnAssistant: { alignSelf: 'flex-start', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderWidth: 1 },
  turnRole: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.caption, fontWeight: '700', marginBottom: 4 },
  turnText: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, lineHeight: 23 },
  empty: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, padding: spacing.xl },
  emptyTitle: { color: colors.text.primary, fontSize: fontSize.heading, fontWeight: '700', textAlign: 'center' },
  emptyCopy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22, textAlign: 'center' },
});
