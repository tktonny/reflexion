import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, TextInput, View } from 'react-native';

import { listFamilyMessagesV1, loadCaregiverHome, type CaregiverHome, type V1FamilyMessage } from '../../src/lib/v1Caregiver';
import { ScreenLayout } from '../../src/components/AppUI';
import { ChatAvatar, ChatBotanical, ChatBrandLockup, ChatStatusBadge } from '../../src/components/ChatVisuals';
import { MotionFadeIn, MotionPressable } from '../../src/components/Motion';
import { colors, fontFamily, fontSize, radius, scaleSize, spacing } from '../../src/theme';
import { useTabBarClearance } from '../../src/lib/useTabBarClearance';

export default function ChatScreen() {
  const router = useRouter(); const clearance = useTabBarClearance();
  const [home, setHome] = useState<CaregiverHome | null>(null);
  const [messages, setMessages] = useState<Record<string, V1FamilyMessage | undefined>>({});
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true); const [error, setError] = useState('');
  const refresh = useCallback(async () => {
    setLoading(true);
    setError(''); try {
      const next = await loadCaregiverHome(); setHome(next);
      const latest = await Promise.all(next.patients.map(async (person) => [person.patientId, (await listFamilyMessagesV1(person.patientId))[0]] as const));
      setMessages(Object.fromEntries(latest));
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'We could not load messages. Check your connection and try again.'); } finally { setLoading(false); }
  }, []);
  useFocusEffect(useCallback(() => { void refresh(); }, [refresh]));
  const visiblePatients = home?.patients.filter((person) => person.displayName.toLowerCase().includes(search.trim().toLowerCase())) || [];
  return <ScreenLayout bottomInset={clearance} contentContainerStyle={styles.content}>
    <View style={styles.hero}>
      <ChatBotanical />
      <MotionPressable accessibilityLabel="Start a new conversation" accessibilityRole="button" onPress={() => { const first = home?.patients[0]; if (first) router.push(`/chat/${first.patientId}/compose`); }} style={styles.newMessageButton}>
        <Feather color={colors.accent} name="plus" size={30} />
      </MotionPressable>
      <ChatBrandLockup />
      <Text accessibilityRole="header" style={styles.title}>Chat</Text>
      <Text style={styles.subtitle}>Stay connected with your loved ones{`\n`}and family.</Text>
    </View>
    <View style={styles.searchWrap}>
      <Feather color={colors.text.secondary} name="search" size={25} />
      <TextInput accessibilityLabel="Search conversations" placeholder="Search conversations" placeholderTextColor={colors.text.secondary} onChangeText={setSearch} style={styles.searchInput} value={search} />
    </View>
    {loading ? <ActivityIndicator color={colors.accent} /> : null}
    {error ? <View style={styles.emptyCard}><Text style={styles.empty}>{error}</Text><MotionPressable accessibilityRole="button" onPress={() => void refresh()} style={styles.retry}><Text style={styles.retryText}>Try again</Text></MotionPressable></View> : null}
    {!error && visiblePatients.map((person, index) => { const latest = messages[person.patientId]; return <MotionFadeIn key={person.patientId} delay={Math.min(index, 4) * 18} playKey={person.patientId}><MotionPressable accessibilityLabel={`Message ${person.displayName}`} accessibilityRole="button" feedback="card" onPress={() => router.push(`/chat/${person.patientId}`)} style={styles.thread}>
      <ChatAvatar name={person.displayName} photoUrl={person.profile.photoUrl} size={58} />
      <View style={styles.copy}>
        <Text style={styles.name} numberOfLines={1}>{person.displayName}</Text>
        <Text style={styles.message} numberOfLines={1}>{latest?.body || (person.deviceId ? 'No family messages yet.' : 'Pair a device before sending a message.')}</Text>
      </View>
      <View style={styles.meta}>
        <Text style={styles.time}>{latest ? formatTime(latest.createdAt) : ''}</Text>
        {latest ? <ChatStatusBadge label={statusCopy(latest.state)} state={latest.state} /> : <ChatStatusBadge label={person.deviceId ? 'Ready to send' : 'No device'} state="queued" />}
      </View>
      <Feather color={colors.text.primary} name="chevron-right" size={24} />
    </MotionPressable></MotionFadeIn>; })}
    {!loading && !error && !visiblePatients.length ? <Text style={styles.empty}>{search.trim() ? 'No conversations match your search.' : 'Add a loved one before starting a conversation.'}</Text> : null}
  </ScreenLayout>;
}
function formatTime(value: string) { return new Intl.DateTimeFormat('en-SG', { hour: 'numeric', minute: '2-digit' }).format(new Date(value)); }
function statusCopy(state: V1FamilyMessage['state']) { return ({ scheduled: 'Scheduled', queued: 'Queued', delivered: 'Delivered', opened: 'Opened', expired: 'Expired', failed: 'Failed' } as Record<V1FamilyMessage['state'], string>)[state]; }
const styles = StyleSheet.create({
  content: { gap: spacing.md, minWidth: 0, paddingTop: spacing.xs },
  hero: { minHeight: scaleSize(205), minWidth: 0, position: 'relative', zIndex: 1 },
  newMessageButton: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.pill, borderWidth: 1, height: scaleSize(58), justifyContent: 'center', position: 'absolute', right: scaleSize(42), top: scaleSize(40), width: scaleSize(58), zIndex: 2 },
  title: { color: colors.text.primary, fontFamily: fontFamily.ui, fontSize: fontSize.display, fontWeight: '700', lineHeight: 42, marginTop: spacing.sm, minWidth: 0 },
  subtitle: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.bodyLarge, lineHeight: 24, marginTop: spacing.sm, minWidth: 0 },
  searchWrap: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, flexDirection: 'row', gap: spacing.md, minHeight: 58, minWidth: 0, paddingHorizontal: spacing.lg },
  searchInput: { color: colors.text.primary, flex: 1, fontSize: fontSize.bodyLarge, minWidth: 0, paddingVertical: spacing.sm },
  thread: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, flexDirection: 'row', gap: spacing.md, minHeight: 94, minWidth: 0, paddingHorizontal: spacing.md, paddingVertical: spacing.md },
  copy: { flex: 1, flexShrink: 1, gap: spacing.xs, minWidth: 0 },
  name: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.subheading, fontWeight: '700', lineHeight: 23, minWidth: 0 },
  message: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 21, minWidth: 0 },
  meta: { alignItems: 'flex-end', flexShrink: 1, gap: spacing.xs, maxWidth: scaleSize(96), minWidth: 0 },
  time: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.caption, lineHeight: 18, textAlign: 'right' },
  empty: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 21, textAlign: 'center' },
  emptyCard: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, minWidth: 0, padding: spacing.xl },
  retry: { alignItems: 'center', borderColor: colors.accent, borderRadius: radius.pill, borderWidth: 1, minHeight: 44, justifyContent: 'center', paddingHorizontal: spacing.lg },
  retryText: { color: colors.accent, fontSize: fontSize.body, fontWeight: '700' },
});
