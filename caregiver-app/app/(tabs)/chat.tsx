import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

import { listFamilyMessagesV1, loadCaregiverHome, type CaregiverHome, type V1FamilyMessage } from '../../src/lib/v1Caregiver';
import { ScreenLayout } from '../../src/components/AppUI';
import { colors, contentColumn, fontFamily, fontSize, radius, spacing } from '../../src/theme';
import { useTabBarClearance } from '../../src/lib/useTabBarClearance';

export default function ChatScreen() {
  const router = useRouter(); const clearance = useTabBarClearance();
  const [home, setHome] = useState<CaregiverHome | null>(null);
  const [messages, setMessages] = useState<Record<string, V1FamilyMessage | undefined>>({});
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
  return <ScreenLayout bottomInset={clearance} contentContainerStyle={styles.content}>
    <Text accessibilityRole="header" style={styles.title}>Chat</Text>
    <Text style={styles.subtitle}>Messages arrive as a notification on the paired Mirror. They are only shown when your loved one chooses to open them.</Text>
    {loading ? <ActivityIndicator color={colors.accent} /> : null}
    {error ? <View style={styles.emptyCard}><Text style={styles.empty}>{error}</Text><TouchableOpacity accessibilityRole="button" onPress={() => void refresh()} style={styles.retry}><Text style={styles.retryText}>Try again</Text></TouchableOpacity></View> : null}
    {!error && home?.patients.map((person) => { const latest = messages[person.patientId]; return <TouchableOpacity key={person.patientId} accessibilityLabel={`Message ${person.displayName}`} accessibilityRole="button" onPress={() => router.push(`/chat/${person.patientId}`)} style={styles.thread}>
      <View style={styles.avatar}><Text style={styles.avatarText}>{person.displayName.slice(0, 1)}</Text></View><View style={styles.copy}><View style={styles.row}><Text style={styles.name}>{person.displayName}</Text><Text style={styles.time}>{latest ? formatTime(latest.createdAt) : ''}</Text></View><Text style={styles.message}>{latest?.body || (person.deviceId ? 'No family messages yet.' : 'Pair a device before sending a message.')}</Text><Text style={styles.status}>{latest ? statusCopy(latest.state) : person.deviceId ? 'Ready to send' : 'No device paired'}</Text></View><Feather color={colors.textDecorative} name="chevron-right" size={20} />
    </TouchableOpacity>; })}
    {!loading && !error && !home?.patients.length ? <Text style={styles.empty}>Add a loved one before starting a conversation.</Text> : null}
  </ScreenLayout>;
}
function formatTime(value: string) { return new Intl.DateTimeFormat('en-SG', { hour: 'numeric', minute: '2-digit' }).format(new Date(value)); }
function statusCopy(state: V1FamilyMessage['state']) { return ({ scheduled: 'Scheduled', queued: 'Queued for the Mirror', delivered: 'Delivered to device', opened: 'Opened' })[state]; }
const styles = StyleSheet.create({ content: { gap: spacing.lg, paddingTop: spacing.xl }, title: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500' }, subtitle: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22, marginBottom: spacing.lg, marginTop: -spacing.sm }, thread: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, flexDirection: 'row', gap: spacing.md, minHeight: 94, padding: spacing.lg }, avatar: { alignItems: 'center', backgroundColor: '#EADFCF', borderRadius: 999, flexShrink: 0, height: 52, justifyContent: 'center', width: 52 }, avatarText: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: 21 }, copy: { flex: 1, flexShrink: 1 }, row: { alignItems: 'flex-start', flexDirection: 'row', gap: spacing.sm, justifyContent: 'space-between' }, name: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.display, fontSize: 22, fontWeight: '500' }, time: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.caption, textAlign: 'right' }, message: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 21, marginTop: 3 }, status: { color: colors.status.green, flexShrink: 1, fontSize: fontSize.caption, fontWeight: '700', marginTop: 4 }, empty: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 21, textAlign: 'center' }, emptyCard: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, padding: spacing.xl }, retry: { alignItems: 'center', borderColor: colors.accent, borderRadius: radius.pill, borderWidth: 1, minHeight: 44, justifyContent: 'center', paddingHorizontal: spacing.lg }, retryText: { color: colors.accent, fontSize: fontSize.body, fontWeight: '700' } });
