import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useCallback, useMemo, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { listFamilyMessagesV1, listReminderOccurrencesV1, listSessionsV1, loadCaregiverHome, type V1FamilyMessage, type V1ReminderOccurrence } from '../../src/lib/v1Caregiver';
import { ScreenLayout, SecondaryButton } from '../../src/components/AppUI';
import { MotionFadeIn, MotionPressable, MotionSheet } from '../../src/components/Motion';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';
import { useTabBarClearance } from '../../src/lib/useTabBarClearance';

type EventKind = 'conversation' | 'routine' | 'chat' | 'technical';
type Event = { id: string; patientId: string; title: string; detail: string; time: string; at: number; kind: EventKind; icon: 'send' | 'monitor' | 'message-circle' | 'clock' };

export default function ActivityScreen() {
  const router = useRouter();
  const clearance = useTabBarClearance();
  const [events, setEvents] = useState<Event[]>([]);
  const [filter, setFilter] = useState<EventKind | 'all'>('all');
  const [filterOpen, setFilterOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    setLoading(true); setError('');
    try {
      const nextHome = await loadCaregiverHome();
      const { from, to } = activityWindow();
      const feeds = await Promise.all(nextHome.patients.map(async (person) => {
        const [messages, sessions, occurrences] = await Promise.all([
          listFamilyMessagesV1(person.patientId),
          listSessionsV1(person.patientId, { limit: 50 }),
          listReminderOccurrencesV1(person.patientId, from, to).catch(() => [] as V1ReminderOccurrence[]),
        ]);
        return { person, messages, sessions: sessions.sessions, occurrences };
      }));
      const next = feeds.flatMap(({ person, messages, sessions, occurrences }) => {
        const messageEvents = messages.map((message) => messageEvent(person.displayName, person.patientId, message));
        const sessionEvents = sessions.filter((session) => session.createdAt).map((session) => ({ id: `session-${session.id}`, patientId: person.patientId, title: 'Conversation recorded', detail: `${person.displayName} · ${formatDuration(session.duration)} · ${session.logs.length ? 'Transcript available' : 'Transcript not available'}.`, time: formatTime(session.createdAt as string), at: Date.parse(session.createdAt as string), kind: 'conversation' as const, icon: 'message-circle' as const }));
        const routineEvents = occurrences.map((occurrence) => routineEvent(person.displayName, person.patientId, occurrence));
        const technicalEvent = person.deviceId ? [{ id: `device-${person.patientId}-${person.lastHeartbeatAt || 'unknown'}`, patientId: person.patientId, title: person.deviceTechnicalState === 'ok' ? 'Device online' : 'Device may be offline', detail: `${person.mirrorName || 'Reflexion Mirror'} · technical status only.`, time: person.lastHeartbeatAt ? formatTime(person.lastHeartbeatAt) : 'No heartbeat yet', at: person.lastHeartbeatAt ? Date.parse(person.lastHeartbeatAt) : 0, kind: 'technical' as const, icon: 'monitor' as const }] : [];
        return [...messageEvents, ...sessionEvents, ...routineEvents, ...technicalEvent];
      });
      setEvents(next.filter((event) => Number.isFinite(event.at)).sort((a, b) => b.at - a.at));
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'We could not load activity. Check your connection and try again.'); }
    finally { setLoading(false); }
  }, []);
  useFocusEffect(useCallback(() => { void refresh(); }, [refresh]));

  const visible = useMemo(() => filter === 'all' ? events : events.filter((event) => event.kind === filter), [events, filter]);
  const filterLabel = filter === 'all' ? 'All activity' : ({ conversation: 'Conversations', routine: 'Routines', chat: 'Chat', technical: 'Technical' } as Record<EventKind, string>)[filter];

  return (
    <ScreenLayout bottomInset={clearance} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <Text accessibilityRole="header" style={styles.title}>Activity</Text>
        <View style={styles.headerActions}>
          <MotionPressable accessibilityLabel="Filter activity" accessibilityRole="button" haptic="selection" onPress={() => setFilterOpen(true)} style={styles.filter}>
            <Feather color={colors.accent} name="sliders" size={19} />
            <Text style={styles.filterText}>{filterLabel}</Text>
          </MotionPressable>
          <MotionPressable accessibilityLabel="Refresh activity" accessibilityRole="button" onPress={() => void refresh()} style={styles.filter}>
            <Feather color={colors.accent} name="refresh-cw" size={19} />
            <Text style={styles.filterText}>Refresh</Text>
          </MotionPressable>
        </View>
      </View>
      <Text style={styles.subtitle}>A factual, cross-household timeline of conversations, routines, family messages and technical events.</Text>
      {loading ? <ActivityIndicator color={colors.accent} /> : null}
      {error ? <View style={styles.empty}><Text style={styles.emptyTitle}>Activity is unavailable</Text><Text style={styles.emptyText}>{error}</Text><SecondaryButton label="Try again" onPress={() => void refresh()} /></View> : null}
      {!loading && !error ? visible.map((event, index) => (
        <MotionFadeIn key={event.id} delay={Math.min(index, 4) * 18} playKey={event.id}>
          <MotionPressable accessibilityLabel={`${event.title}. ${event.detail}`} accessibilityRole="button" feedback="card" onPress={() => router.push({ pathname: '/activity/[eventId]', params: { eventId: encodeURIComponent(event.id), title: event.title, detail: event.detail, time: event.time, patientId: event.patientId } })} style={styles.event}>
            <View style={styles.dot}><Feather color={colors.accent} name={event.icon} size={15} /></View>
            <View style={styles.eventCard}><Text style={styles.time}>{event.time}</Text><Text style={styles.eventTitle}>{event.title}</Text><Text style={styles.eventDetail}>{event.detail}</Text></View>
          </MotionPressable>
        </MotionFadeIn>
      )) : null}
      {!loading && !error && !visible.length ? <View style={styles.empty}><Text style={styles.emptyTitle}>{filter === 'all' ? 'No activity to show yet' : 'No matching activity'}</Text><Text style={styles.emptyText}>{filter === 'all' ? 'Pair a device, record a conversation, create a routine or send a family message to create the first factual entry.' : 'Choose another filter to view the available timeline.'}</Text></View> : null}
      <MotionSheet title="Filter activity" visible={filterOpen} onClose={() => setFilterOpen(false)}>
        <Text style={styles.sheetNote}>The timeline currently covers the last 31 days.</Text>
        {(['all', 'conversation', 'routine', 'chat', 'technical'] as const).map((option) => <MotionPressable key={option} accessibilityRole="button" accessibilityState={{ selected: filter === option }} feedback="card" haptic="selection" onPress={() => { setFilter(option); setFilterOpen(false); }} style={[styles.filterOption, filter === option && styles.filterOptionSelected]}><Text style={styles.filterOptionText}>{option === 'all' ? 'All activity' : ({ conversation: 'Conversations', routine: 'Routines', chat: 'Chat', technical: 'Technical' } as Record<EventKind, string>)[option]}</Text><Feather color={filter === option ? colors.accent : colors.textDecorative} name={filter === option ? 'check-circle' : 'circle'} size={19} /></MotionPressable>)}
        <SecondaryButton label="Close" onPress={() => setFilterOpen(false)} />
      </MotionSheet>
    </ScreenLayout>
  );
}

function activityWindow() {
  const end = new Date();
  const start = new Date(end.getTime() - 30 * 86_400_000);
  return { from: start.toISOString(), to: new Date(end.getTime() + 86_400_000).toISOString() };
}

function formatTime(value: string) { const date = new Date(value); return Number.isNaN(date.getTime()) ? 'Time unavailable' : new Intl.DateTimeFormat('en-SG', { dateStyle: 'medium', timeStyle: 'short' }).format(date); }
function formatDuration(seconds: number) { const total = Math.max(0, Math.round(seconds)); const minutes = Math.floor(total / 60); return minutes ? `${minutes} min` : `${total} sec`; }
function messageEvent(name: string, patientId: string, message: V1FamilyMessage): Event { const at = Date.parse(message.createdAt); const status = ({ scheduled: 'scheduled', queued: 'queued', delivered: 'delivered to the device', opened: 'delivered to the device and viewed', expired: 'expired', failed: 'not delivered' } as Record<V1FamilyMessage['state'], string>)[message.state]; return { id: `message-${message.messageId}`, patientId, title: 'Family message', detail: `A message for ${name} is ${status}.`, time: formatTime(message.createdAt), at, kind: 'chat', icon: 'send' }; }
function routineEvent(name: string, patientId: string, occurrence: V1ReminderOccurrence): Event { const at = Date.parse(occurrence.scheduledAt); return { id: `routine-${occurrence.occurrenceId}`, patientId, title: 'Reminder response recorded', detail: `${name} · ${occurrence.displayText} · ${routineStatus(occurrence.status)}.`, time: formatTime(occurrence.scheduledAt), at, kind: 'routine', icon: 'clock' }; }
function routineStatus(status: string) { return ({ scheduled: 'Scheduled', presented: 'Presented', 'reported-complete': 'Reported complete', deferred: 'Deferred', declined: 'Declined', 'no-response': 'No response', 'device-unavailable': 'Device unavailable' } as Record<string, string>)[status] || status; }

const styles = StyleSheet.create({ content: { gap: spacing.md, paddingTop: spacing.xl }, header: { alignItems: 'stretch', gap: spacing.md, minWidth: 0 }, headerActions: { alignItems: 'flex-start', flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, minWidth: 0 }, filter: { alignItems: 'center', borderColor: colors.border.default, borderRadius: radius.pill, borderWidth: 1, flexShrink: 0, flexDirection: 'row', gap: 6, minHeight: 44, paddingHorizontal: spacing.md }, filterText: { color: colors.accent, flexShrink: 0, fontSize: fontSize.body, fontWeight: '700', lineHeight: 20 }, subtitle: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22, marginBottom: spacing.xl, marginTop: spacing.md, minWidth: 0 }, event: { flexDirection: 'row', gap: spacing.md, minWidth: 0 }, dot: { alignItems: 'center', backgroundColor: '#EEF3E9', borderRadius: 999, flexShrink: 0, height: 34, justifyContent: 'center', marginTop: spacing.sm, width: 34 }, eventCard: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, flex: 1, minWidth: 0, padding: spacing.lg }, time: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.caption }, eventTitle: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, fontWeight: '700', marginTop: 6 }, eventDetail: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 21, marginTop: 3 }, empty: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, gap: spacing.sm, minWidth: 0, padding: spacing.xl }, emptyTitle: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.display, fontSize: fontSize.heading }, emptyText: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 }, modalBackdrop: { backgroundColor: 'rgba(22,50,74,0.24)', flex: 1, justifyContent: 'flex-end' }, sheet: { backgroundColor: colors.surface.card, borderTopLeftRadius: radius.xl, borderTopRightRadius: radius.xl, gap: spacing.md, minWidth: 0, padding: spacing.xl }, sheetTitle: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.heading, fontWeight: '500' }, sheetNote: { color: colors.text.secondary, fontSize: fontSize.caption }, filterOption: { alignItems: 'center', borderColor: colors.border.default, borderRadius: radius.md, borderWidth: 1, flexDirection: 'row', justifyContent: 'space-between', minHeight: 52, minWidth: 0, paddingHorizontal: spacing.lg }, filterOptionSelected: { backgroundColor: '#E7F3F0', borderColor: colors.accent }, filterOptionText: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, minWidth: 0 } });
