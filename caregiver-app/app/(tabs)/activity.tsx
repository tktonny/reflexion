import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useCallback, useMemo, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { listFamilyMessagesV1, listReminderOccurrencesV1, listSessionsV1, loadCaregiverHome, type V1FamilyMessage, type V1ReminderOccurrence } from '../../src/lib/v1Caregiver';
import { ScreenLayout, SecondaryButton } from '../../src/components/AppUI';
import { MotionFadeIn, MotionPressable, MotionSheet } from '../../src/components/Motion';
import { cardShadow, colors, fontFamily, fontSize, radius, scaleSize, spacing } from '../../src/theme';
import { useTabBarClearance } from '../../src/lib/useTabBarClearance';

type EventKind = 'conversation' | 'routine' | 'chat' | 'technical';
type EventTone = 'teal' | 'amber' | 'muted';
type EventIcon = keyof typeof Feather.glyphMap;
type Event = {
  id: string;
  patientId: string;
  person: string;
  title: string;
  detail: string;
  time: string;
  clock: string;
  at: number;
  kind: EventKind;
  icon: EventIcon;
  tone: EventTone;
};
type ActivitySection = { key: string; label: string; events: Event[] };

const EVENT_TONES: Record<EventTone, { background: string; color: string; dot: string }> = {
  teal: { background: '#E7F3F0', color: '#146B66', dot: '#1A6D68' },
  amber: { background: '#FFF1DC', color: '#C67805', dot: '#D58A12' },
  muted: { background: '#EEF1F1', color: '#68777B', dot: '#7A888A' },
};

const FILTER_LABELS: Record<EventKind | 'all', string> = {
  all: 'All',
  conversation: 'Conversations',
  routine: 'Routines',
  chat: 'Chat',
  technical: 'Technical',
};

const CATEGORY_FILTERS: { kind: EventKind; icon: EventIcon; label: string }[] = [
  { kind: 'conversation', icon: 'message-circle', label: 'Conversations' },
  { kind: 'routine', icon: 'calendar', label: 'Routines' },
  { kind: 'chat', icon: 'message-square', label: 'Chat' },
  { kind: 'technical', icon: 'wifi', label: 'Technical' },
];

export default function ActivityScreen() {
  const router = useRouter();
  const clearance = useTabBarClearance();
  const [events, setEvents] = useState<Event[]>([]);
  const [filter, setFilter] = useState<EventKind | 'all'>('all');
  const [filterOpen, setFilterOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    setLoading(true);
    setError('');
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
        const sessionEvents = sessions
          .filter((session) => session.createdAt)
          .map((session) => sessionEvent(person.displayName, person.patientId, session.id, session.createdAt as string, session.duration));
        const routineEvents = occurrences.map((occurrence) => routineEvent(person.displayName, person.patientId, occurrence));
        const technicalEvents = person.deviceId
          ? [technicalEvent(person.displayName, person.patientId, person.deviceTechnicalState, person.lastHeartbeatAt)]
          : [];
        return [...messageEvents, ...sessionEvents, ...routineEvents, ...technicalEvents];
      });

      setEvents(next.filter((event) => Number.isFinite(event.at)).sort((a, b) => b.at - a.at));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'We could not load activity. Check your connection and try again.');
    } finally {
      setLoading(false);
    }
  }, []);

  useFocusEffect(useCallback(() => {
    void refresh();
  }, [refresh]));

  const visible = useMemo(
    () => filter === 'all' ? events : events.filter((event) => event.kind === filter),
    [events, filter],
  );
  const sections = useMemo(() => groupEvents(visible), [visible]);

  return (
    <ScreenLayout bottomInset={clearance} contentContainerStyle={styles.content}>
      <View style={styles.heroBlock}>
        <ActivityBrand />
        <View style={styles.heroCopy}>
          <Text accessibilityRole="header" style={styles.heroTitle}>Activity</Text>
          <Text style={styles.heroSubtitle}>A timeline of important events across your loved ones.</Text>
        </View>
      </View>

      <View style={styles.filterControls}>
        <View style={styles.primaryFilterRow}>
          <ActivityFilterButton
            active={filter === 'all'}
            icon="layers"
            label="All"
            onPress={() => {
              setFilter('all');
              setFilterOpen(true);
            }}
          />
          <ActivityFilterButton icon="user" label="Person" onPress={() => setFilterOpen(true)} />
          <ActivityFilterButton icon="calendar" label="Date range" onPress={() => setFilterOpen(true)} />
        </View>

        <View style={styles.categoryRow}>
          {CATEGORY_FILTERS.map((item) => {
            const visualActive = filter === item.kind || (filter === 'all' && item.kind === 'conversation');
            return (
              <MotionPressable
                key={item.kind}
                accessibilityLabel={item.label}
                accessibilityRole="button"
                accessibilityState={{ selected: filter === item.kind }}
                feedback="none"
                haptic="selection"
                onPress={() => setFilter(item.kind)}
                style={[styles.categoryChip, visualActive && styles.categoryChipActive]}
              >
              <Feather color={visualActive ? '#146B66' : colors.text.secondary} name={item.icon} size={16} />
                <Text style={[styles.categoryChipText, visualActive && styles.categoryChipTextActive]}>{item.label}</Text>
              </MotionPressable>
            );
          })}
          <View accessibilityElementsHidden importantForAccessibility="no-hide-descendants" pointerEvents="none" style={styles.categoryChip}>
            <Feather color={colors.text.secondary} name="user" size={16} />
            <Text style={styles.categoryChipText}>Caregiver actions</Text>
          </View>
        </View>
      </View>

      {loading ? <ActivityIndicator color={colors.accent} style={styles.loading} /> : null}
      {error ? (
        <View style={styles.empty}>
          <Text style={styles.emptyTitle}>Activity is unavailable</Text>
          <Text style={styles.emptyText}>{error}</Text>
          <SecondaryButton label="Try again" onPress={() => void refresh()} />
        </View>
      ) : null}
      {!loading && !error ? sections.map((section) => (
        <View key={section.key} style={styles.section}>
          <Text accessibilityRole="header" style={styles.sectionTitle}>{section.label}</Text>
          <View style={styles.timeline}>
            {section.events.map((event, index) => (
              <MotionFadeIn key={event.id} delay={Math.min(index, 4) * 18} playKey={event.id} style={styles.fade}>
                <ActivityEventRow
                  event={event}
                  last={index === section.events.length - 1}
                  onPress={() => router.push({
                    pathname: '/activity/[eventId]',
                    params: {
                      eventId: encodeURIComponent(event.id),
                      title: event.title,
                      detail: event.detail,
                      time: event.time,
                      patientId: event.patientId,
                    },
                  })}
                />
              </MotionFadeIn>
            ))}
          </View>
        </View>
      )) : null}
      {!loading && !error && !visible.length ? (
        <View style={styles.empty}>
          <Text style={styles.emptyTitle}>{filter === 'all' ? 'No activity to show yet' : 'No matching activity'}</Text>
          <Text style={styles.emptyText}>{filter === 'all' ? 'Pair a device, record a conversation, create a routine or send a family message to create the first factual entry.' : 'Choose another filter to view the available timeline.'}</Text>
        </View>
      ) : null}

      <MotionSheet title="Filter activity" visible={filterOpen} onClose={() => setFilterOpen(false)}>
        <Text style={styles.sheetNote}>Choose which factual events appear in the timeline.</Text>
        {(['all', ...CATEGORY_FILTERS.map((item) => item.kind)] as const).map((option) => (
          <MotionPressable
            key={option}
            accessibilityRole="button"
            accessibilityState={{ selected: filter === option }}
            feedback="card"
            haptic="selection"
            onPress={() => {
              setFilter(option);
              setFilterOpen(false);
            }}
            style={[styles.filterOption, filter === option && styles.filterOptionSelected]}
          >
            <Text style={styles.filterOptionText}>{FILTER_LABELS[option]}</Text>
            <Feather color={filter === option ? colors.accent : colors.textDecorative} name={filter === option ? 'check-circle' : 'circle'} size={19} />
          </MotionPressable>
        ))}
        <MotionPressable accessibilityRole="button" feedback="button" onPress={() => { setFilterOpen(false); void refresh(); }} style={styles.sheetAction}>
          <Feather color={colors.accent} name="refresh-cw" size={19} />
          <Text style={styles.sheetActionText}>Refresh activity</Text>
        </MotionPressable>
        <SecondaryButton label="Close" onPress={() => setFilterOpen(false)} />
      </MotionSheet>
    </ScreenLayout>
  );
}

function ActivityBrand() {
  return (
    <View accessible accessibilityLabel="Reflexion" style={styles.brandLockup}>
      <Text style={styles.brandName}>Reflexion</Text>
    </View>
  );
}

function ActivityFilterButton({ active = false, icon, label, onPress }: { active?: boolean; icon: EventIcon; label: string; onPress: () => void }) {
  return (
    <MotionPressable
      accessibilityLabel={label}
      accessibilityRole="button"
      accessibilityState={{ selected: active }}
      feedback="button"
      haptic="selection"
      onPress={onPress}
      style={[styles.filterButton, active && styles.filterButtonActive]}
    >
      <Feather color={active ? colors.text.onAccent : colors.accent} name={icon} size={18} />
      <Text style={[styles.filterButtonText, active && styles.filterButtonTextActive]}>{label}</Text>
      {label !== 'All' ? <Feather color={active ? colors.text.onAccent : colors.accent} name="chevron-down" size={16} /> : null}
    </MotionPressable>
  );
}

function ActivityEventRow({ event, last, onPress }: { event: Event; last: boolean; onPress: () => void }) {
  const tone = EVENT_TONES[event.tone];
  return (
    <MotionPressable
      accessibilityLabel={event.title + '. ' + event.detail + '. ' + event.person}
      accessibilityRole="button"
      feedback="card"
      onPress={onPress}
      style={styles.eventRow}
    >
      <View style={styles.timeColumn}><Text style={styles.eventTime}>{event.clock}</Text></View>
      <View style={styles.rail}>
        {!last ? <View style={styles.railLine} /> : null}
        <View style={[styles.railDot, { backgroundColor: tone.dot }]} />
      </View>
      <View style={styles.eventCard}>
        <View style={[styles.eventIcon, { backgroundColor: tone.background }]}>
          <Feather color={tone.color} name={event.icon} size={21} />
        </View>
        <View style={styles.eventCopy}>
          <Text style={styles.eventTitle} numberOfLines={2}>{event.title}</Text>
          <Text style={styles.eventDetail} numberOfLines={2}>{event.detail}</Text>
        </View>
        <View style={styles.personBlock}>
          <Text style={styles.personName} numberOfLines={1}>{event.person}</Text>
          <Feather color={colors.text.primary} name="chevron-right" size={19} />
        </View>
      </View>
    </MotionPressable>
  );
}

function activityWindow() {
  const end = new Date();
  const start = new Date(end.getTime() - 30 * 86_400_000);
  return { from: start.toISOString(), to: new Date(end.getTime() + 86_400_000).toISOString() };
}

function formatTime(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'Time unavailable' : new Intl.DateTimeFormat('en-SG', { dateStyle: 'medium', timeStyle: 'short' }).format(date);
}

function formatClock(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '—' : new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit', hour12: true }).format(date);
}

function formatDuration(seconds: number) {
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  return minutes ? minutes + ' min' : total + ' sec';
}

function messageEvent(name: string, patientId: string, message: V1FamilyMessage): Event {
  const at = Date.parse(message.createdAt);
  const status = ({
    scheduled: 'scheduled',
    queued: 'queued',
    delivered: 'delivered to the device',
    opened: 'delivered to the device and viewed',
    expired: 'expired',
    failed: 'not delivered',
  } as Record<V1FamilyMessage['state'], string>)[message.state];
  const delivered = message.state === 'delivered' || message.state === 'opened';
  return {
    id: ['message', patientId, message.messageId].join('-'),
    patientId,
    person: name,
    title: delivered ? 'Family message delivered' : 'Family message ' + status,
    detail: message.senderName && message.senderName !== 'Your family' ? 'From ' + message.senderName : delivered ? 'From family' : status,
    time: formatTime(message.createdAt),
    clock: formatClock(message.createdAt),
    at,
    kind: 'chat',
    icon: 'message-circle',
    tone: message.state === 'failed' || message.state === 'expired' ? 'amber' : 'teal',
  };
}

function sessionEvent(name: string, patientId: string, sessionId: string, createdAt: string, duration: number): Event {
  return {
    id: ['session', patientId, sessionId].join('-'),
    patientId,
    person: name,
    title: 'Conversation recorded',
    detail: 'Duration: ' + formatDuration(duration),
    time: formatTime(createdAt),
    clock: formatClock(createdAt),
    at: Date.parse(createdAt),
    kind: 'conversation',
    icon: 'phone',
    tone: 'teal',
  };
}

function routineEvent(name: string, patientId: string, occurrence: V1ReminderOccurrence): Event {
  const at = Date.parse(occurrence.scheduledAt);
  const status = routineStatus(occurrence.status);
  const completed = occurrence.status === 'reported-complete';
  const presented = occurrence.status === 'presented';
  return {
    id: ['routine', patientId, occurrence.occurrenceId].join('-'),
    patientId,
    person: name,
    title: presented ? 'Reminder presented' : completed ? 'Reminder response recorded' : 'Routine updated',
    detail: completed ? 'Marked as completed' : occurrence.displayText || status,
    time: formatTime(occurrence.scheduledAt),
    clock: formatClock(occurrence.scheduledAt),
    at,
    kind: 'routine',
    icon: presented ? 'bell' : completed ? 'check-circle' : 'calendar',
    tone: 'teal',
  };
}

function technicalEvent(name: string, patientId: string, state: 'ok' | 'possible_issue' | 'unknown', heartbeat: string | null): Event {
  const issue = state === 'possible_issue';
  const unknown = state === 'unknown';
  const createdAt = heartbeat || new Date().toISOString();
  return {
    id: ['device', patientId, heartbeat || 'unknown'].join('-'),
    patientId,
    person: name,
    title: issue ? 'Device offline' : unknown ? 'Device status unavailable' : 'Device reconnected',
    detail: issue ? 'No connection' : unknown ? 'No heartbeat yet' : 'Connection restored',
    time: heartbeat ? formatTime(heartbeat) : 'No heartbeat yet',
    clock: heartbeat ? formatClock(heartbeat) : '—',
    at: Date.parse(createdAt),
    kind: 'technical',
    icon: issue ? 'wifi-off' : 'wifi',
    tone: issue ? 'amber' : unknown ? 'muted' : 'teal',
  };
}

function routineStatus(status: string) {
  return ({
    scheduled: 'Scheduled',
    presented: 'Presented',
    'reported-complete': 'Reported complete',
    deferred: 'Deferred',
    declined: 'Declined',
    'no-response': 'No response',
    'device-unavailable': 'Device unavailable',
  } as Record<string, string>)[status] || status;
}

function groupEvents(events: Event[]): ActivitySection[] {
  const groups = new Map<string, ActivitySection>();
  events.forEach((event) => {
    const label = sectionLabel(event.at);
    const key = label + '-' + dayKey(new Date(event.at));
    const current = groups.get(key);
    if (current) {
      current.events.push(event);
    } else {
      groups.set(key, { key, label, events: [event] });
    }
  });
  return Array.from(groups.values());
}

function sectionLabel(at: number) {
  const date = new Date(at);
  const today = new Date();
  const yesterday = new Date();
  yesterday.setDate(today.getDate() - 1);
  const key = dayKey(date);
  if (key === dayKey(today)) return 'Today';
  if (key === dayKey(yesterday)) return 'Yesterday';
  if (date > today) return 'Upcoming';
  return new Intl.DateTimeFormat('en-SG', { day: 'numeric', month: 'long' }).format(date);
}

function dayKey(date: Date) {
  return date.getFullYear() + '-' + (date.getMonth() + 1) + '-' + date.getDate();
}

const styles = StyleSheet.create({
  content: { alignSelf: 'stretch', gap: 0, minWidth: 0, paddingTop: 0, width: '100%' },
  heroBlock: { alignSelf: 'stretch', minHeight: 190, minWidth: 0, paddingTop: spacing.sm, position: 'relative', width: '100%' },
  brandLockup: { alignItems: 'flex-start', flexDirection: 'row', height: 38, minWidth: 0, position: 'relative', zIndex: 1 },
  brandName: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: scaleSize(32), fontWeight: '500', lineHeight: scaleSize(37) },
  heroCopy: { marginTop: spacing.xl, minWidth: 0, position: 'relative', zIndex: 1 },
  heroTitle: { color: colors.text.primary, fontFamily: fontFamily.ui, fontSize: fontSize.display, fontWeight: '700', lineHeight: 40, minWidth: 0 },
  heroSubtitle: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 22, marginTop: spacing.sm, maxWidth: '100%', minWidth: 0, width: '100%' },
  filterControls: { alignSelf: 'stretch', gap: spacing.sm, minWidth: 0, width: '100%' },
  primaryFilterRow: { alignItems: 'center', alignSelf: 'stretch', flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, minWidth: 0, width: '100%' },
  filterButton: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.lg, borderWidth: 1, flexDirection: 'row', flexShrink: 0, gap: 4, justifyContent: 'center', minHeight: 44, minWidth: 0, paddingHorizontal: spacing.xs },
  filterButtonActive: { backgroundColor: colors.accent, borderColor: colors.accent },
  filterButtonText: { color: colors.accent, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.body, fontWeight: '600', lineHeight: 20 },
  filterButtonTextActive: { color: colors.text.onAccent },
  categoryRow: { alignItems: 'center', alignSelf: 'stretch', flexDirection: 'row', flexWrap: 'wrap', gap: spacing.xs, minWidth: 0, width: '100%' },
  categoryChip: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.md, borderWidth: 1, flexDirection: 'row', flexShrink: 0, gap: 5, minHeight: 36, minWidth: 0, paddingHorizontal: spacing.sm },
  categoryChipActive: { backgroundColor: '#F1F7F4', borderColor: '#A9CEC7' },
  categoryChipText: { color: colors.text.secondary, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.caption, lineHeight: 18, minWidth: 0 },
  categoryChipTextActive: { color: colors.text.primary },
  loading: { marginTop: spacing.lg },
  section: { alignSelf: 'stretch', minWidth: 0, paddingTop: spacing.xl, width: '100%' },
  sectionTitle: { color: colors.text.primary, fontFamily: fontFamily.ui, fontSize: fontSize.heading, fontWeight: '700', lineHeight: 27, marginBottom: spacing.sm, minWidth: 0 },
  timeline: { alignSelf: 'stretch', gap: spacing.sm, minWidth: 0, width: '100%' },
  fade: { alignSelf: 'stretch', minWidth: 0, width: '100%' },
  eventRow: { alignItems: 'stretch', alignSelf: 'stretch', flexDirection: 'row', gap: spacing.xs, minWidth: 0, width: '100%' },
  timeColumn: { alignItems: 'flex-end', flexShrink: 0, justifyContent: 'flex-start', paddingTop: spacing.md, width: scaleSize(50) },
  eventTime: { color: colors.text.secondary, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 20, textAlign: 'right' },
  rail: { alignItems: 'center', flexShrink: 0, position: 'relative', width: scaleSize(16) },
  railLine: { backgroundColor: '#C9D3D0', bottom: -spacing.sm, left: scaleSize(7.5), position: 'absolute', top: scaleSize(22), width: 1 },
  railDot: { borderRadius: radius.pill, height: 9, marginTop: spacing.lg, width: 9, zIndex: 1 },
  eventCard: { ...cardShadow, alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.lg, borderWidth: 1, flex: 1, flexDirection: 'row', gap: spacing.xs, minHeight: 58, minWidth: 0, paddingHorizontal: spacing.xs, paddingVertical: spacing.sm },
  eventIcon: { alignItems: 'center', borderRadius: radius.pill, flexShrink: 0, height: 38, justifyContent: 'center', width: 38 },
  eventCopy: { flex: 1, flexShrink: 1, minWidth: 0 },
  eventTitle: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 21, minWidth: 0 },
  eventDetail: { color: colors.text.secondary, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 19, marginTop: 1, minWidth: 0 },
  personBlock: { alignItems: 'center', flexDirection: 'row', flexShrink: 1, gap: 2, justifyContent: 'flex-end', maxWidth: '30%', minWidth: 0 },
  personName: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.display, fontSize: fontSize.body, fontStyle: 'italic', lineHeight: 20, minWidth: 0, textAlign: 'right' },
  empty: { ...cardShadow, backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, gap: spacing.sm, marginTop: spacing.xl, minWidth: 0, padding: spacing.xl },
  emptyTitle: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.display, fontSize: fontSize.heading, lineHeight: 27, minWidth: 0 },
  emptyText: { color: colors.text.secondary, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 22, minWidth: 0 },
  sheetNote: { color: colors.text.secondary, fontFamily: fontFamily.ui, fontSize: fontSize.caption, lineHeight: 19 },
  filterOption: { alignItems: 'center', borderColor: colors.border.default, borderRadius: radius.md, borderWidth: 1, flexDirection: 'row', justifyContent: 'space-between', minHeight: 52, minWidth: 0, paddingHorizontal: spacing.lg },
  filterOptionSelected: { backgroundColor: '#E7F3F0', borderColor: colors.accent },
  filterOptionText: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.bodyLarge, minWidth: 0 },
  sheetAction: { alignItems: 'center', alignSelf: 'flex-start', flexDirection: 'row', gap: spacing.sm, minHeight: 44, paddingHorizontal: spacing.sm },
  sheetActionText: { color: colors.accent, fontFamily: fontFamily.ui, fontSize: fontSize.body, fontWeight: '700' },
});
