import { useQuery } from '@tanstack/react-query';
import React, { useCallback, useMemo, useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useLocalSearchParams } from 'expo-router';
import { Feather } from '@expo/vector-icons';
import { EmptyState, ErrorState, LoadingState } from '../../src/components/ScreenState';
import { getSessionDayV1 } from '../../src/lib/v1Caregiver';
import {
  MIN_TOUCH_TARGET, cardShadow, colors, fontFamily, fontSize, radius, scaleSize, spacing,
} from '../../src/theme';

type ConversationLog = {
  sentence: string;
  role: string;
  words: number;
  duration: number;
  wordsPerSecond: number;
};

type RealConversationSession = {
  id: string;
  patientId: string;
  patientName: string;
  duration: number;
  words: number;
  exchanges: number;
  avgLatency: number;
  createdAt: string | null;
  updatedAt: string | null;
  logs: ConversationLog[];
};

type TodaySessionsResponse = {
  date: string;
  patientId: string;
  patientName: string;
  sessions: RealConversationSession[];
};

export default function SessionReplayScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const [selectedSessionIndex, setSelectedSessionIndex] = useState(0);
  const [showTranscript, setShowTranscript] = useState(true);
  // Any non-empty id is real. This used to test /^[0-9a-f]{24}$/, which was the legacy nurse/patient
  // ObjectId hex — but v1 mints `pat_…` ids for loved ones created since the migration, and CLAUDE.md is
  // explicit that v1 ids are opaque strings. The old guard silently blanked the screen for them.
  const shouldLoadRealSession = Boolean(id);
  const today = getSingaporeDateKey(new Date());
  const todaySessionsQuery = useQuery({
    enabled: shouldLoadRealSession,
    queryKey: ['sessionDay', id, today],
    queryFn: async () => {
      const body = await getSessionDayV1(id, today);
      return {
        date: body?.date || today,
        patientId: body?.patientId || id,
        patientName: body?.patientName || 'Patient',
        sessions: Array.isArray(body?.sessions) ? body.sessions : [],
      } satisfies TodaySessionsResponse;
    },
  });
  const { refetch: refetchTodaySessions } = todaySessionsQuery;
  useFocusEffect(
    useCallback(() => {
      if (shouldLoadRealSession) {
        void refetchTodaySessions();
      }
    }, [refetchTodaySessions, shouldLoadRealSession]),
  );
  const todaySessions = todaySessionsQuery.data || null;
  // Clamped, because the index survives a refetch that returns fewer sessions than it points at. When that
  // happened the screen fell through to the empty state and told the caregiver there was no conversation on
  // a day that had one — and with only one session left the tab row is hidden, so there was no control left
  // to get back.
  const availableSessions = todaySessions?.sessions || [];
  const safeSessionIndex = Math.min(selectedSessionIndex, Math.max(availableSessions.length - 1, 0));
  const realSession = availableSessions[safeSessionIndex] || null;
  const realTranscript = useMemo(
    () => buildTranscript(realSession?.logs || []),
    [realSession?.logs],
  );

  // Four situations, four screens: link we cannot open, still loading, request failed, nothing recorded.
  // They used to collapse into one "not ready to show yet" card, so a dead request looked exactly like a
  // quiet day — and the caregiver had no way to retry.
  if (!shouldLoadRealSession) return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.content}>
        <EmptyState
          icon="activity"
          title="Bear with us"
          message="This session is not ready to show yet."
        />
      </View>
    </SafeAreaView>
  );

  if (!todaySessions && todaySessionsQuery.isLoading) return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.content}>
        <LoadingState message="Fetching today's conversation." />
      </View>
    </SafeAreaView>
  );

  // Never the server's error text — a raw string like "Not found" reads as news about the person.
  if (!todaySessions && todaySessionsQuery.error) return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.content}>
        <ErrorState
          title="We could not load this conversation"
          // isLoading stays false once a query is in `error`, so without isFetching the retry gave no
          // acknowledgement at all and a caregiver on a slow connection just tapped again.
          message={todaySessionsQuery.isFetching
            ? 'Trying again…'
            : 'This is usually a connection problem, not something about your loved one.'}
          onRetry={todaySessionsQuery.isFetching ? undefined : () => void todaySessionsQuery.refetch()}
        />
      </View>
    </SafeAreaView>
  );

  if (!todaySessions) return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.content}>
        <EmptyState
          icon="message-circle"
          title="Nothing recorded here yet"
          message="Today's conversation will appear on this page once it has been saved."
        />
      </View>
    </SafeAreaView>
  );

  const selectedSession = realSession;

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.content}>
        {/* A refresh that fails once we already have data must not be silent: this screen refetches on
            every focus, so without this the caregiver could sit on yesterday's card believing it is live. */}
        {todaySessionsQuery.error ? (
          <View accessibilityLiveRegion="polite">
            <ErrorState
              compact
              title="We could not refresh just now"
              message="This is usually a connection problem, not something about your loved one."
              retryLabel="Refresh"
              onRetry={() => void todaySessionsQuery.refetch()}
            />
          </View>
        ) : null}

        {todaySessions.sessions.length > 1 ? (
          <View style={styles.sessionTabs}>
            {todaySessions.sessions.map((item, index) => (
              <TouchableOpacity
                accessibilityLabel={`Conversation ${index + 1} of ${todaySessions.sessions.length}, ${formatSessionTabLabel(item, index)}`}
                accessibilityRole="tab"
                accessibilityState={{ selected: safeSessionIndex === index }}
                key={item.id || index}
                onPress={() => setSelectedSessionIndex(index)}
                style={[styles.sessionTab, safeSessionIndex === index && styles.sessionTabActive]}
              >
                <Text style={[styles.sessionTabText, safeSessionIndex === index && styles.sessionTabTextActive]}>
                  {formatSessionTabLabel(item, index)}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        ) : null}

        {!selectedSession ? (
          // Reports only what this screen actually knows — that nothing has been recorded here yet. The
          // earlier wording ("Today's check-in has not happened yet") asserted a fact about the person's
          // day that the screen cannot verify.
          <EmptyState
            icon="message-circle"
            title="Nothing recorded here yet"
            message="Today's conversation will appear on this page once it has been saved."
          />
        ) : (
          <>
        <View style={styles.card}>
          <View style={styles.metaRow}>
            <Text maxFontSizeMultiplier={1.4} style={styles.metaName}>{todaySessions.patientName}</Text>
          </View>
          <Text style={styles.metaDate}>{formatDateTime(selectedSession.createdAt)}</Text>
          <View style={styles.statsRow}>
            <StatChip icon="clock" label="Duration" value={formatDuration(selectedSession.duration)} />
            <StatChip icon="message-circle" label="Words" value={String(selectedSession.words)} />
            <StatChip icon="repeat" label="Exchanges" value={String(selectedSession.exchanges)} />
            <StatChip icon="zap" label="Speech lag" value={`${selectedSession.avgLatency.toFixed(1)}s`} highlight />
          </View>
        </View>

        <View style={styles.card}>
          <TouchableOpacity
            accessibilityRole="button"
            accessibilityState={{ expanded: showTranscript }}
            style={styles.transcriptHeader}
            onPress={() => setShowTranscript(v => !v)}
          >
            <Text style={styles.cardTitle}>View the full conversation with {todaySessions.patientName} below</Text>
            {/* Chevron duplicates the expanded state already announced above. */}
            <Feather
              accessibilityElementsHidden
              importantForAccessibility="no"
              name={showTranscript ? 'chevron-up' : 'chevron-down'}
              size={16}
              color={colors.accent}
            />
          </TouchableOpacity>

          {showTranscript && realTranscript.length > 0 && (
            <View style={styles.transcript}>
              {realTranscript.map((line, i) => {
                const speaker = line.speaker === 'Aria' ? 'Aria' : todaySessions.patientName;
                return (
                  // Grouped into one label so a screen reader reads a whole turn instead of speaker,
                  // sentence and timestamp as three separate stops.
                  <View
                    accessible
                    accessibilityLabel={`${speaker}, ${formatSeconds(line.timestamp)}. ${line.text}`}
                    key={i}
                    style={[styles.line, line.speaker === 'Aria' ? styles.lineAria : styles.lineUser]}
                  >
                    <Text style={styles.lineLabel}>{speaker}</Text>
                    <Text style={styles.lineText}>{line.text}</Text>
                    <Text style={styles.lineTime}>{formatSeconds(line.timestamp)}</Text>
                  </View>
                );
              })}
            </View>
          )}
          {showTranscript && realTranscript.length === 0 && (
            <Text style={styles.emptyTranscript}>No transcript available for this session.</Text>
          )}
        </View>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function StatChip({
  highlight = false,
  icon,
  label,
  value,
}: {
  highlight?: boolean;
  icon: any;
  label: string;
  value: string;
}) {
  return (
    <View
      accessible
      accessibilityLabel={`${label}: ${value}`}
      style={[styles.statChip, highlight && styles.statChipHighlight]}
    >
      <Feather
        accessibilityElementsHidden
        importantForAccessibility="no"
        name={icon}
        size={14}
        color={highlight ? colors.accentPressed : colors.text.tertiary}
      />
      {/* Four chips share one row, so the number is the one thing here that cannot grow without clipping. */}
      <Text maxFontSizeMultiplier={1.6} style={[styles.statValue, highlight && styles.statValueHighlight]}>
        {value}
      </Text>
      <Text style={[styles.statLabel, highlight && styles.statLabelHighlight]}>{label}</Text>
    </View>
  );
}

function formatSeconds(s: number): string {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

function formatDuration(seconds: number): string {
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function formatDateTime(value: string | null) {
  if (!value) {
    return 'No date available';
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return 'No date available';
  }

  return `${date.toLocaleDateString('en-CA')} · ${date.toLocaleTimeString('en-SG', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })}`;
}

function formatSessionTabLabel(session: RealConversationSession, index: number) {
  const value = session.createdAt;
  if (!value) return `Session ${index + 1}`;

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return `Session ${index + 1}`;

  return date.toLocaleTimeString('en-SG', {
    hour: 'numeric',
    minute: '2-digit',
  }).replace(/\s/g, '').toLowerCase();
}

function buildTranscript(logs: ConversationLog[]) {
  let timestamp = 0;
  return logs.map((log) => {
    const line = {
      speaker: log.role.toLowerCase() === 'ai' ? 'Aria' : 'Patient',
      text: log.sentence,
      timestamp: Math.round(timestamp),
    };
    timestamp += log.duration || 0;
    return line;
  });
}

function getSingaporeDateKey(date: Date) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    day: '2-digit',
    month: '2-digit',
    timeZone: 'Asia/Singapore',
    year: 'numeric',
  }).formatToParts(date);
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}`;
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.surface.page },

  content: { paddingHorizontal: spacing.xl, paddingBottom: 48, paddingTop: spacing.lg },

  sessionTabs: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
    marginBottom: 14,
  },
  sessionTab: {
    alignItems: 'center',
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: radius.pill,
    borderWidth: 1,
    justifyContent: 'center',
    // hitSlop would overlap the neighbouring pill (8pt gap), so the pill itself carries the 44pt target.
    minHeight: MIN_TOUCH_TARGET,
    paddingHorizontal: 14,
    paddingVertical: spacing.sm,
  },
  sessionTabActive: {
    backgroundColor: colors.accent,
    borderColor: colors.accent,
  },
  sessionTabText: {
    color: colors.text.secondary,
    fontSize: fontSize.body,
    fontWeight: '700',
  },
  sessionTabTextActive: {
    color: colors.text.onAccent,
  },

  card: {
    backgroundColor: colors.surface.card,
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border.default,
    padding: 18,
    marginBottom: 14,
    ...cardShadow,
  },
  metaRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: spacing.xs },
  metaName: { fontSize: scaleSize(18), fontWeight: '500', color: colors.text.primary, fontFamily: fontFamily.display },
  metaDate: { fontSize: fontSize.body, color: colors.text.tertiary, marginBottom: spacing.lg },

  statsRow: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', gap: spacing.sm },
  statChip: {
    width: '47%',
    backgroundColor: colors.surface.page,
    borderRadius: 12,
    padding: 10,
    alignItems: 'center',
    gap: spacing.xs,
    borderWidth: 1,
    borderColor: colors.border.default,
  },
  statValue: { fontSize: fontSize.subheading, fontWeight: '600', color: colors.text.primary },
  statLabel: { fontSize: fontSize.caption, color: colors.text.tertiary, textAlign: 'center' },
  statChipHighlight: { borderColor: '#CFA7B7', backgroundColor: '#F7EEF2' },
  statValueHighlight: { color: colors.accentPressed },
  statLabelHighlight: { color: colors.accentPressed, fontWeight: '700' },

  cardTitle: {
    flex: 1,
    fontSize: fontSize.caption,
    fontWeight: '600',
    color: colors.text.tertiary,
    textTransform: 'uppercase',
    letterSpacing: 0.6,
    marginBottom: 10,
  },
  summaryText: { fontSize: fontSize.bodyLarge, color: colors.text.secondary, lineHeight: 21 },
  topicRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, marginTop: spacing.md },
  topicChip: {
    backgroundColor: colors.surface.muted,
    paddingHorizontal: spacing.md,
    paddingVertical: 5,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.border.default,
  },
  topicText: { fontSize: fontSize.caption, color: colors.text.secondary, fontWeight: '600' },

  transcriptHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: spacing.md,
    justifyContent: 'space-between',
    marginBottom: spacing.xs,
    minHeight: MIN_TOUCH_TARGET,
  },
  transcript: { gap: 10, marginTop: 10 },
  line: { borderRadius: 12, padding: spacing.md },
  lineAria: { backgroundColor: '#F3E8ED', borderLeftWidth: 3, borderLeftColor: colors.accent },
  lineUser: { backgroundColor: colors.surface.muted, borderLeftWidth: 3, borderLeftColor: '#B9AA99' },
  lineLabel: { fontSize: fontSize.caption, fontWeight: '700', color: colors.text.tertiary, marginBottom: spacing.xs },
  lineText: { fontSize: fontSize.bodyLarge, color: colors.text.primary, lineHeight: 20 },
  lineTime: { fontSize: fontSize.caption, color: colors.text.tertiary, marginTop: spacing.xs, textAlign: 'right' },
  emptyTranscript: {
    fontSize: fontSize.bodyLarge,
    color: colors.text.tertiary,
    textAlign: 'center',
    paddingVertical: spacing.lg,
  },
});
