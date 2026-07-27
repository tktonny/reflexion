import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useLocalSearchParams } from 'expo-router';
import { EmptyState, ErrorState, LoadingState } from '../../../src/components/ScreenState';
import { generateSessionSummaryV1, getSessionDayV1 } from '../../../src/lib/v1Caregiver';
import { cardShadow, colors, fontFamily, fontSize, MIN_TOUCH_TARGET, radius, scaleSize, spacing } from '../../../src/theme';

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

type SessionsByDayResponse = {
  date: string;
  patientId: string;
  patientName: string;
  aiSummary: string;
  sessions: RealConversationSession[];
};

export default function SessionHistoryDayScreen() {
  const { id, date } = useLocalSearchParams<{ id: string; date: string }>();
  const queryClient = useQueryClient();
  const [selectedSessionIndex, setSelectedSessionIndex] = useState(0);
  const [generatedSummary, setGeneratedSummary] = useState('');
  const [showTranscript, setShowTranscript] = useState(true);
  // Any non-empty id is real. This used to test /^[0-9a-f]{24}$/, which was the legacy nurse/patient
  // ObjectId hex — but v1 mints `pat_…` ids for loved ones created since the migration, and CLAUDE.md is
  // explicit that v1 ids are opaque strings. The old guard silently blanked the screen for them.
  const shouldLoadRealSession = Boolean(id && date && /^\d{4}-\d{2}-\d{2}$/.test(date));
  const daySessionsQuery = useQuery({
    enabled: shouldLoadRealSession,
    queryKey: ['sessionDay', id, date],
    queryFn: async () => {
      const body = await getSessionDayV1(id, date);
      return {
        date: body?.date || date,
        patientId: body?.patientId || id,
        patientName: body?.patientName || 'Patient',
        // v1 has no cached summary on the day resource — it is generated on demand by the button below,
        // which is why the legacy field was usually empty here anyway.
        aiSummary: '',
        sessions: Array.isArray(body?.sessions) ? body.sessions : [],
      } satisfies SessionsByDayResponse;
    },
  });
  const { refetch: refetchDaySessions } = daySessionsQuery;
  useFocusEffect(
    useCallback(() => {
      if (shouldLoadRealSession) {
        void refetchDaySessions();
      }
    }, [refetchDaySessions, shouldLoadRealSession]),
  );
  const daySessions = daySessionsQuery.data || null;
  const summaryMutation = useMutation({
    mutationFn: () => generateSessionSummaryV1(id, date),
    onSuccess: async (body) => {
      // summary:null with a reason is how a quiet day answers — a result to show, not a failure.
      setGeneratedSummary(body?.summary || 'There was no conversation to summarise for this day.');
      await queryClient.invalidateQueries({ queryKey: ['sessionDay', id, date] });
    },
    onError: (err) => {
      Alert.alert(
        'Unable to generate summary',
        err instanceof Error ? err.message : 'Unable to generate summary.',
      );
    },
  });

  const sessions = daySessions?.sessions || [];
  const hasSessions = sessions.length > 0;
  const selectedSession = sessions[selectedSessionIndex] || null;
  const transcript = useMemo(
    () => buildTranscript(selectedSession?.logs || []),
    [selectedSession?.logs],
  );

  useEffect(() => {
    if (!daySessionsQuery.data) return;
    setGeneratedSummary(daySessionsQuery.data.aiSummary || '');
    setSelectedSessionIndex(0);
    setShowTranscript(true);
  }, [daySessionsQuery.data]);

  // The route params are missing or malformed, so the query never runs. Nothing to retry here — this is the
  // disabled case, deliberately kept distinct from a request that actually failed below.
  if (!shouldLoadRealSession) {
    return (
      <SafeAreaView style={styles.safe}>
        <View style={styles.placeholder}>
          <EmptyState
            icon="activity"
            title="Bear with us"
            message="This session day is not ready to show yet."
          />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.card}>
          <Text style={styles.cardTitle}>{formatSelectedDate(date)}</Text>
          {hasSessions ? (
            <View style={styles.sessionTabs}>
              {sessions.map((item, index) => (
                <TouchableOpacity
                  accessibilityLabel={`Session ${index + 1} of ${sessions.length}, ${formatSessionTabLabel(item, index)}`}
                  accessibilityRole="tab"
                  accessibilityState={{ selected: selectedSessionIndex === index }}
                  key={item.id || index}
                  onPress={() => setSelectedSessionIndex(index)}
                  style={[styles.sessionTab, selectedSessionIndex === index && styles.sessionTabActive]}
                >
                  <Text style={[styles.sessionTabText, selectedSessionIndex === index && styles.sessionTabTextActive]}>
                    {formatSessionTabLabel(item, index)}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          ) : null}
        </View>

        {/* Loading, failed and genuinely-empty are three different answers to "did she talk to Aria that day?"
            and the caregiver deserves to be told which. The failure branch never renders the server's own
            message: a 404 on this endpoint used to reach the screen as the headline "Not found". */}
        {/* Order matters: the error card only appears when there is nothing to show. A failed refetch keeps
            its cached data (retry:1 + staleTime Infinity + the focus refetch above), so checking isError
            first would stack "we could not load this day" directly on top of the day's own, still-accurate
            session cards. */}
        {daySessionsQuery.isLoading ? (
          <LoadingState title="Loading this day" />
        ) : daySessionsQuery.isError && !hasSessions ? (
          <ErrorState
            title="We could not load this day"
            onRetry={() => void daySessionsQuery.refetch()}
          />
        ) : !hasSessions ? (
          <EmptyState
            icon="calendar"
            title="No check-in on this day"
            message="Nothing was recorded for this date. The other days in the history list are still there."
          />
        ) : null}

        {selectedSession && daySessions ? (
          <>
            <View style={styles.card}>
              <View style={styles.summaryHeader}>
                <Text style={styles.cardTitle}>AI summary</Text>
                {!generatedSummary ? (
                  <TouchableOpacity
                    accessibilityRole="button"
                    disabled={summaryMutation.isPending}
                    onPress={() => summaryMutation.mutate()}
                    style={[styles.summaryBtn, summaryMutation.isPending && styles.summaryBtnDisabled]}
                  >
                    <Feather
                      accessibilityElementsHidden
                      importantForAccessibility="no"
                      name="cpu"
                      size={14}
                      color={colors.accent}
                    />
                    <Text style={styles.summaryBtnText}>
                      {summaryMutation.isPending ? 'Generating...' : 'Generate'}
                    </Text>
                  </TouchableOpacity>
                ) : null}
              </View>
              {/* Announced when it arrives, so pressing Generate is not a silent action for a screen reader. */}
              <Text
                accessibilityLiveRegion="polite"
                style={generatedSummary ? styles.summaryText : styles.emptyText}
              >
                {generatedSummary || 'No summary yet.'}
              </Text>
            </View>

            <View style={styles.card}>
              {/* No status indicator here on purpose: this card is one day's raw conversation detail, and the
                  only status the app may show is the one the server computes on the dashboard/profile. */}
              <View style={styles.metaRow}>
                <Text maxFontSizeMultiplier={1.4} style={styles.metaName}>{daySessions.patientName}</Text>
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
                onPress={() => setShowTranscript(value => !value)}
                style={styles.transcriptHeader}
              >
                {/* flexShrink keeps this sentence inside the row instead of pushing the chevron off-screen
                    once the system font is scaled up. */}
                <Text style={[styles.cardTitle, styles.transcriptTitle]}>
                  View the full conversation with {daySessions.patientName} below
                </Text>
                <Feather
                  accessibilityElementsHidden
                  importantForAccessibility="no"
                  name={showTranscript ? 'chevron-up' : 'chevron-down'}
                  size={16}
                  color={colors.accent}
                />
              </TouchableOpacity>

              {showTranscript && transcript.length > 0 ? (
                <View style={styles.transcript}>
                  {transcript.map((line, index) => {
                    const speaker = line.speaker === 'Aria' ? 'Aria' : daySessions.patientName;
                    return (
                      // Grouped into one label so a long transcript is not read as three fragments per line.
                      <View
                        accessible
                        accessibilityLabel={`${speaker}, at ${formatSeconds(line.timestamp)}: ${line.text}`}
                        key={index}
                        style={[styles.line, line.speaker === 'Aria' ? styles.lineAria : styles.lineUser]}
                      >
                        <Text style={styles.lineLabel}>{speaker}</Text>
                        <Text style={styles.lineText}>{line.text}</Text>
                        <Text style={styles.lineTime}>{formatSeconds(line.timestamp)}</Text>
                      </View>
                    );
                  })}
                </View>
              ) : null}
              {showTranscript && transcript.length === 0 ? (
                <Text style={styles.emptyTranscript}>No transcript available for this session.</Text>
              ) : null}
            </View>
          </>
        ) : null}
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
    // One label per chip ("Words, 214"), otherwise the four chips read as eight disconnected fragments.
    <View
      accessible
      accessibilityLabel={`${label}, ${value}`}
      style={[styles.statChip, highlight && styles.statChipHighlight]}
    >
      <Feather
        accessibilityElementsHidden
        importantForAccessibility="no"
        name={icon}
        size={14}
        color={highlight ? colors.accentPressed : colors.text.tertiary}
      />
      <Text style={[styles.statValue, highlight && styles.statValueHighlight]}>{value}</Text>
      <Text style={[styles.statLabel, highlight && styles.statLabelHighlight]}>{label}</Text>
    </View>
  );
}

function formatSelectedDate(dateKey: string) {
  const [year, month, day] = dateKey.split('-').map(Number);
  return new Date(year, month - 1, day).toLocaleDateString('en-SG', {
    day: 'numeric',
    month: 'long',
    weekday: 'long',
    year: 'numeric',
  });
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
  if (!value) return 'No date available';

  const sessionDate = new Date(value);
  if (Number.isNaN(sessionDate.getTime())) return 'No date available';

  return `${sessionDate.toLocaleDateString('en-CA')} · ${sessionDate.toLocaleTimeString('en-SG', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })}`;
}

function formatSessionTabLabel(session: RealConversationSession, index: number) {
  const value = session.createdAt;
  if (!value) return `Session ${index + 1}`;

  const sessionDate = new Date(value);
  if (Number.isNaN(sessionDate.getTime())) return `Session ${index + 1}`;

  return sessionDate.toLocaleTimeString('en-SG', {
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

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.surface.page },
  placeholder: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: spacing.md,
  },
  content: { paddingBottom: 48, paddingHorizontal: spacing.md, paddingTop: spacing.lg },
  card: {
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: radius.xl,
    borderWidth: 1,
    marginBottom: 14,
    padding: 14,
    ...cardShadow,
  },
  cardTitle: {
    color: colors.text.tertiary,
    fontSize: fontSize.caption,
    fontWeight: '600',
    letterSpacing: 0.6,
    marginBottom: 10,
    textTransform: 'uppercase',
  },
  emptyText: { color: colors.text.tertiary, fontSize: fontSize.bodyLarge, lineHeight: 21 },
  summaryHeader: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    gap: spacing.sm,
    marginBottom: 10,
  },
  summaryBtn: {
    alignItems: 'center',
    backgroundColor: colors.surface.muted,
    borderColor: colors.border.default,
    borderRadius: radius.pill,
    borderWidth: 1,
    flexDirection: 'row',
    gap: 6,
    maxWidth: '100%',
    // 44pt: this is the only action on the card and it sits in a tight header row.
    minHeight: MIN_TOUCH_TARGET,
    paddingHorizontal: 10,
    paddingVertical: 7,
  },
  summaryBtnDisabled: {
    opacity: 0.65,
  },
  summaryBtnText: {
    color: colors.accent,
    fontSize: fontSize.body,
    fontWeight: '700',
    flexShrink: 1,
  },
  summaryText: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 21 },
  sessionTabs: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  sessionTab: {
    alignItems: 'center',
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: radius.pill,
    borderWidth: 1,
    justifyContent: 'center',
    // These pills were ~34pt tall; switching session is the main control on this screen.
    minHeight: MIN_TOUCH_TARGET,
    minWidth: MIN_TOUCH_TARGET,
    paddingHorizontal: 14,
    paddingVertical: spacing.sm,
  },
  sessionTabActive: {
    backgroundColor: colors.accent,
    borderColor: colors.accent,
  },
  sessionTabText: { color: colors.text.secondary, fontSize: fontSize.body, fontWeight: '700' },
  sessionTabTextActive: { color: colors.text.onAccent },
  metaRow: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between', marginBottom: spacing.xs },
  metaName: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: scaleSize(18), fontWeight: '500' },
  metaDate: { color: colors.text.tertiary, fontSize: fontSize.body, marginBottom: spacing.lg },
  statsRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, justifyContent: 'space-between' },
  statChip: {
    alignItems: 'center',
    backgroundColor: colors.surface.page,
    borderColor: colors.border.default,
    borderRadius: 12,
    borderWidth: 1,
    gap: spacing.xs,
    minHeight: 74,
    paddingHorizontal: 6,
    paddingVertical: spacing.sm,
    width: '47%',
  },
  statValue: {
    color: colors.text.primary,
    fontSize: fontSize.subheading,
    fontWeight: '600',
    lineHeight: 19,
    textAlign: 'center',
  },
  // 10px was unreadable for the caregivers this app is for; these four words are the point of the chip.
  statLabel: { color: colors.text.tertiary, fontSize: fontSize.caption, lineHeight: 15, textAlign: 'center' },
  statChipHighlight: { backgroundColor: '#F7EEF2', borderColor: '#CFA7B7' },
  statValueHighlight: { color: colors.accentPressed },
  statLabelHighlight: { color: colors.accentPressed, fontWeight: '700' },
  transcriptHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: spacing.sm,
    justifyContent: 'space-between',
    marginBottom: spacing.xs,
    // The whole row is the expand/collapse target, so it has to be reachable in a hurry.
    minHeight: MIN_TOUCH_TARGET,
  },
  transcriptTitle: { flexShrink: 1 },
  transcript: { gap: 10, marginTop: 10 },
  line: { borderRadius: radius.md, padding: spacing.md },
  lineAria: { backgroundColor: '#F3E8ED', borderLeftColor: colors.accent, borderLeftWidth: 3 },
  lineUser: { backgroundColor: colors.surface.muted, borderLeftColor: '#B9AA99', borderLeftWidth: 3 },
  lineLabel: { color: colors.text.tertiary, fontSize: fontSize.caption, fontWeight: '700', marginBottom: spacing.xs },
  lineText: { color: colors.text.primary, fontSize: fontSize.bodyLarge, lineHeight: 20 },
  lineTime: { color: colors.text.tertiary, fontSize: fontSize.caption, marginTop: spacing.xs, textAlign: 'right' },
  emptyTranscript: {
    color: colors.text.tertiary,
    fontSize: fontSize.bodyLarge,
    paddingVertical: spacing.lg,
    textAlign: 'center',
  },
});
