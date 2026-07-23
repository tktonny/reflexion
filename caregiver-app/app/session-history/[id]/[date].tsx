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
import StatusBadge from '../../../src/components/StatusBadge';
import { apiGet, apiSend } from '../../../src/lib/apiClient';

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
  const shouldLoadRealSession = Boolean(
    id && /^[0-9a-f]{24}$/i.test(id) && date && /^\d{4}-\d{2}-\d{2}$/.test(date),
  );
  const daySessionsQuery = useQuery({
    enabled: shouldLoadRealSession,
    queryKey: ['sessionDay', id, date],
    queryFn: async () => {
      const body = await apiGet<Partial<SessionsByDayResponse>>(
        `/api/conversation-sessions-by-day?id=${encodeURIComponent(id)}&date=${encodeURIComponent(date)}`,
      );
      return {
        date: body?.date || date,
        patientId: body?.patientId || id,
        patientName: body?.patientName || 'Patient',
        aiSummary: body?.aiSummary || '',
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
    mutationFn: () => apiSend<{ summary?: string }>('/api/patient-summary', {
      method: 'POST',
      body: JSON.stringify({ patientId: id, date }),
    }),
    onSuccess: async (body) => {
      setGeneratedSummary(body?.summary || 'No summary generated.');
      await queryClient.invalidateQueries({ queryKey: ['sessionDay', id, date] });
    },
    onError: (err) => {
      Alert.alert(
        'Unable to generate summary',
        err instanceof Error ? err.message : 'Unable to generate summary.',
      );
    },
  });

  const selectedSession = daySessions?.sessions[selectedSessionIndex] || null;
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

  if (!shouldLoadRealSession) {
    return (
      <SafeAreaView style={styles.safe}>
        <View style={styles.placeholder}>
          <Feather name="activity" size={28} color="#87566A" />
          <Text style={styles.placeholderTitle}>Bear with us</Text>
          <Text style={styles.placeholderText}>This session day is not ready to show yet.</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.card}>
          <Text style={styles.cardTitle}>{formatSelectedDate(date)}</Text>
          {daySessionsQuery.isLoading ? (
            <Text style={styles.emptyText}>Loading sessions...</Text>
          ) : daySessions && daySessions.sessions.length > 0 ? (
            <View style={styles.sessionTabs}>
              {daySessions.sessions.map((item, index) => (
                <TouchableOpacity
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
          ) : (
            <Text style={styles.emptyText}>No sessions recorded on this day.</Text>
          )}
        </View>

        {selectedSession && daySessions ? (
          <>
            <View style={styles.card}>
              <View style={styles.summaryHeader}>
                <Text style={styles.cardTitle}>AI summary</Text>
                {!generatedSummary ? (
                  <TouchableOpacity
                    disabled={summaryMutation.isPending}
                    onPress={() => summaryMutation.mutate()}
                    style={[styles.summaryBtn, summaryMutation.isPending && styles.summaryBtnDisabled]}
                  >
                    <Feather name="cpu" size={14} color="#87566A" />
                    <Text style={styles.summaryBtnText}>
                      {summaryMutation.isPending ? 'Generating...' : 'Generate'}
                    </Text>
                  </TouchableOpacity>
                ) : null}
              </View>
              <Text style={generatedSummary ? styles.summaryText : styles.emptyText}>
                {generatedSummary || 'No summary yet.'}
              </Text>
            </View>

            <View style={styles.card}>
              <View style={styles.metaRow}>
                <Text style={styles.metaName}>{daySessions.patientName}</Text>
                <StatusBadge status="green" label="Normal" />
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
              <TouchableOpacity style={styles.transcriptHeader} onPress={() => setShowTranscript(value => !value)}>
                <Text style={styles.cardTitle}>View the full conversation with {daySessions.patientName} below</Text>
                <Feather name={showTranscript ? 'chevron-up' : 'chevron-down'} size={16} color="#87566A" />
              </TouchableOpacity>

              {showTranscript && transcript.length > 0 ? (
                <View style={styles.transcript}>
                  {transcript.map((line, index) => (
                    <View key={index} style={[styles.line, line.speaker === 'Aria' ? styles.lineAria : styles.lineUser]}>
                      <Text style={styles.lineLabel}>
                        {line.speaker === 'Aria' ? 'Aria' : daySessions.patientName}
                      </Text>
                      <Text style={styles.lineText}>{line.text}</Text>
                      <Text style={styles.lineTime}>{formatSeconds(line.timestamp)}</Text>
                    </View>
                  ))}
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
    <View style={[styles.statChip, highlight && styles.statChipHighlight]}>
      <Feather name={icon} size={14} color={highlight ? '#6E2F48' : '#A69C92'} />
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
  safe: { flex: 1, backgroundColor: '#F8F3EC' },
  placeholder: {
    alignItems: 'center',
    flex: 1,
    gap: 8,
    justifyContent: 'center',
    paddingHorizontal: 28,
  },
  placeholderTitle: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 24, fontWeight: '500' },
  placeholderText: { color: '#756C64', fontSize: 15, lineHeight: 22, textAlign: 'center' },
  content: { paddingBottom: 48, paddingHorizontal: 14, paddingTop: 16 },
  card: {
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 16,
    borderWidth: 1,
    elevation: 2,
    marginBottom: 14,
    padding: 14,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.035,
    shadowRadius: 10,
  },
  cardTitle: {
    color: '#A69C92',
    fontSize: 12,
    fontWeight: '600',
    letterSpacing: 0.6,
    marginBottom: 10,
    textTransform: 'uppercase',
  },
  emptyText: { color: '#A69C92', fontSize: 14, lineHeight: 21 },
  summaryHeader: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    gap: 8,
    marginBottom: 10,
  },
  summaryBtn: {
    alignItems: 'center',
    backgroundColor: '#F4F0EA',
    borderColor: '#E7DED2',
    borderRadius: 999,
    borderWidth: 1,
    flexDirection: 'row',
    gap: 6,
    maxWidth: '100%',
    minHeight: 36,
    paddingHorizontal: 10,
    paddingVertical: 7,
  },
  summaryBtnDisabled: {
    opacity: 0.65,
  },
  summaryBtnText: {
    color: '#87566A',
    fontSize: 11,
    fontWeight: '700',
    flexShrink: 1,
  },
  summaryText: { color: '#756C64', fontSize: 14, lineHeight: 21 },
  sessionTabs: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  sessionTab: {
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 999,
    borderWidth: 1,
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  sessionTabActive: {
    backgroundColor: '#87566A',
    borderColor: '#87566A',
  },
  sessionTabText: { color: '#756C64', fontSize: 13, fontWeight: '700' },
  sessionTabTextActive: { color: '#FFFFFF' },
  metaRow: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between', marginBottom: 4 },
  metaName: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 18, fontWeight: '500' },
  metaDate: { color: '#A69C92', fontSize: 13, marginBottom: 16 },
  statsRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, justifyContent: 'space-between' },
  statChip: {
    alignItems: 'center',
    backgroundColor: '#F8F3EC',
    borderColor: '#E7DED2',
    borderRadius: 12,
    borderWidth: 1,
    gap: 4,
    minHeight: 74,
    paddingHorizontal: 6,
    paddingVertical: 8,
    width: '47%',
  },
  statValue: { color: '#2B2522', fontSize: 15, fontWeight: '600', lineHeight: 19, textAlign: 'center' },
  statLabel: { color: '#A69C92', fontSize: 10, lineHeight: 12, textAlign: 'center' },
  statChipHighlight: { backgroundColor: '#F7EEF2', borderColor: '#CFA7B7' },
  statValueHighlight: { color: '#6E2F48' },
  statLabelHighlight: { color: '#6E2F48', fontWeight: '700' },
  transcriptHeader: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between', marginBottom: 4 },
  transcript: { gap: 10, marginTop: 10 },
  line: { borderRadius: 12, padding: 12 },
  lineAria: { backgroundColor: '#F3E8ED', borderLeftColor: '#87566A', borderLeftWidth: 3 },
  lineUser: { backgroundColor: '#F4F0EA', borderLeftColor: '#B9AA99', borderLeftWidth: 3 },
  lineLabel: { color: '#A69C92', fontSize: 11, fontWeight: '700', marginBottom: 4 },
  lineText: { color: '#2B2522', fontSize: 14, lineHeight: 20 },
  lineTime: { color: '#C4B9AF', fontSize: 11, marginTop: 4, textAlign: 'right' },
  emptyTranscript: { color: '#A69C92', fontSize: 14, paddingVertical: 16, textAlign: 'center' },
});
