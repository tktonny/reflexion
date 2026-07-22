import { useQuery } from '@tanstack/react-query';
import React, { useCallback, useMemo, useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useLocalSearchParams } from 'expo-router';
import { Feather } from '@expo/vector-icons';
import StatusBadge from '../../src/components/StatusBadge';
import { apiGet } from '../../src/lib/apiClient';

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
  const shouldLoadRealSession = Boolean(id && /^[0-9a-f]{24}$/i.test(id));
  const today = getSingaporeDateKey(new Date());
  const todaySessionsQuery = useQuery({
    enabled: shouldLoadRealSession,
    queryKey: ['sessionDay', id, today],
    queryFn: async () => {
      const body = await apiGet<Partial<TodaySessionsResponse>>(
        `/api/conversation-sessions-by-day?id=${encodeURIComponent(id)}&date=${encodeURIComponent(today)}`,
      );
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
  const realSession = todaySessions?.sessions[selectedSessionIndex] || null;
  const realTranscript = useMemo(
    () => buildTranscript(realSession?.logs || []),
    [realSession?.logs],
  );

  if (!todaySessions && todaySessionsQuery.isLoading) return (
    <SafeAreaView style={styles.safe}>
      <Text style={styles.notFound}>Loading session...</Text>
    </SafeAreaView>
  );

  if (!todaySessions) return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.placeholder}>
        <Feather name="activity" size={28} color="#87566A" />
        <Text style={styles.placeholderTitle}>Bear with us</Text>
        <Text style={styles.placeholderText}>This session is not ready to show yet.</Text>
      </View>
    </SafeAreaView>
  );

  if (todaySessions) {
    const selectedSession = realSession;

    return (
      <SafeAreaView style={styles.safe}>
        <ScrollView contentContainerStyle={styles.content}>
          {todaySessions.sessions.length > 1 ? (
            <View style={styles.sessionTabs}>
              {todaySessions.sessions.map((item, index) => (
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
          ) : null}

          {!selectedSession ? (
            <View style={styles.card}>
              <Text style={styles.emptyText}>No sessions recorded today.</Text>
            </View>
          ) : (
            <>
          <View style={styles.card}>
            <View style={styles.metaRow}>
              <Text style={styles.metaName}>{todaySessions.patientName}</Text>
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
            <TouchableOpacity style={styles.transcriptHeader} onPress={() => setShowTranscript(v => !v)}>
              <Text style={styles.cardTitle}>View the full conversation with {todaySessions.patientName} below</Text>
              <Feather name={showTranscript ? 'chevron-up' : 'chevron-down'} size={16} color="#87566A" />
            </TouchableOpacity>

            {showTranscript && realTranscript.length > 0 && (
              <View style={styles.transcript}>
                {realTranscript.map((line, i) => (
                  <View key={i} style={[styles.line, line.speaker === 'Aria' ? styles.lineAria : styles.lineUser]}>
                    <Text style={styles.lineLabel}>
                      {line.speaker === 'Aria' ? 'Aria' : todaySessions.patientName}
                    </Text>
                    <Text style={styles.lineText}>{line.text}</Text>
                    <Text style={styles.lineTime}>{formatSeconds(line.timestamp)}</Text>
                  </View>
                ))}
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
  safe: { flex: 1, backgroundColor: '#F8F3EC' },
  notFound: { padding: 30, textAlign: 'center', color: '#A69C92', fontSize: 15 },
  placeholder: {
    alignItems: 'center',
    flex: 1,
    gap: 8,
    justifyContent: 'center',
    paddingHorizontal: 28,
  },
  placeholderTitle: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 24, fontWeight: '500' },
  placeholderText: { color: '#756C64', fontSize: 15, lineHeight: 22, textAlign: 'center' },

  content: { paddingHorizontal: 20, paddingBottom: 48, paddingTop: 16 },

  sessionTabs: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 14,
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
  sessionTabText: {
    color: '#756C64',
    fontSize: 13,
    fontWeight: '700',
  },
  sessionTabTextActive: {
    color: '#FFFFFF',
  },

  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#E7DED2',
    padding: 18,
    marginBottom: 14,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.035,
    shadowRadius: 10,
    elevation: 2,
  },
  metaRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 },
  metaName: { fontSize: 18, fontWeight: '500', color: '#2B2522', fontFamily: 'Georgia' },
  metaDate: { fontSize: 13, color: '#A69C92', marginBottom: 16 },

  statsRow: { flexDirection: 'row', gap: 8 },
  statChip: {
    flex: 1,
    backgroundColor: '#F8F3EC',
    borderRadius: 12,
    padding: 10,
    alignItems: 'center',
    gap: 4,
    borderWidth: 1,
    borderColor: '#E7DED2',
  },
  statValue: { fontSize: 15, fontWeight: '600', color: '#2B2522' },
  statLabel: { fontSize: 10, color: '#A69C92', textAlign: 'center' },
  statChipHighlight: { borderColor: '#CFA7B7', backgroundColor: '#F7EEF2' },
  statValueHighlight: { color: '#6E2F48' },
  statLabelHighlight: { color: '#6E2F48', fontWeight: '700' },

  cardTitle: {
    fontSize: 12,
    fontWeight: '600',
    color: '#A69C92',
    textTransform: 'uppercase',
    letterSpacing: 0.6,
    marginBottom: 10,
  },
  summaryText: { fontSize: 14, color: '#756C64', lineHeight: 21 },
  emptyText: { fontSize: 14, color: '#A69C92', lineHeight: 21 },
  topicRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 12 },
  topicChip: {
    backgroundColor: '#F4F0EA',
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: '#E7DED2',
  },
  topicText: { fontSize: 12, color: '#756C64', fontWeight: '600' },

  transcriptHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 },
  transcript: { gap: 10, marginTop: 10 },
  line: { borderRadius: 12, padding: 12 },
  lineAria: { backgroundColor: '#F3E8ED', borderLeftWidth: 3, borderLeftColor: '#87566A' },
  lineUser: { backgroundColor: '#F4F0EA', borderLeftWidth: 3, borderLeftColor: '#B9AA99' },
  lineLabel: { fontSize: 11, fontWeight: '700', color: '#A69C92', marginBottom: 4 },
  lineText: { fontSize: 14, color: '#2B2522', lineHeight: 20 },
  lineTime: { fontSize: 11, color: '#C4B9AF', marginTop: 4, textAlign: 'right' },
  emptyTranscript: { fontSize: 14, color: '#A69C92', textAlign: 'center', paddingVertical: 16 },
});
