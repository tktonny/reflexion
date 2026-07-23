import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import React, { useCallback, useMemo, useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, TextInput, ActivityIndicator, Alert, Linking, Image,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import { Feather } from '@expo/vector-icons';
import MiniSparkline from '../../src/components/MiniSparkline';
import type { Status } from '../../src/data/mockData';
import { fetchPatientTrend } from '../../src/lib/patientTrendClient';
import { apiSend } from '../../src/lib/apiClient';
import {
  createAwayPeriodV1,
  createManualFlagV1,
  patientStatusQueryKey,
  usePatientStatusV1,
} from '../../src/lib/v1Client';
import {
  STATUS_META,
  NEUTRAL_STATUS_COLOR,
  getStatusLabel,
  getReasonText,
  getBaselineProgressText,
  getTechnicalNote,
  formatLastInteraction,
} from '../../src/lib/v1Status';

const AVATAR_BG: Record<Status, string> = {
  green: '#F0F3ED',
  yellow: '#F6EFE5',
  red: '#F3E8ED',
};

const AVATAR_TEXT: Record<Status, string> = {
  green: '#4A5745',
  yellow: '#7A5C30',
  red: '#6B3D50',
};

type RealPatientProfile = {
  name: string;
  phoneNumber: string;
  photoUrl?: string;
  status: 'doing_well' | 'worth_checking' | 'needs_attention';
  statusLabel: string;
  lastSpokenAt: string | null;
  lastSpokenLabel: string;
  duration: number;
};

type RealTrendDay = {
  date: string;
  duration: number;
  status: Status;
  missed: boolean;
};

export default function ProfileScreen() {
  const { id, patient } = useLocalSearchParams<{ id: string; patient?: string }>();
  const router = useRouter();
  const queryClient = useQueryClient();
  const [generatedSummary, setGeneratedSummary] = useState('');
  const shouldLoadRealProfile = Boolean(id && !id.startsWith('el-'));
  const realProfile = useMemo(() => parsePatientParam(patient), [patient]);
  const trendQuery = useQuery({
    enabled: shouldLoadRealProfile,
    queryKey: ['patientTrend', id, 7],
    queryFn: () => fetchPatientTrend(id, 7),
  });
  const { refetch: refetchTrend } = trendQuery;
  useFocusEffect(
    useCallback(() => {
      if (shouldLoadRealProfile) {
        void refetchTrend();
      }
    }, [refetchTrend, shouldLoadRealProfile]),
  );
  const realTrend = trendQuery.data || [];
  const summaryMutation = useMutation({
    mutationFn: () => apiSend<{ summary?: string }>('/api/patient-summary', {
      method: 'POST',
      body: JSON.stringify({ patientId: id }),
    }),
    onSuccess: async (body) => {
      setGeneratedSummary(body?.summary || 'No summary generated.');
      await queryClient.invalidateQueries({ queryKey: ['sessionDay', id] });
    },
    onError: (err) => {
      Alert.alert(
        'Unable to generate summary',
        err instanceof Error ? err.message : 'Unable to generate summary.',
      );
    },
  });

  // Authoritative status from the v1 read model (baseline §4). The route id equals the v1 patient _id
  // (migration reuses the legacy ObjectId hex), so this is the same id the trend/session screens use.
  const statusQuery = usePatientStatusV1(shouldLoadRealProfile ? id : null);
  const v1Status = statusQuery.data;

  const [activeForm, setActiveForm] = useState<'none' | 'flag' | 'away'>('none');
  const [flagSeverity, setFlagSeverity] = useState<'worth_checking' | 'needs_attention'>('worth_checking');
  const [flagReason, setFlagReason] = useState('');
  const deviceTimezone = useMemo(() => {
    try {
      return Intl.DateTimeFormat().resolvedOptions().timeZone || 'Asia/Singapore';
    } catch {
      return 'Asia/Singapore';
    }
  }, []);
  const [awayStart, setAwayStart] = useState('');
  const [awayEnd, setAwayEnd] = useState('');
  const [awayTimezone, setAwayTimezone] = useState(deviceTimezone);
  const [awayReason, setAwayReason] = useState('');

  const flagMutation = useMutation({
    mutationFn: () => createManualFlagV1(id, flagSeverity, flagReason.trim()),
    onSuccess: async () => {
      setFlagReason('');
      setActiveForm('none');
      await queryClient.invalidateQueries({ queryKey: patientStatusQueryKey(id) });
      Alert.alert('Concern flagged', 'Thanks — this has been noted on their status.');
    },
    onError: (err) => {
      Alert.alert('Could not flag', err instanceof Error ? err.message : 'Please try again.');
    },
  });

  const awayMutation = useMutation({
    mutationFn: () => createAwayPeriodV1(id, {
      startsOn: awayStart.trim(),
      endsOn: awayEnd.trim(),
      timezone: awayTimezone.trim(),
      reason: awayReason.trim() || undefined,
    }),
    onSuccess: async () => {
      setAwayStart('');
      setAwayEnd('');
      setAwayReason('');
      setActiveForm('none');
      await queryClient.invalidateQueries({ queryKey: patientStatusQueryKey(id) });
      Alert.alert('Marked as away', 'These days will not count against their check-in streak.');
    },
    onError: (err) => {
      Alert.alert('Could not save', err instanceof Error ? err.message : 'Please try again.');
    },
  });

  function submitFlag() {
    if (!flagReason.trim()) {
      Alert.alert('Add a note', 'Please add a short reason before flagging.');
      return;
    }
    flagMutation.mutate();
  }

  function submitAway() {
    const start = awayStart.trim();
    const end = awayEnd.trim();
    if (!/^\d{4}-\d{2}-\d{2}$/.test(start) || !/^\d{4}-\d{2}-\d{2}$/.test(end)) {
      Alert.alert('Check the dates', 'Please enter both dates as YYYY-MM-DD.');
      return;
    }
    if (end < start) {
      Alert.alert('Check the dates', 'The end date must be on or after the start date.');
      return;
    }
    if (!awayTimezone.trim()) {
      Alert.alert('Add a timezone', 'Please provide a timezone (e.g. Asia/Singapore).');
      return;
    }
    awayMutation.mutate();
  }

  const realInitials = useMemo(
    () => (realProfile ? getNameInitials(realProfile.name) : ''),
    [realProfile],
  );

  if (realProfile) {
    // Avatar tint is soft/decorative and reassurance-first: establishing is treated as calm (never red).
    const avatarStatus: Status = v1Status?.status === 'needs_attention'
      ? 'red'
      : v1Status?.status === 'worth_checking'
        ? 'yellow'
        : 'green';
    const pillColor = v1Status ? STATUS_META[v1Status.status].dot : NEUTRAL_STATUS_COLOR;
    const pillLabel = v1Status
      ? getStatusLabel(v1Status.status, realProfile.name)
      : statusQuery.isLoading
        ? 'Checking in…'
        : 'Status updating';
    const reasonLine = v1Status
      ? v1Status.status === 'establishing'
        ? getBaselineProgressText(v1Status.baselineProgress)
        : getReasonText(v1Status.primaryReason, realProfile.name)
      : '';
    const technicalNote = v1Status ? getTechnicalNote(v1Status.technicalState) : null;
    const lastInteractionText = v1Status
      ? formatLastInteraction(v1Status.lastInteractionAt)
      : formatProfileLastSpoken(realProfile.lastSpokenLabel);
    const durationText = formatDuration(realProfile.duration);
    const talkedDays = realTrend.filter((day) => !day.missed).length;
    const avgDuration = talkedDays
      ? Math.round(realTrend.filter((day) => !day.missed).reduce((sum, day) => sum + day.duration, 0) / talkedDays)
      : 0;

    return (
      <SafeAreaView style={styles.safe}>
        <ScrollView contentContainerStyle={styles.content}>
          <View style={styles.banner}>
            <View style={styles.bannerTop}>
              <View style={styles.statusPill}>
                <View style={[styles.statusPillDot, { backgroundColor: pillColor }]} />
                <Text style={styles.statusPillText}>{pillLabel}</Text>
              </View>
              <View style={[styles.avatar, { backgroundColor: AVATAR_BG[avatarStatus] }]}>
                {realProfile.photoUrl ? (
                  <Image source={{ uri: realProfile.photoUrl }} style={styles.avatarImage} />
                ) : (
                  <Text style={[styles.avatarText, { color: AVATAR_TEXT[avatarStatus] }]}>{realInitials}</Text>
                )}
              </View>
            </View>
            <Text style={styles.bannerName}>{realProfile.name}</Text>
            {reasonLine ? <Text style={styles.reasonLine}>{reasonLine}</Text> : null}
            <Text style={styles.lastSeen}>{lastInteractionText}</Text>
            <Text style={styles.duration}>Duration: {durationText}</Text>
            {technicalNote ? (
              <View style={styles.techNote}>
                <Feather name="wifi-off" size={13} color="#8E877C" />
                <Text style={styles.techNoteText}>{technicalNote}</Text>
              </View>
            ) : null}
          </View>

          <TouchableOpacity style={styles.callBtn} onPress={() => callPatient(realProfile)}>
            <Feather name="phone" size={17} color="#FFFFFF" />
            <Text style={styles.callBtnText}>Call {realProfile.name}</Text>
          </TouchableOpacity>

          <View style={styles.card}>
            <View style={styles.summaryHeader}>
              <Text style={styles.cardTitle}>Today's summary</Text>
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
            </View>
            <Text style={generatedSummary ? styles.summaryText : styles.emptyText}>
              {generatedSummary || 'No summary yet.'}
            </Text>
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>This week</Text>
            {realTrend.length > 0 ? (
              <>
                <MiniSparkline data={realTrend} days={7} height={52} />
                <Text style={styles.weekStat}>
                  Talked {talkedDays} of 7 days · Avg {formatDuration(avgDuration)}
                </Text>
              </>
            ) : (
              <View style={styles.emptyChart} />
            )}
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>Caregiver actions</Text>
            <View style={styles.actionRow}>
              <TouchableOpacity
                style={[styles.pillBtn, activeForm === 'flag' && styles.pillBtnActive]}
                onPress={() => setActiveForm(activeForm === 'flag' ? 'none' : 'flag')}
              >
                <Feather name="flag" size={14} color="#9B5F4E" />
                <Text style={styles.pillBtnText}>Flag a concern</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.pillBtn, activeForm === 'away' && styles.pillBtnActive]}
                onPress={() => setActiveForm(activeForm === 'away' ? 'none' : 'away')}
              >
                <Feather name="calendar" size={14} color="#596C56" />
                <Text style={styles.pillBtnText}>Mark as away</Text>
              </TouchableOpacity>
            </View>

            {activeForm === 'flag' ? (
              <View style={styles.form}>
                <Text style={styles.formLabel}>How would you describe it?</Text>
                <View style={styles.segment}>
                  {(['worth_checking', 'needs_attention'] as const).map((option) => (
                    <TouchableOpacity
                      key={option}
                      style={[styles.segmentItem, flagSeverity === option && styles.segmentItemActive]}
                      onPress={() => setFlagSeverity(option)}
                    >
                      <Text style={[styles.segmentText, flagSeverity === option && styles.segmentTextActive]}>
                        {option === 'worth_checking' ? 'Worth checking' : 'Needs attention'}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
                <Text style={styles.formLabel}>What did you notice?</Text>
                <TextInput
                  style={styles.textArea}
                  multiline
                  placeholder="A short note for your own reference"
                  placeholderTextColor="#A69C92"
                  value={flagReason}
                  onChangeText={setFlagReason}
                />
                <TouchableOpacity
                  style={[styles.submitBtn, flagMutation.isPending && styles.submitBtnDisabled]}
                  disabled={flagMutation.isPending}
                  onPress={submitFlag}
                >
                  {flagMutation.isPending ? (
                    <ActivityIndicator color="#FFFFFF" />
                  ) : (
                    <Text style={styles.submitBtnText}>Flag concern</Text>
                  )}
                </TouchableOpacity>
              </View>
            ) : null}

            {activeForm === 'away' ? (
              <View style={styles.form}>
                <Text style={styles.formHint}>Away days will not count against their check-in streak.</Text>
                <Text style={styles.formLabel}>From</Text>
                <TextInput
                  style={styles.input}
                  placeholder="YYYY-MM-DD"
                  placeholderTextColor="#A69C92"
                  autoCapitalize="none"
                  value={awayStart}
                  onChangeText={setAwayStart}
                />
                <Text style={styles.formLabel}>To</Text>
                <TextInput
                  style={styles.input}
                  placeholder="YYYY-MM-DD"
                  placeholderTextColor="#A69C92"
                  autoCapitalize="none"
                  value={awayEnd}
                  onChangeText={setAwayEnd}
                />
                <Text style={styles.formLabel}>Timezone</Text>
                <TextInput
                  style={styles.input}
                  placeholder="Asia/Singapore"
                  placeholderTextColor="#A69C92"
                  autoCapitalize="none"
                  value={awayTimezone}
                  onChangeText={setAwayTimezone}
                />
                <Text style={styles.formLabel}>Reason (optional)</Text>
                <TextInput
                  style={styles.input}
                  placeholder="e.g. Visiting family"
                  placeholderTextColor="#A69C92"
                  value={awayReason}
                  onChangeText={setAwayReason}
                />
                <TouchableOpacity
                  style={[styles.submitBtn, awayMutation.isPending && styles.submitBtnDisabled]}
                  disabled={awayMutation.isPending}
                  onPress={submitAway}
                >
                  {awayMutation.isPending ? (
                    <ActivityIndicator color="#FFFFFF" />
                  ) : (
                    <Text style={styles.submitBtnText}>Save away period</Text>
                  )}
                </TouchableOpacity>
              </View>
            ) : null}
          </View>

          <View style={styles.actionGrid}>
            <ActionCard icon="activity" label="Full session" onPress={() => router.push(`/session/${id}`)} />
            <ActionCard icon="bar-chart-2" label="30-day trend" onPress={() => router.push(`/trend/${id}`)} />
            <ActionCard icon="calendar" label="Session history" onPress={() => router.push(`/session-history/${id}`)} />
          </View>
        </ScrollView>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.placeholder}>
        <Feather name="user" size={28} color="#87566A" />
        <Text style={styles.placeholderTitle}>Bear with us</Text>
        <Text style={styles.placeholderText}>This profile is not ready to show yet.</Text>
      </View>
    </SafeAreaView>
  );
}

function ActionCard({ icon, label, onPress }: { icon: any; label: string; onPress: () => void }) {
  return (
    <TouchableOpacity style={styles.actionCard} onPress={onPress} activeOpacity={0.75}>
      <Feather name={icon} size={20} color="#87566A" />
      <Text style={styles.actionLabel}>{label}</Text>
    </TouchableOpacity>
  );
}

function getNameInitials(name: string) {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  return parts.slice(0, 2).map((part) => part[0]?.toUpperCase()).join('') || '?';
}

function parsePatientParam(value?: string): RealPatientProfile | null {
  if (!value) return null;

  try {
    const parsed = JSON.parse(value) as Partial<RealPatientProfile>;
    if (!parsed.name) return null;

    return {
      name: parsed.name,
      phoneNumber: parsed.phoneNumber || '',
      photoUrl: parsed.photoUrl || '',
      status: parsed.status || 'needs_attention',
      statusLabel: parsed.statusLabel || 'Needs attention',
      lastSpokenAt: parsed.lastSpokenAt || null,
      lastSpokenLabel: parsed.lastSpokenLabel || 'No interaction yet',
      duration: Number(parsed.duration || 0),
    };
  } catch {
    return null;
  }
}

async function callPatient(profile: RealPatientProfile) {
  if (!profile.phoneNumber.trim()) {
    Alert.alert('No phone number', `${profile.name} does not have a phone number saved.`);
    return;
  }

  const phoneNumber = profile.phoneNumber.replace(/[^\d+]/g, '');
  try {
    await Linking.openURL(`tel:${phoneNumber}`);
  } catch {
    Alert.alert('Unable to call', `Could not open the phone app for ${profile.phoneNumber}.`);
  }
}

function formatDuration(seconds: number) {
  if (!seconds) {
    return '0m 0s';
  }

  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function formatProfileLastSpoken(value: string) {
  if (!value || value === 'No interaction yet') {
    return 'Last spoken: no conversation yet';
  }

  if (value.startsWith('Today,')) return value.replace('Today,', 'Last spoke today,');
  if (value.startsWith('Yesterday,')) return value.replace('Yesterday,', 'Last spoke yesterday,');

  return value;
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F8F3EC' },
  notFound: { padding: 30, fontSize: 16, color: '#A69C92', textAlign: 'center' },
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

  banner: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#E7DED2',
    padding: 20,
    marginBottom: 14,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.035,
    shadowRadius: 10,
    elevation: 2,
  },
  bannerTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  avatar: {
    width: 52,
    height: 52,
    borderRadius: 999,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  avatarImage: { height: '100%', width: '100%' },
  avatarText: { fontSize: 18, fontWeight: '500', fontFamily: 'Georgia' },
  bannerName: { fontSize: 22, fontWeight: '500', color: '#2B2522', fontFamily: 'Georgia', marginBottom: 4 },
  reasonLine: { fontSize: 14, color: '#4A433C', lineHeight: 20, marginBottom: 4 },
  lastSeen: { fontSize: 14, color: '#756C64' },
  duration: { fontSize: 13, color: '#A69C92', marginTop: 2 },

  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: '#F4F0EA',
    borderWidth: 1,
    borderColor: '#E7DED2',
  },
  statusPillDot: { width: 9, height: 9, borderRadius: 999 },
  statusPillText: { fontSize: 13, fontWeight: '600', color: '#4A433C' },
  techNote: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 7,
    marginTop: 12,
    padding: 10,
    borderRadius: 10,
    backgroundColor: '#F4F0EA',
  },
  techNoteText: { flex: 1, fontSize: 12, color: '#6E6459', lineHeight: 17 },

  actionRow: { flexDirection: 'row', gap: 10, flexWrap: 'wrap' },
  pillBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 999,
    backgroundColor: '#F4F0EA',
    borderWidth: 1,
    borderColor: '#E7DED2',
  },
  pillBtnActive: { borderColor: '#C4B9AF', backgroundColor: '#EFE9E0' },
  pillBtnText: { fontSize: 13, fontWeight: '600', color: '#4A433C' },
  form: { marginTop: 16, gap: 8 },
  formLabel: { fontSize: 13, fontWeight: '600', color: '#756C64', marginTop: 6 },
  formHint: { fontSize: 13, color: '#8E877C', lineHeight: 18 },
  segment: { flexDirection: 'row', gap: 8 },
  segmentItem: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 10,
    borderRadius: 10,
    backgroundColor: '#F4F0EA',
    borderWidth: 1,
    borderColor: '#E7DED2',
  },
  segmentItemActive: { backgroundColor: '#EFE9E0', borderColor: '#C4B9AF' },
  segmentText: { fontSize: 13, fontWeight: '600', color: '#756C64' },
  segmentTextActive: { color: '#2B2522' },
  input: {
    backgroundColor: '#FBF8F4',
    borderColor: '#E7DED2',
    borderRadius: 10,
    borderWidth: 1,
    color: '#2B2522',
    fontSize: 15,
    paddingHorizontal: 12,
    paddingVertical: 11,
  },
  textArea: {
    backgroundColor: '#FBF8F4',
    borderColor: '#E7DED2',
    borderRadius: 10,
    borderWidth: 1,
    color: '#2B2522',
    fontSize: 15,
    minHeight: 72,
    paddingHorizontal: 12,
    paddingVertical: 11,
    textAlignVertical: 'top',
  },
  submitBtn: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#87566A',
    borderRadius: 12,
    paddingVertical: 13,
    marginTop: 12,
    minHeight: 46,
  },
  submitBtnDisabled: { opacity: 0.65 },
  submitBtnText: { color: '#FFFFFF', fontSize: 15, fontWeight: '600' },

  callBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#87566A',
    borderRadius: 12,
    paddingVertical: 15,
    marginBottom: 14,
  },
  callBtnText: { color: '#FFFFFF', fontSize: 15, fontWeight: '600' },

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
  cardTitle: { fontSize: 13, fontWeight: '600', color: '#A69C92', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 10 },
  summaryHeader: {
    alignItems: 'center',
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
    paddingHorizontal: 10,
    paddingVertical: 7,
  },
  summaryBtnDisabled: {
    opacity: 0.65,
  },
  summaryBtnText: {
    color: '#87566A',
    fontSize: 12,
    fontWeight: '700',
    flexShrink: 1,
  },
  summaryText: { fontSize: 14, color: '#756C64', lineHeight: 21 },
  emptyText: { fontSize: 14, color: '#A69C92', lineHeight: 21 },
  emptyChart: {
    backgroundColor: '#F4F0EA',
    borderRadius: 8,
    height: 52,
  },
  topicRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 14 },
  topicChip: {
    backgroundColor: '#F4F0EA',
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: '#E7DED2',
  },
  topicText: { fontSize: 12, color: '#756C64', fontWeight: '600' },
  weekStat: { fontSize: 13, color: '#756C64', marginTop: 12 },
  trendPill: {
    alignSelf: 'flex-start',
    marginTop: 8,
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 999,
    backgroundColor: '#F4F0EA',
  },
  trendText: { fontSize: 12, fontWeight: '600', color: '#66735D' },

  actionGrid: { flexDirection: 'row', gap: 10 },
  actionCard: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#E7DED2',
    padding: 16,
    alignItems: 'center',
    gap: 8,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.035,
    shadowRadius: 10,
    elevation: 2,
  },
  actionLabel: { fontSize: 12, color: '#756C64', fontWeight: '600', textAlign: 'center' },
});
