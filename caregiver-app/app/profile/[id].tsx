import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import React, { useCallback, useMemo, useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, Alert, Linking, Image,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import { Feather } from '@expo/vector-icons';
import StatusBadge from '../../src/components/StatusBadge';
import MiniSparkline from '../../src/components/MiniSparkline';
import type { Status } from '../../src/data/mockData';
import { fetchPatientTrend } from '../../src/lib/patientTrendClient';
import { apiSend } from '../../src/lib/apiClient';

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

  const realInitials = useMemo(
    () => (realProfile ? getNameInitials(realProfile.name) : ''),
    [realProfile],
  );

  if (realProfile) {
    const status = getProfileBadgeStatus(realProfile.status);
    const latestSpokenText = formatProfileLastSpoken(realProfile.lastSpokenLabel);
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
              <StatusBadge status={status} label={realProfile.statusLabel} />
              <View style={[styles.avatar, { backgroundColor: AVATAR_BG[status] }]}>
                {realProfile.photoUrl ? (
                  <Image source={{ uri: realProfile.photoUrl }} style={styles.avatarImage} />
                ) : (
                  <Text style={[styles.avatarText, { color: AVATAR_TEXT[status] }]}>{realInitials}</Text>
                )}
              </View>
            </View>
            <Text style={styles.bannerName}>{realProfile.name}</Text>
            <Text style={styles.lastSeen}>{latestSpokenText}</Text>
            <Text style={styles.duration}>Duration: {durationText}</Text>
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

function getProfileBadgeStatus(status: RealPatientProfile['status']): Status {
  if (status === 'doing_well') return 'green';
  if (status === 'worth_checking') return 'yellow';
  return 'red';
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
  lastSeen: { fontSize: 14, color: '#756C64' },
  duration: { fontSize: 13, color: '#A69C92', marginTop: 2 },

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
