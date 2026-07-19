import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Feather } from '@expo/vector-icons';
import { getApiUrl } from '../../src/lib/apiUrl';
import { getStoredAuthSession } from '../../src/lib/authSession';

const STATUS_DOT: Record<string, string> = {
  green: '#66735D',
  yellow: '#B2844B',
  red: '#87566A',
};

type PatientStatus = 'doing_well' | 'worth_checking' | 'needs_attention';

const STATUS_COPY: Record<PatientStatus, { color: string; label: string }> = {
  doing_well: { color: STATUS_DOT.green, label: 'Doing well' },
  worth_checking: { color: STATUS_DOT.yellow, label: 'Worth checking' },
  needs_attention: { color: STATUS_DOT.red, label: 'Needs attention' },
};

type DashboardPatient = {
  id: string;
  patientId: string;
  name: string;
  phoneNumber: string;
  age: number;
  mirrorName: string;
  photoUrl?: string;
  status: PatientStatus;
  statusLabel: string;
  lastSpokenAt: string | null;
  lastSpokenLabel: string;
  duration: number;
};

type LatestConfigResponse = {
  caregiverName?: string;
  patients?: DashboardPatient[];
  error?: string;
};

export default function HomeScreen() {
  const router = useRouter();
  const [configuredPatients, setConfiguredPatients] = useState<DashboardPatient[]>([]);
  const [caregiverName, setCaregiverName] = useState('');
  const [isLoadingConfig, setIsLoadingConfig] = useState(true);
  const today = new Date().toLocaleDateString('en-SG', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });
  const displayName = getFirstName(caregiverName) || 'there';

  const doingWell = configuredPatients.filter((patient) => patient.status === 'doing_well').length;
  const checkIn = configuredPatients.filter((patient) => patient.status === 'worth_checking').length;
  const attention = configuredPatients.filter((patient) => patient.status === 'needs_attention').length;

  useEffect(() => {
    let isMounted = true;

    const loadLatestConfig = async () => {
      setIsLoadingConfig(true);
      try {
        const session = getStoredAuthSession();
        const query = session?.nurseId ? `?nurseId=${encodeURIComponent(session.nurseId)}` : '';
        const url = getApiUrl(`/api/nurse-patient-config/latest${query}`);
        const response = await fetch(url);
        const body = await readJsonResponse<LatestConfigResponse>(response, url);

        if (!response.ok) {
          throw new Error(body?.error || 'Unable to load configured patients.');
        }

        if (isMounted) {
          setConfiguredPatients(Array.isArray(body?.patients) ? body.patients : []);
          setCaregiverName(typeof body?.caregiverName === 'string' ? body.caregiverName : '');
        }
      } catch (err) {
        console.warn('[HomeScreen] load configured patients failed', err);
        if (isMounted) {
          setConfiguredPatients([]);
          setCaregiverName('');
        }
      } finally {
        if (isMounted) {
          setIsLoadingConfig(false);
        }
      }
    };

    void loadLatestConfig();
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
        {/* Header */}
        <View style={styles.header}>
          <View>
            <Text style={styles.greeting}>Good morning, {displayName}</Text>
            <Text style={styles.date}>{today}</Text>
          </View>
          <TouchableOpacity onPress={() => router.push('/onboarding?mode=add-patient')} style={styles.addBtn}>
            <Feather name="plus" size={16} color="#FFFFFF" />
          </TouchableOpacity>
        </View>

        {/* Status Summary Strip */}
        <View style={styles.summaryStrip}>
          <SummaryChip dot={STATUS_DOT.green} count={doingWell} label="Doing well" />
          <View style={styles.divider} />
          <SummaryChip dot={STATUS_DOT.yellow} count={checkIn} label="Worth checking" />
          <View style={styles.divider} />
          <SummaryChip dot={STATUS_DOT.red} count={attention} label="Needs attention" />
        </View>

        {/* Loved One Cards */}
        <Text style={styles.sectionTitle}>Your loved ones</Text>
        {isLoadingConfig ? (
          <LoadingCard />
        ) : configuredPatients.length ? (
          configuredPatients.map((patient) => {
            return (
              <TouchableOpacity
                activeOpacity={0.8}
                key={patient.id}
                onPress={() =>
                  router.push({
                    pathname: '/profile/[id]',
                    params: {
                      id: patient.patientId || patient.id,
                      patient: JSON.stringify(toProfileRoutePatient(patient)),
                    },
                  })
                }
                style={styles.patientCard}
              >
                <View style={styles.patientAvatar}>
                  <Text style={styles.patientAvatarText}>{getInitials(patient.name)}</Text>
                </View>
                <View style={styles.patientInfo}>
                  <Text style={styles.patientName}>{patient.name}</Text>
                  <View style={styles.patientStatusRow}>
                    <View style={[styles.patientStatusDot, { backgroundColor: STATUS_COPY[patient.status].color }]} />
                    <Text style={styles.patientStatusText}>{patient.statusLabel}</Text>
                  </View>
                  <Text style={styles.patientMeta}>{patient.lastSpokenLabel}</Text>
                </View>
                <Text style={styles.chevron}>›</Text>
              </TouchableOpacity>
            );
          })
        ) : (
          <EmptyCard />
        )}

        {/* Quick Links */}
        <Text style={styles.sectionTitle}>Quick links</Text>
        <View style={styles.quickGrid}>
          <QuickLink icon="book-open" label="Guide" sub="Tips and resources" onPress={() => router.push('/faq')} />
          <QuickLink icon="headphones" label="Support" sub="Get help anytime" onPress={() => router.push('/chatbot')} />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function LoadingCard() {
  return (
    <View style={styles.stateCard}>
      <ActivityIndicator color="#87566A" />
      <Text style={styles.stateTitle}>Bear with us</Text>
      <Text style={styles.stateText}>We are loading your loved ones.</Text>
    </View>
  );
}

function EmptyCard() {
  return (
    <View style={styles.stateCard}>
      <Feather name="user-plus" size={24} color="#87566A" />
      <Text style={styles.stateTitle}>Bear with us</Text>
      <Text style={styles.stateText}>No loved one profiles are ready to show yet.</Text>
    </View>
  );
}

async function readJsonResponse<T>(response: Response, url: string): Promise<T> {
  const text = await response.text();

  try {
    return text ? JSON.parse(text) as T : {} as T;
  } catch {
    const preview = text.replace(/\s+/g, ' ').trim().slice(0, 120);
    throw new Error(
      `Expected JSON from ${url}, received ${response.status} ${response.statusText}: ${preview}`,
    );
  }
}

function getFirstName(name: string) {
  return name.trim().split(/\s+/)[0] || '';
}

function getInitials(name: string) {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return '?';
  return parts.slice(0, 2).map((part) => part[0]?.toUpperCase()).join('');
}

function toProfileRoutePatient(patient: DashboardPatient) {
  return {
    name: patient.name,
    phoneNumber: patient.phoneNumber,
    status: patient.status,
    statusLabel: patient.statusLabel,
    lastSpokenAt: patient.lastSpokenAt,
    lastSpokenLabel: patient.lastSpokenLabel,
    duration: patient.duration,
  };
}

function SummaryChip({ dot, count, label }: { dot: string; count: number; label: string }) {
  return (
    <View style={styles.chip}>
      <Text style={styles.chipCount}>{count}</Text>
      <View style={styles.chipRow}>
        <View style={[styles.chipDot, { backgroundColor: dot }]} />
        <Text style={styles.chipLabel}>{label}</Text>
      </View>
    </View>
  );
}

function QuickLink({ icon, label, sub, onPress }: { icon: any; label: string; sub: string; onPress: () => void }) {
  return (
    <TouchableOpacity style={styles.quickLink} onPress={onPress} activeOpacity={0.7}>
      <Feather name={icon} size={22} color="#87566A" />
      <Text style={styles.quickLabel}>{label}</Text>
      <Text style={styles.quickSub}>{sub}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F8F3EC' },
  scroll: { flex: 1 },
  content: { paddingHorizontal: 20, paddingTop: 20, paddingBottom: 48 },

  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 22,
  },
  greeting: { fontSize: 26, fontWeight: '500', color: '#2B2522', fontFamily: 'Georgia' },
  date: { fontSize: 13, color: '#A69C92', marginTop: 3 },
  addBtn: {
    width: 36,
    height: 36,
    borderRadius: 999,
    backgroundColor: '#87566A',
    alignItems: 'center',
    justifyContent: 'center',
  },

  summaryStrip: {
    flexDirection: 'row',
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#E7DED2',
    padding: 20,
    marginBottom: 28,
    justifyContent: 'space-around',
    alignItems: 'center',
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.035,
    shadowRadius: 10,
    elevation: 2,
  },
  chip: { alignItems: 'center', gap: 6 },
  chipCount: { fontSize: 26, fontWeight: '500', color: '#2B2522', fontFamily: 'Georgia' },
  chipRow: { alignItems: 'center', gap: 6 },
  chipDot: { width: 7, height: 7, borderRadius: 999 },
  chipLabel: { color: '#756C64', fontSize: 12, textAlign: 'center' },
  divider: { width: 1, height: 36, backgroundColor: '#E7DED2' },

  sectionTitle: {
    fontSize: 17,
    fontWeight: '600',
    color: '#2B2522',
    marginBottom: 14,
    marginTop: 4,
  },
  patientCard: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 16,
    borderWidth: 1,
    flexDirection: 'row',
    gap: 12,
    marginBottom: 12,
    padding: 16,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.035,
    shadowRadius: 10,
    elevation: 2,
  },
  patientAvatar: {
    alignItems: 'center',
    backgroundColor: '#F4F0EA',
    borderRadius: 999,
    height: 56,
    justifyContent: 'center',
    width: 56,
  },
  patientAvatarText: {
    color: '#756C64',
    fontFamily: 'Georgia',
    fontSize: 18,
    fontWeight: '500',
  },
  patientInfo: { flex: 1 },
  patientName: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 20, fontWeight: '400' },
  patientStatusRow: { alignItems: 'center', flexDirection: 'row', gap: 7, marginTop: 7 },
  patientStatusDot: { borderRadius: 999, height: 8, width: 8 },
  patientStatusText: { color: '#756C64', fontSize: 13, fontWeight: '400' },
  patientMeta: { color: '#A69C92', fontSize: 13, fontWeight: '700', marginTop: 7 },
  chevron: { fontSize: 20, color: '#C4B9AF', fontWeight: '300' },
  stateCard: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 16,
    borderWidth: 1,
    gap: 8,
    marginBottom: 16,
    padding: 24,
  },
  stateTitle: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 20, fontWeight: '500' },
  stateText: { color: '#756C64', fontSize: 14, lineHeight: 20, textAlign: 'center' },

  quickGrid: { flexDirection: 'row', gap: 12 },
  quickLink: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#E7DED2',
    padding: 18,
    gap: 6,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.035,
    shadowRadius: 10,
    elevation: 2,
  },
  quickLabel: { fontSize: 14, fontWeight: '600', color: '#2B2522' },
  quickSub: { fontSize: 12, color: '#A69C92' },
});
