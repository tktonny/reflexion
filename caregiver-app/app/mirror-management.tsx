import { Feather } from '@expo/vector-icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useCallback } from 'react';
import {
  ActivityIndicator,
  Alert,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { apiGet, apiSend } from '../src/lib/apiClient';
import { getStoredAuthSession } from '../src/lib/authSession';

type MirrorPatient = {
  patientId: string;
  patientName: string;
  mirrorId: string;
  mirrorName: string;
  mirrorVerified: boolean;
  mirrorPairingStatus: string;
  mirrorPairingCode: string;
  mirrorPairedAt: string | null;
  deviceAuthTokenPresent: boolean;
  timezone: string;
};

export default function MirrorManagementScreen() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const session = getStoredAuthSession();
  const mirrorsQuery = useQuery({
    enabled: Boolean(session?.nurseId),
    queryKey: ['mirrors', session?.nurseId || ''],
    queryFn: async () => {
      const body = await apiGet<{ patients?: MirrorPatient[] }>(
        `/api/nurse-patient-config/mirrors?nurseId=${encodeURIComponent(session?.nurseId || '')}`,
      );
      return Array.isArray(body?.patients) ? body.patients : [];
    },
  });
  const patchMirrorMutation = useMutation({
    mutationFn: (body: { action: 'unlink'; patientId: string }) => apiSend<{
      deletedMirrorMapCount?: number;
      deletedPairingSessionCount?: number;
    }>('/api/nurse-patient-config/mirrors', {
      method: 'PATCH',
      body: JSON.stringify({
        ...body,
        nurseId: session?.nurseId,
      }),
    }),
    onSuccess: async (result) => {
      await queryClient.invalidateQueries({ queryKey: ['mirrors', session?.nurseId || ''] });
      await queryClient.invalidateQueries({ queryKey: ['latestConfig'] });
      showMessage(
        'Mirror unlinked',
        `Removed ${result?.deletedMirrorMapCount ?? 0} mirror map and ${result?.deletedPairingSessionCount ?? 0} pairing session(s). You can add a new connection for this patient.`,
      );
    },
    onError: (err) => {
      showMessage('Unable to update mirror', err instanceof Error ? err.message : 'Please try again.');
    },
  });
  const patients = mirrorsQuery.data || [];
  const savingPatientId = patchMirrorMutation.variables?.patientId || '';
  const { refetch: refetchMirrors } = mirrorsQuery;

  useFocusEffect(
    useCallback(() => {
      if (session?.nurseId) {
        void refetchMirrors();
      }
    }, [refetchMirrors, session?.nurseId]),
  );

  function confirmUnlink(patient: MirrorPatient) {
    const unlink = () => void patchMirror({ action: 'unlink', patientId: patient.patientId });

    if (Platform.OS === 'web') {
      const confirmed = window.confirm(
        `Delete mirror connection?\n\nThis will unlink ${patient.mirrorName || 'the mirror'} from ${patient.patientName}.`,
      );
      if (confirmed) {
        unlink();
      }
      return;
    }

    Alert.alert(
      'Delete mirror connection?',
      `This will unlink ${patient.mirrorName || 'the mirror'} from ${patient.patientName}.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete connection',
          style: 'destructive',
          onPress: unlink,
        },
      ],
    );
  }

  async function patchMirror(body: {
    action: 'unlink';
    patientId: string;
  }) {
    if (!session?.nurseId || patchMirrorMutation.isPending) return;
    patchMirrorMutation.mutate(body);
  }

  function goBack() {
    if (router.canGoBack()) {
      router.back();
      return;
    }

    router.replace('/(tabs)/settings');
  }

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <TouchableOpacity style={styles.backButton} onPress={goBack}>
            <Feather name="chevron-left" size={24} color="#87566A" />
          </TouchableOpacity>
          <View style={styles.headerTextBlock}>
            <Text style={styles.eyebrow}>Voice Companion</Text>
            <Text style={styles.title}>Manage linked mirrors</Text>
          </View>
        </View>

        {mirrorsQuery.isLoading ? (
          <View style={styles.loadingCard}>
            <ActivityIndicator color="#87566A" />
            <Text style={styles.loadingText}>Loading mirror connections.</Text>
          </View>
        ) : (
          patients.map((patient) => {
            const isSavingThisPatient = savingPatientId === patient.patientId;
            const isPaired = patient.mirrorVerified && patient.mirrorPairingStatus === 'paired';
            return (
              <View key={patient.patientId} style={styles.card}>
                <View style={styles.cardHeader}>
                  <View>
                    <Text style={styles.patientName}>{patient.patientName}</Text>
                    <Text style={styles.mirrorName}>{patient.mirrorName || 'No mirror linked'}</Text>
                  </View>
                  <View style={[styles.statusPill, isPaired ? styles.statusPaired : styles.statusUnpaired]}>
                    <Text style={[styles.statusText, isPaired ? styles.statusTextPaired : styles.statusTextUnpaired]}>
                      {isPaired ? 'Paired' : 'Unpaired'}
                    </Text>
                  </View>
                </View>

                <InfoRow label="Mirror ID" value={patient.mirrorId ? compactId(patient.mirrorId) : 'None'} />
                <InfoRow label="Pairing code" value={patient.mirrorPairingCode || 'None'} />
                <InfoRow label="Paired at" value={formatDate(patient.mirrorPairedAt)} />
                <InfoRow label="Device token" value={patient.deviceAuthTokenPresent ? 'Saved' : 'None'} />

                {isPaired ? (
                  <TouchableOpacity
                    disabled={isSavingThisPatient}
                    onPress={() => confirmUnlink(patient)}
                    style={[styles.deleteButton, isSavingThisPatient && styles.disabledOutlineButton]}
                  >
                    {isSavingThisPatient ? (
                      <ActivityIndicator color="#B45F56" />
                    ) : (
                      <Feather name="trash-2" size={17} color="#B45F56" />
                    )}
                    <Text style={styles.deleteButtonText}>
                      {isSavingThisPatient ? 'Deleting connection...' : 'Delete connection'}
                    </Text>
                  </TouchableOpacity>
                ) : (
                  <TouchableOpacity
                    onPress={() => router.push(`/mirror-management/add?patientId=${patient.patientId}`)}
                    style={styles.primaryButton}
                  >
                    <Feather name="plus" size={18} color="#FFFFFF" />
                    <Text style={styles.primaryButtonText}>Add connection</Text>
                  </TouchableOpacity>
                )}
              </View>
            );
          })
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function showMessage(title: string, message: string) {
  if (Platform.OS === 'web') {
    window.alert(`${title}\n\n${message}`);
    return;
  }

  Alert.alert(title, message);
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.infoRow}>
      <Text style={styles.infoLabel}>{label}</Text>
      <Text style={styles.infoValue}>{value}</Text>
    </View>
  );
}

function compactId(value: string) {
  return value.length > 16 ? `${value.slice(0, 8)}...${value.slice(-5)}` : value;
}

function formatDate(value: string | null) {
  if (!value) return 'None';
  return new Intl.DateTimeFormat('en-SG', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value));
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F8F3EC' },
  scroll: { flex: 1 },
  content: { gap: 16, padding: 20, paddingBottom: 52 },
  header: { alignItems: 'center', flexDirection: 'row', gap: 12, marginBottom: 4 },
  backButton: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 22,
    borderWidth: 1,
    height: 44,
    justifyContent: 'center',
    width: 44,
  },
  headerTextBlock: { flex: 1 },
  eyebrow: {
    color: '#A69C92',
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 0.8,
    textTransform: 'uppercase',
  },
  title: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 28, fontWeight: '500', marginTop: 4 },
  loadingCard: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 16,
    borderWidth: 1,
    gap: 10,
    padding: 28,
  },
  loadingText: { color: '#756C64', fontSize: 15 },
  card: {
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 18,
    borderWidth: 1,
    gap: 12,
    padding: 18,
    shadowColor: '#6E5B4B',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.08,
    shadowRadius: 18,
  },
  cardHeader: { alignItems: 'flex-start', flexDirection: 'row', gap: 12, justifyContent: 'space-between' },
  patientName: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 24, fontWeight: '500' },
  mirrorName: { color: '#756C64', fontSize: 15, marginTop: 4 },
  statusPill: { borderRadius: 999, borderWidth: 1, paddingHorizontal: 12, paddingVertical: 6 },
  statusPaired: { backgroundColor: '#F1F7ED', borderColor: '#ABC5A1' },
  statusUnpaired: { backgroundColor: '#F8F3EC', borderColor: '#D8CFC3' },
  statusText: { fontSize: 12, fontWeight: '700' },
  statusTextPaired: { color: '#617A58' },
  statusTextUnpaired: { color: '#8E7F6D' },
  infoRow: {
    alignItems: 'center',
    borderTopColor: '#F3EDE6',
    borderTopWidth: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingTop: 10,
  },
  infoLabel: { color: '#8E7F6D', fontSize: 13, fontWeight: '600' },
  infoValue: { color: '#2B2522', flex: 1, fontSize: 14, textAlign: 'right' },
  primaryButton: {
    alignItems: 'center',
    backgroundColor: '#87566A',
    borderRadius: 12,
    flexDirection: 'row',
    gap: 8,
    justifyContent: 'center',
    minHeight: 48,
  },
  primaryButtonText: { color: '#FFFFFF', fontSize: 15, fontWeight: '800' },
  deleteButton: {
    alignItems: 'center',
    borderColor: '#E6C8C4',
    borderRadius: 12,
    borderWidth: 1,
    flexDirection: 'row',
    gap: 8,
    justifyContent: 'center',
    minHeight: 46,
  },
  disabledOutlineButton: { borderColor: '#E7DED2', opacity: 0.75 },
  deleteButtonText: { color: '#B45F56', fontSize: 15, fontWeight: '700' },
});
