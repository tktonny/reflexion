import { Feather } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { getApiUrl } from '../../src/lib/apiUrl';
import { getStoredAuthSession } from '../../src/lib/authSession';

type MirrorPatient = {
  patientId: string;
  patientName: string;
  mirrorName: string;
  timezone: string;
};

export default function AddMirrorConnectionScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ patientId?: string }>();
  const session = getStoredAuthSession();
  const [patients, setPatients] = useState<MirrorPatient[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [mirrorName, setMirrorName] = useState('');
  const [pairingCode, setPairingCode] = useState('');
  const [timezone, setTimezone] = useState('Asia/Singapore');

  const patient = useMemo(
    () => patients.find((candidate) => candidate.patientId === params.patientId) || null,
    [params.patientId, patients],
  );

  useEffect(() => {
    void loadPatient();
  }, [session?.nurseId]);

  useEffect(() => {
    if (!patient) return;
    setMirrorName(patient.mirrorName || `Mirror for ${patient.patientName}`);
    setTimezone(patient.timezone || 'Asia/Singapore');
  }, [patient]);

  async function loadPatient() {
    if (!session?.nurseId) {
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(
        getApiUrl(`/api/nurse-patient-config/mirrors?nurseId=${encodeURIComponent(session.nurseId)}`),
      );
      const body = await response.json();
      if (!response.ok) {
        throw new Error(body?.error || 'Unable to load patient.');
      }
      setPatients(Array.isArray(body?.patients) ? body.patients : []);
    } catch (err) {
      Alert.alert('Unable to load patient', err instanceof Error ? err.message : 'Please try again.');
    } finally {
      setIsLoading(false);
    }
  }

  function goBack() {
    if (router.canGoBack()) {
      router.back();
      return;
    }

    router.replace('/mirror-management');
  }

  async function saveConnection() {
    if (!session?.nurseId || !patient || isSaving) return;

    const normalizedPairingCode = pairingCode.replace(/\D/g, '');
    if (normalizedPairingCode.length !== 6) {
      Alert.alert('Pairing code needed', 'Enter the 6 digit code shown on the mirror.');
      return;
    }

    setIsSaving(true);
    try {
      const response = await fetch(getApiUrl('/api/nurse-patient-config/mirrors/connect'), {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          nurseId: session.nurseId,
          patientId: patient.patientId,
          mirrorName: mirrorName.trim() || `Mirror for ${patient.patientName}`,
          pairingCode: normalizedPairingCode,
          timezone: timezone.trim() || 'Asia/Singapore',
        }),
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result?.error || 'Unable to add mirror connection.');
      }

      router.replace('/mirror-management');
    } catch (err) {
      Alert.alert('Unable to connect mirror', err instanceof Error ? err.message : 'Please try again.');
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <TouchableOpacity style={styles.backButton} onPress={goBack}>
            <Feather name="chevron-left" size={24} color="#87566A" />
          </TouchableOpacity>
          <View style={styles.headerTextBlock}>
            <Text style={styles.eyebrow}>Mirror pairing</Text>
            <Text style={styles.title}>Add connection</Text>
          </View>
        </View>

        {isLoading ? (
          <View style={styles.card}>
            <ActivityIndicator color="#87566A" />
            <Text style={styles.loadingText}>Loading pairing details.</Text>
          </View>
        ) : !patient ? (
          <View style={styles.card}>
            <Text style={styles.emptyTitle}>Patient not found</Text>
            <Text style={styles.emptyText}>Go back and choose a patient from Manage linked mirrors.</Text>
          </View>
        ) : (
          <>
            <View style={styles.infoBox}>
              <Text style={styles.infoTitle}>{patient.patientName}</Text>
              <Text style={styles.infoText}>
                On the mirror, open setup and enter the 6-digit pairing code shown there.
              </Text>
            </View>

            <View style={styles.card}>
              <Label>Mirror name</Label>
              <TextInput
                onChangeText={setMirrorName}
                placeholder={`Mirror for ${patient.patientName}`}
                placeholderTextColor="#B7ACA1"
                style={styles.input}
                value={mirrorName}
              />

              <Label>Mirror pairing code</Label>
              <TextInput
                keyboardType="number-pad"
                maxLength={7}
                onChangeText={(value) => setPairingCode(formatPairingInput(value))}
                placeholder="482 913"
                placeholderTextColor="#B7ACA1"
                style={styles.input}
                value={pairingCode}
              />

              <Label>Mirror timezone</Label>
              <TextInput
                autoCapitalize="none"
                onChangeText={setTimezone}
                placeholder="Asia/Singapore"
                placeholderTextColor="#B7ACA1"
                style={styles.input}
                value={timezone}
              />

              <TouchableOpacity
                onPress={() =>
                  Alert.alert(
                    'Pairing instructions',
                    'Enter the code displayed on the mirror, or scan the mirror QR in the caregiver app once scanner support is enabled.',
                  )
                }
                style={styles.secondaryButton}
              >
                <Feather name="help-circle" size={17} color="#87566A" />
                <Text style={styles.secondaryButtonText}>How pairing works</Text>
              </TouchableOpacity>

              <TouchableOpacity
                disabled={isSaving}
                onPress={() => void saveConnection()}
                style={[styles.primaryButton, isSaving && styles.disabledButton]}
              >
                {isSaving ? (
                  <ActivityIndicator color="#FFFFFF" />
                ) : (
                  <>
                    <Feather name="link" size={18} color="#FFFFFF" />
                    <Text style={styles.primaryButtonText}>Add connection</Text>
                  </>
                )}
              </TouchableOpacity>
            </View>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <Text style={styles.label}>{children}</Text>;
}

function formatPairingInput(value: string) {
  return value.replace(/\D/g, '').slice(0, 6).replace(/(\d{3})(\d{1,3})/, '$1 $2');
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
  card: {
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 18,
    borderWidth: 1,
    gap: 12,
    padding: 18,
  },
  infoBox: {
    backgroundColor: '#FFF6EA',
    borderColor: '#E7DED2',
    borderRadius: 18,
    borderWidth: 1,
    gap: 8,
    padding: 18,
  },
  infoTitle: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 24, fontWeight: '500' },
  infoText: { color: '#756C64', fontSize: 15, lineHeight: 22 },
  loadingText: { color: '#756C64', fontSize: 15, textAlign: 'center' },
  emptyTitle: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 22, fontWeight: '500' },
  emptyText: { color: '#756C64', fontSize: 15, lineHeight: 22 },
  label: { color: '#756C64', fontSize: 13, fontWeight: '700', marginTop: 4 },
  input: {
    backgroundColor: '#FFFDF8',
    borderColor: '#D8CFC3',
    borderRadius: 12,
    borderWidth: 1,
    color: '#2B2522',
    fontSize: 18,
    fontWeight: '600',
    paddingHorizontal: 14,
    paddingVertical: 13,
  },
  primaryButton: {
    alignItems: 'center',
    backgroundColor: '#87566A',
    borderRadius: 12,
    flexDirection: 'row',
    gap: 8,
    justifyContent: 'center',
    minHeight: 48,
    marginTop: 8,
  },
  primaryButtonText: { color: '#FFFFFF', fontSize: 15, fontWeight: '800' },
  secondaryButton: {
    alignItems: 'center',
    borderColor: '#D8CFC3',
    borderRadius: 12,
    borderWidth: 1,
    flexDirection: 'row',
    gap: 8,
    justifyContent: 'center',
    minHeight: 46,
  },
  secondaryButtonText: { color: '#87566A', fontSize: 15, fontWeight: '700' },
  disabledButton: { opacity: 0.7 },
});
