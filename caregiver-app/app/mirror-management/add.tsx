import { Feather } from '@expo/vector-icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Modal,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, useCameraPermissions } from 'expo-camera';

import { EmptyState, ErrorState, LoadingState } from '../../src/components/ScreenState';
import { apiGet, apiSend } from '../../src/lib/apiClient';
import { getStoredAuthSession } from '../../src/lib/authSession';
import { invalidateCaregiverConfig } from '../../src/lib/queryKeys';
import { colors, fontFamily, fontSize, MIN_TOUCH_TARGET, radius, scaleSize, spacing } from '../../src/theme';

type MirrorPatient = {
  patientId: string;
  patientName: string;
  mirrorName: string;
  timezone: string;
};

// Four different situations, four different things to say. This screen used to show one loading card and
// then the headline "Patient not found", so a failed request read as news about the person instead of a
// connection problem — and a signed-out session looked the same as an empty list.
type PairingState = 'loading' | 'signed-out' | 'failed' | 'no-match' | 'ready';

export default function AddMirrorConnectionScreen() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const params = useLocalSearchParams<{ patientId?: string }>();
  const session = getStoredAuthSession();
  const [mirrorName, setMirrorName] = useState('');
  const [pairingCode, setPairingCode] = useState('');
  const [timezone, setTimezone] = useState('Asia/Singapore');
  const [scanning, setScanning] = useState(false);
  const [cameraNotice, setCameraNotice] = useState('');
  const [permission, requestPermission] = useCameraPermissions();
  const mirrorsQuery = useQuery({
    enabled: Boolean(session?.userId),
    queryKey: ['mirrors', session?.userId || ''],
    queryFn: async () => {
      const body = await apiGet<{ patients?: MirrorPatient[] }>(
        `/api/nurse-patient-config/mirrors?nurseId=${encodeURIComponent(session?.userId || '')}`,
      );
      return Array.isArray(body?.patients) ? body.patients : [];
    },
  });
  const { refetch: refetchMirrors } = mirrorsQuery;
  useFocusEffect(
    useCallback(() => {
      if (session?.userId) {
        void refetchMirrors();
      }
    }, [refetchMirrors, session?.userId]),
  );
  const connectMirrorMutation = useMutation({
    mutationFn: (body: unknown) => apiSend('/api/nurse-patient-config/mirrors/connect', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['mirrors', session?.userId || ''] });
      await invalidateCaregiverConfig(queryClient);
      router.replace('/mirror-management');
    },
    onError: (err) => {
      Alert.alert('Unable to connect mirror', err instanceof Error ? err.message : 'Please try again.');
    },
  });
  const patients = mirrorsQuery.data || [];

  const patient = useMemo(
    () => patients.find((candidate) => candidate.patientId === params.patientId) || null,
    [params.patientId, patients],
  );

  // Usable cached data outranks a stale error. This list shares its cache key with the mirror list this
  // screen is opened from, and it refetches on every focus, so a single failed background refetch would
  // otherwise replace the whole pairing form — mirror name, the code the caregiver has already typed, the
  // Scan QR button — with an error card, even though everything needed to pair is right there and the
  // POST would still succeed.
  const pairingState: PairingState = !session?.userId
    ? 'signed-out'
    : patient
      ? 'ready'
      : mirrorsQuery.isLoading
        ? 'loading'
        : mirrorsQuery.error
          ? 'failed'
          : 'no-match';

  useEffect(() => {
    if (!patient) return;
    setMirrorName(patient.mirrorName || `Mirror for ${patient.patientName}`);
    setTimezone(patient.timezone || 'Asia/Singapore');
  }, [patient]);

  function goBack() {
    if (router.canGoBack()) {
      router.back();
      return;
    }

    router.replace('/mirror-management');
  }

  async function saveConnection() {
    if (!session?.userId || !patient || connectMirrorMutation.isPending) return;

    const normalizedPairingCode = pairingCode.replace(/\D/g, '');
    if (normalizedPairingCode.length !== 6) {
      Alert.alert('Pairing code needed', 'Enter the 6 digit code shown on the mirror.');
      return;
    }

    connectMirrorMutation.mutate({
      nurseId: session.userId,
      patientId: patient.patientId,
      mirrorName: mirrorName.trim() || `Mirror for ${patient.patientName}`,
      pairingCode: normalizedPairingCode,
      timezone: timezone.trim() || 'Asia/Singapore',
    });
  }

  // The mirror shows a QR of { type: 'reflexion_device_pairing_v2', pairingId, pairingCode }.
  function extractPairingCode(data: string): string {
    const raw = data.trim();
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object') {
        return String((parsed as Record<string, unknown>).pairingCode || (parsed as Record<string, unknown>).code || '')
          .replace(/\D/g, '').slice(0, 6);
      }
    } catch {}
    return raw.replace(/\D/g, '').slice(0, 6);
  }

  async function openScanner() {
    setCameraNotice('');
    if (!permission?.granted) {
      const res = await requestPermission();
      if (!res.granted) {
        // A declined camera prompt left the screen unchanged behind a one-line alert, so the caregiver was
        // stuck with no visible way forward. Say it calmly on the screen and point at the code they can type.
        setCameraNotice(
          res.canAskAgain
            ? 'Scanning needs the camera, and it is not open yet. You can type the 6-digit code shown on the mirror instead.'
            : 'Camera access for Reflexion is turned off in your phone settings. You can type the 6-digit code shown on the mirror instead.',
        );
        return;
      }
    }
    setScanning(true);
  }

  function onScan(result: { data: string }) {
    if (!scanning) return;
    setScanning(false);
    const code = extractPairingCode(result.data);
    if (code.length !== 6) {
      // Told on the screen rather than in an alert: this fires in the same tick as the scanner modal
      // dismissing, and an alert raised mid-dismiss can be swallowed, leaving an unreadable QR silent.
      setCameraNotice('That QR did not have a 6-digit pairing code in it. Try again, or type the code shown on the mirror.');
      return;
    }
    setCameraNotice('');
    setPairingCode(formatPairingInput(code));
  }

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <TouchableOpacity
            accessibilityLabel="Go back"
            accessibilityRole="button"
            onPress={goBack}
            style={styles.backButton}
          >
            <Feather
              accessibilityElementsHidden
              importantForAccessibility="no"
              name="chevron-left"
              size={24}
              color={colors.accent}
            />
          </TouchableOpacity>
          <View style={styles.headerTextBlock}>
            <Text style={styles.eyebrow}>Mirror pairing</Text>
            <Text maxFontSizeMultiplier={1.3} style={styles.title}>Add connection</Text>
          </View>
        </View>

        {pairingState !== 'ready' ? (
          // Live region so retrying announces the new outcome instead of silently swapping the card.
          <View accessibilityLiveRegion="polite">
            <PairingPlaceholder
              hasPatients={patients.length > 0}
              onRetry={() => void mirrorsQuery.refetch()}
              state={pairingState}
            />
          </View>
        ) : patient ? (
          <>
            <View style={styles.infoBox}>
              <Text maxFontSizeMultiplier={1.3} style={styles.infoTitle}>{patient.patientName}</Text>
              <Text style={styles.infoText}>
                On the mirror, open setup and enter the 6-digit pairing code shown there.
              </Text>
            </View>

            <View style={styles.card}>
              <Label>Mirror name</Label>
              <TextInput
                accessibilityLabel="Mirror name"
                onChangeText={setMirrorName}
                placeholder={`Mirror for ${patient.patientName}`}
                placeholderTextColor={colors.placeholder}
                style={styles.input}
                value={mirrorName}
              />

              <Label>Mirror pairing code</Label>
              <TextInput
                accessibilityLabel="Mirror pairing code, 6 digits"
                keyboardType="number-pad"
                maxLength={7}
                onChangeText={(value) => setPairingCode(formatPairingInput(value))}
                placeholder="482 913"
                placeholderTextColor={colors.placeholder}
                style={styles.input}
                value={pairingCode}
              />

              <TouchableOpacity
                accessibilityLabel="Scan mirror QR"
                accessibilityRole="button"
                onPress={() => void openScanner()}
                style={styles.scanButton}
              >
                <Feather
                  accessibilityElementsHidden
                  importantForAccessibility="no"
                  name="camera"
                  size={17}
                  color={colors.accent}
                />
                <Text style={styles.scanButtonText}>Scan mirror QR</Text>
              </TouchableOpacity>

              {cameraNotice ? (
                <Text accessibilityLiveRegion="polite" style={styles.notice}>{cameraNotice}</Text>
              ) : null}

              <Label>Mirror timezone</Label>
              <TextInput
                accessibilityLabel="Mirror timezone"
                autoCapitalize="none"
                onChangeText={setTimezone}
                placeholder="Asia/Singapore"
                placeholderTextColor={colors.placeholder}
                style={styles.input}
                value={timezone}
              />

              <TouchableOpacity
                accessibilityRole="button"
                onPress={() =>
                  Alert.alert(
                    'Pairing instructions',
                    'Enter the code displayed on the mirror, or scan the mirror QR in the caregiver app once scanner support is enabled.',
                  )
                }
                style={styles.secondaryButton}
              >
                <Feather
                  accessibilityElementsHidden
                  importantForAccessibility="no"
                  name="help-circle"
                  size={17}
                  color={colors.accent}
                />
                <Text style={styles.secondaryButtonText}>How pairing works</Text>
              </TouchableOpacity>

              <TouchableOpacity
                accessibilityLabel="Add connection"
                accessibilityRole="button"
                accessibilityState={{ busy: connectMirrorMutation.isPending, disabled: connectMirrorMutation.isPending }}
                disabled={connectMirrorMutation.isPending}
                onPress={() => void saveConnection()}
                style={[styles.primaryButton, connectMirrorMutation.isPending && styles.disabledButton]}
              >
                {connectMirrorMutation.isPending ? (
                  <ActivityIndicator color={colors.text.onAccent} />
                ) : (
                  <>
                    <Feather
                      accessibilityElementsHidden
                      importantForAccessibility="no"
                      name="link"
                      size={18}
                      color={colors.text.onAccent}
                    />
                    <Text style={styles.primaryButtonText}>Add connection</Text>
                  </>
                )}
              </TouchableOpacity>
            </View>
          </>
        ) : null}
      </ScrollView>
      <Modal visible={scanning} animationType="slide" onRequestClose={() => setScanning(false)}>
        <View style={styles.scannerWrap}>
          <CameraView
            style={StyleSheet.absoluteFill}
            facing="back"
            barcodeScannerSettings={{ barcodeTypes: ['qr'] }}
            onBarcodeScanned={onScan}
          />
          <View style={styles.scannerOverlay}>
            {/* Announced when the scanner opens: a camera view is silent to a screen reader otherwise. */}
            <Text accessibilityLiveRegion="polite" style={styles.scannerHint}>
              Point at the mirror’s pairing QR
            </Text>
            <TouchableOpacity
              accessibilityLabel="Cancel scanning"
              accessibilityRole="button"
              onPress={() => setScanning(false)}
              style={styles.scannerCancel}
            >
              <Text style={styles.scannerCancelText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <Text style={styles.label}>{children}</Text>;
}

/**
 * Nothing here ever renders the server's error text. A caregiver opening this screen is trying to reach
 * their mother's mirror, and a raw string like "Not found" reads as news about her rather than about the
 * request — so a failure is always framed as a connection problem with a way to try again.
 */
function PairingPlaceholder({
  hasPatients,
  onRetry,
  state,
}: {
  hasPatients: boolean;
  onRetry: () => void;
  state: Exclude<PairingState, 'ready'>;
}) {
  if (state === 'loading') {
    return <LoadingState title="Loading pairing details" message="Getting the mirror settings for this profile." />;
  }

  if (state === 'signed-out') {
    return (
      <EmptyState
        icon="lock"
        title="Sign in again to pair a mirror"
        message="Your mirrors are kept private to your account. Signing out and back in will reconnect them."
      />
    );
  }

  if (state === 'failed') {
    return (
      <ErrorState
        onRetry={onRetry}
        title="We could not load the pairing details"
        message="This is usually a connection problem, not something about your loved one. The mirror keeps its code on screen while you try again."
      />
    );
  }

  return (
    <EmptyState
      icon="users"
      title={hasPatients ? 'We could not match this profile' : 'No profiles to pair yet'}
      message={
        hasPatients
          ? 'Go back to Manage linked mirrors and choose who this mirror belongs to.'
          : 'Add a loved one first, then come back here to pair their mirror.'
      }
    />
  );
}

function formatPairingInput(value: string) {
  return value.replace(/\D/g, '').slice(0, 6).replace(/(\d{3})(\d{1,3})/, '$1 $2');
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.surface.page },
  scroll: { flex: 1 },
  content: { gap: spacing.lg, padding: spacing.xl, paddingBottom: 52 },
  header: { alignItems: 'center', flexDirection: 'row', gap: spacing.md, marginBottom: spacing.xs },
  backButton: {
    alignItems: 'center',
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: MIN_TOUCH_TARGET / 2,
    borderWidth: 1,
    height: MIN_TOUCH_TARGET,
    justifyContent: 'center',
    width: MIN_TOUCH_TARGET,
  },
  headerTextBlock: { flex: 1 },
  eyebrow: {
    color: colors.text.tertiary,
    fontSize: fontSize.caption,
    fontWeight: '700',
    letterSpacing: 0.8,
    textTransform: 'uppercase',
  },
  title: {
    color: colors.text.primary,
    fontFamily: fontFamily.display,
    fontSize: scaleSize(28),
    fontWeight: '500',
    marginTop: spacing.xs,
  },
  card: {
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: scaleSize(18),
    borderWidth: 1,
    gap: spacing.md,
    padding: scaleSize(18),
  },
  infoBox: {
    backgroundColor: '#FFF6EA',
    borderColor: colors.border.default,
    borderRadius: scaleSize(18),
    borderWidth: 1,
    gap: spacing.sm,
    padding: scaleSize(18),
  },
  infoTitle: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: scaleSize(24), fontWeight: '500' },
  infoText: { color: colors.text.secondary, fontSize: fontSize.subheading, lineHeight: scaleSize(22) },
  label: { color: colors.text.secondary, fontSize: fontSize.body, fontWeight: '700', marginTop: spacing.xs },
  notice: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: scaleSize(19) },
  input: {
    backgroundColor: '#FFFDF8',
    borderColor: colors.border.strong,
    borderRadius: scaleSize(12),
    borderWidth: 1,
    color: colors.text.primary,
    fontSize: scaleSize(18),
    fontWeight: '600',
    paddingHorizontal: scaleSize(14),
    paddingVertical: scaleSize(13),
  },
  primaryButton: {
    alignItems: 'center',
    backgroundColor: colors.accent,
    borderRadius: scaleSize(12),
    flexDirection: 'row',
    gap: spacing.sm,
    justifyContent: 'center',
    minHeight: 48,
    marginTop: spacing.sm,
  },
  primaryButtonText: { color: colors.text.onAccent, fontSize: fontSize.subheading, fontWeight: '800' },
  secondaryButton: {
    alignItems: 'center',
    borderColor: colors.border.strong,
    borderRadius: scaleSize(12),
    borderWidth: 1,
    flexDirection: 'row',
    gap: spacing.sm,
    justifyContent: 'center',
    minHeight: 46,
  },
  secondaryButtonText: { color: colors.accent, fontSize: fontSize.subheading, fontWeight: '700' },
  disabledButton: { opacity: 0.7 },
  scanButton: {
    alignItems: 'center', borderColor: colors.border.strong, borderRadius: scaleSize(12), borderWidth: 1,
    flexDirection: 'row', gap: spacing.sm, justifyContent: 'center', minHeight: 46,
  },
  scanButtonText: { color: colors.accent, fontSize: fontSize.subheading, fontWeight: '700' },
  // Opaque black behind the camera preview, not the shadow colour that happens to share its value.
  scannerWrap: { backgroundColor: '#000000', flex: 1 },
  // Padded so the hint still has margins when it wraps to two or three lines at large system text sizes.
  scannerOverlay: {
    alignItems: 'center',
    bottom: scaleSize(48),
    gap: spacing.lg,
    left: 0,
    paddingHorizontal: scaleSize(24),
    position: 'absolute',
    right: 0,
  },
  scannerHint: {
    backgroundColor: 'rgba(0,0,0,0.55)', borderRadius: radius.md, color: colors.text.onAccent,
    fontSize: fontSize.subheading, fontWeight: '700', overflow: 'hidden', paddingHorizontal: scaleSize(14),
    paddingVertical: spacing.sm,
  },
  scannerCancel: {
    alignItems: 'center',
    backgroundColor: colors.surface.card,
    borderRadius: radius.md,
    justifyContent: 'center',
    // Was ~43pt tall; the only escape from a full-screen camera must not be a near-miss tap.
    minHeight: MIN_TOUCH_TARGET,
    paddingHorizontal: spacing.xxl,
    paddingVertical: spacing.md,
  },
  scannerCancelText: { color: colors.text.primary, fontSize: fontSize.subheading, fontWeight: '800' },
});
