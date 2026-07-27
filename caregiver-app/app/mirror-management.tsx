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

import { EmptyState, ErrorState, LoadingState } from '../src/components/ScreenState';
import { apiGet, apiSend } from '../src/lib/apiClient';
import { getStoredAuthSession } from '../src/lib/authSession';
import { invalidateCaregiverConfig } from '../src/lib/queryKeys';
import { colors, fontFamily, fontSize, MIN_TOUCH_TARGET, radius, scaleSize, spacing } from '../src/theme';

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
    enabled: Boolean(session?.userId),
    queryKey: ['mirrors', session?.userId || ''],
    queryFn: async () => {
      const body = await apiGet<{ patients?: MirrorPatient[] }>(
        `/api/nurse-patient-config/mirrors?nurseId=${encodeURIComponent(session?.userId || '')}`,
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
        nurseId: session?.userId,
      }),
    }),
    onSuccess: async (result) => {
      await queryClient.invalidateQueries({ queryKey: ['mirrors', session?.userId || ''] });
      await invalidateCaregiverConfig(queryClient);
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
  // AND-ed with isPending on purpose. react-query keeps `variables` after a mutation settles, so relying on
  // it alone left a card spinning "Deleting connection…" forever after a FAILED unlink — with the only
  // control on that card disabled and announced as busy, so there was no way to retry short of a remount.
  const savingPatientId = patchMirrorMutation.isPending ? patchMirrorMutation.variables?.patientId || '' : '';
  const { refetch: refetchMirrors } = mirrorsQuery;

  useFocusEffect(
    useCallback(() => {
      if (session?.userId) {
        void refetchMirrors();
      }
    }, [refetchMirrors, session?.userId]),
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
    if (!session?.userId || patchMirrorMutation.isPending) return;
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
          <TouchableOpacity
            accessibilityLabel="Go back"
            accessibilityRole="button"
            style={styles.backButton}
            onPress={goBack}
          >
            <Feather name="chevron-left" size={24} color={colors.accent} />
          </TouchableOpacity>
          <View style={styles.headerTextBlock}>
            <Text style={styles.eyebrow}>Voice Companion</Text>
            <Text maxFontSizeMultiplier={1.3} style={styles.title}>Manage linked mirrors</Text>
          </View>
        </View>

        {patients.length === 0 ? (
          <MirrorsPlaceholder
            isSignedIn={Boolean(session?.userId)}
            isLoading={mirrorsQuery.isLoading}
            hasError={Boolean(mirrorsQuery.error)}
            onRetry={() => void mirrorsQuery.refetch()}
          />
        ) : (
          patients.map((patient) => {
            const isSavingThisPatient = savingPatientId === patient.patientId;
            const isPaired = patient.mirrorVerified && patient.mirrorPairingStatus === 'paired';
            return (
              <View key={patient.patientId} style={styles.card}>
                <View style={styles.cardHeader}>
                  {/* flex:1 so a long name wraps instead of being squeezed off-card once the pill grows
                      with the system font size. */}
                  <View style={styles.cardHeaderTextBlock}>
                    <Text maxFontSizeMultiplier={1.3} style={styles.patientName}>{patient.patientName}</Text>
                    <Text style={styles.mirrorName}>{patient.mirrorName || 'No mirror linked'}</Text>
                  </View>
                  {/* One spoken phrase for the pill, so a reader does not announce a bare word "Unpaired"
                      with no idea what it belongs to. */}
                  <View
                    accessible
                    accessibilityLabel={isPaired ? 'Mirror is paired' : 'Mirror is not paired yet'}
                    style={[styles.statusPill, isPaired ? styles.statusPaired : styles.statusUnpaired]}
                  >
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
                    // Destructive and irreversible from here, so the label names exactly what disappears —
                    // "Delete connection" alone tells a screen-reader user nothing about whose mirror it is.
                    // The in-progress state lives in the label itself. A live region on the child <Text>
                    // does not work here: Android redirects the announcement to this focusable ancestor,
                    // whose own label would still read "Delete the mirror connection…", and iOS ignores
                    // live regions entirely — so a screen-reader caregiver got no confirmation at all.
                    accessibilityLabel={isSavingThisPatient
                      ? `Deleting the mirror connection for ${patient.patientName}`
                      : `Delete the mirror connection between ${patient.mirrorName || 'the mirror'} and ${patient.patientName}`}
                    accessibilityRole="button"
                    accessibilityState={{ busy: isSavingThisPatient, disabled: isSavingThisPatient }}
                    disabled={isSavingThisPatient}
                    onPress={() => confirmUnlink(patient)}
                    style={[styles.deleteButton, isSavingThisPatient && styles.disabledOutlineButton]}
                  >
                    {isSavingThisPatient ? (
                      <ActivityIndicator accessibilityElementsHidden importantForAccessibility="no" color="#AA554B" />
                    ) : (
                      <Feather accessibilityElementsHidden importantForAccessibility="no" name="trash-2" size={17} color="#AA554B" />
                    )}
                    <Text style={styles.deleteButtonText}>
                      {isSavingThisPatient ? 'Deleting connection...' : 'Delete connection'}
                    </Text>
                  </TouchableOpacity>
                ) : (
                  <TouchableOpacity
                    accessibilityLabel={`Add a mirror connection for ${patient.patientName}`}
                    accessibilityRole="button"
                    onPress={() => router.push(`/mirror-management/add?patientId=${patient.patientId}`)}
                    style={styles.primaryButton}
                  >
                    <Feather name="plus" size={18} color={colors.text.onAccent} />
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

/**
 * Loading / signed-out / failed / genuinely-empty are four different situations and the caregiver deserves
 * to be told which. This screen is where someone lands when a mirror has gone quiet, so "nothing linked"
 * and "we could not reach the server" must never look alike. The failure branch never renders the server's
 * error text — those strings are for logs, not for a headline above someone's mother's name.
 */
function MirrorsPlaceholder({
  isSignedIn,
  isLoading,
  hasError,
  onRetry,
}: {
  isSignedIn: boolean;
  isLoading: boolean;
  hasError: boolean;
  onRetry: () => void;
}) {
  if (isLoading) {
    return <LoadingState title="Loading mirror connections" />;
  }

  if (!isSignedIn) {
    return (
      <EmptyState
        icon="lock"
        title="Sign in again to manage mirrors"
        message="Your mirror connections are kept private to your account. Signing out and back in will reconnect them."
      />
    );
  }

  if (hasError) {
    return (
      <ErrorState
        title="We could not load your mirror connections"
        message="This is usually a connection problem, not something about your loved one."
        onRetry={onRetry}
      />
    );
  }

  return (
    <EmptyState
      icon="monitor"
      title="No mirrors to manage yet"
      message="Once a loved one has been added, their mirror connection will show up here."
    />
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
  // Grouped into a single spoken phrase ("Pairing code, None") — read as two elements the label and its
  // value drift apart while swiping through four rows in a row.
  return (
    <View accessible accessibilityLabel={`${label}: ${value}`} style={styles.infoRow}>
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
    // Deliberately deeper and warmer than the shared cardShadow — these cards carry a whole connection,
    // so the literal stays until the theme has a token for it.
    shadowColor: '#6E5B4B',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.08,
    shadowRadius: 18,
  },
  cardHeader: { alignItems: 'flex-start', flexDirection: 'row', gap: spacing.md, justifyContent: 'space-between' },
  cardHeaderTextBlock: { flex: 1 },
  patientName: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: scaleSize(24), fontWeight: '500' },
  mirrorName: { color: colors.text.secondary, fontSize: fontSize.subheading, marginTop: spacing.xs },
  statusPill: {
    borderRadius: radius.pill,
    borderWidth: 1,
    flexShrink: 0,
    paddingHorizontal: spacing.md,
    paddingVertical: 6,
  },
  statusPaired: { backgroundColor: '#F1F7ED', borderColor: '#ABC5A1' },
  statusUnpaired: { backgroundColor: colors.surface.page, borderColor: colors.border.strong },
  // Whether the mirror is actually paired is the whole point of this screen, so this reads at 13 rather
  // than the old 12 and is left free to grow with the system font (the pill is padding-sized, not fixed).
  statusText: { fontSize: fontSize.caption, fontWeight: '700' },
  statusTextPaired: { color: '#617A58' },
  statusTextUnpaired: { color: '#786C5C' },
  infoRow: {
    alignItems: 'center',
    borderTopColor: colors.border.subtle,
    borderTopWidth: 1,
    flexDirection: 'row',
    // gap keeps the label and the right-aligned value from colliding once both grow at large font sizes.
    gap: spacing.md,
    justifyContent: 'space-between',
    paddingTop: scaleSize(10),
  },
  infoLabel: { color: '#786C5C', fontSize: fontSize.body, fontWeight: '600' },
  infoValue: { color: colors.text.primary, flex: 1, fontSize: fontSize.bodyLarge, textAlign: 'right' },
  primaryButton: {
    alignItems: 'center',
    backgroundColor: colors.accent,
    borderRadius: scaleSize(12),
    flexDirection: 'row',
    gap: spacing.sm,
    justifyContent: 'center',
    minHeight: 48,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  // flexShrink lets the label wrap inside the button at large system font sizes; without it the row
  // overflows the rounded edge and the last word is clipped.
  primaryButtonText: { color: colors.text.onAccent, flexShrink: 1, fontSize: fontSize.subheading, fontWeight: '800' },
  deleteButton: {
    alignItems: 'center',
    borderColor: '#E6C8C4',
    borderRadius: scaleSize(12),
    borderWidth: 1,
    flexDirection: 'row',
    gap: spacing.sm,
    justifyContent: 'center',
    minHeight: scaleSize(46),
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  disabledOutlineButton: { borderColor: colors.border.default, opacity: 0.75 },
  deleteButtonText: { color: '#AA554B', flexShrink: 1, fontSize: fontSize.subheading, fontWeight: '700' },
});
