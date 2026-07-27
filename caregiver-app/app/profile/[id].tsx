import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import React, { useCallback, useMemo, useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, TextInput, ActivityIndicator, Alert, Linking, Image,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import { Feather } from '@expo/vector-icons';
import MiniSparkline from '../../src/components/MiniSparkline';
import { EmptyState, ErrorState, LoadingState } from '../../src/components/ScreenState';
import { fetchPatientTrend, type TrendDay } from '../../src/lib/patientTrendClient';
import { apiSend } from '../../src/lib/apiClient';
import { hasV1Session } from '../../src/lib/v1AuthSession';
import {
  createAwayPeriodV1,
  createManualFlagV1,
  invalidatePatientStatuses,
  patientStatusQueryKey,
  usePatientStatusV1,
} from '../../src/lib/v1Client';
import type { V1Status } from '../../src/lib/v1Status';
import {
  STATUS_META,
  NEUTRAL_STATUS_COLOR,
  getStatusLabel,
  getReasonText,
  getBaselineProgressText,
  getTechnicalNote,
  formatLastInteraction,
} from '../../src/lib/v1Status';
import {
  colors, spacing, radius, fontSize, fontFamily, cardShadow, MIN_TOUCH_TARGET, scaleSize,
} from '../../src/theme';

// Keyed on the authoritative four-state status, not the legacy green/yellow/red vocabulary. Keeping a
// second three-state `Status` type alive next to V1Status is what made "establishing shown as red"
// possible in the first place, so the app now has exactly one status vocabulary.
// The tint is soft and decorative, and `establishing` is deliberately calm — never an alarm colour.
const AVATAR_BG: Record<V1Status, string> = {
  establishing: '#F0F3ED',
  doing_well: '#F0F3ED',
  worth_checking: '#F6EFE5',
  needs_attention: '#F3E8ED',
};

const AVATAR_TEXT: Record<V1Status, string> = {
  establishing: '#4A5745',
  doing_well: '#4A5745',
  worth_checking: '#7A5C30',
  needs_attention: '#6B3D50',
};

// Identity only. Status deliberately does NOT travel through the route param: this screen reads the
// authoritative v1 status itself, and a caller-supplied colour would silently win the race on first paint
// — the old default was 'needs_attention', which is how a patient still establishing their baseline could
// be shown as red in violation of the product rule.
type RealPatientProfile = {
  name: string;
  phoneNumber: string;
  photoUrl?: string;
  lastSpokenAt: string | null;
  lastSpokenLabel: string;
  duration: number;
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
        // Status shares its cache entry with the dashboard, so returning to this screen must re-read it
        // rather than re-render whatever was cached on the first visit.
        void invalidatePatientStatuses(queryClient);
      }
    }, [queryClient, refetchTrend, shouldLoadRealProfile]),
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
  // The status query is disabled without a v1 session, so "no status" has two innocent causes (signed out,
  // never fetched) and one that needs a retry (the request failed). Kept apart so the caregiver is told which.
  const canReadStatus = hasV1Session();

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
    // No status arriving yet reads as calm, never as an alarm.
    const avatarStatus: V1Status = v1Status?.status ?? 'establishing';
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
              {/* One label for the whole pill: the colour dot is decorative, so a screen reader reads the
                  status once instead of announcing an unnamed view next to it. */}
              <View accessible accessibilityLabel={`Status: ${pillLabel}`} style={styles.statusPill}>
                <View
                  accessibilityElementsHidden
                  importantForAccessibility="no"
                  style={[styles.statusPillDot, { backgroundColor: pillColor }]}
                />
                <Text
                  maxFontSizeMultiplier={1.4}
                  numberOfLines={1}
                  style={styles.statusPillText}
                >
                  {pillLabel}
                </Text>
              </View>
              <View
                accessibilityElementsHidden
                importantForAccessibility="no"
                style={[styles.avatar, { backgroundColor: AVATAR_BG[avatarStatus] }]}
              >
                {realProfile.photoUrl ? (
                  <Image source={{ uri: realProfile.photoUrl }} style={styles.avatarImage} />
                ) : (
                  // The circle is a fixed 52pt, so the initials are the one place a cap is warranted.
                  <Text
                    maxFontSizeMultiplier={1.6}
                    style={[styles.avatarText, { color: AVATAR_TEXT[avatarStatus] }]}
                  >
                    {realInitials}
                  </Text>
                )}
              </View>
            </View>
            <Text maxFontSizeMultiplier={1.3} style={styles.bannerName}>{realProfile.name}</Text>
            {reasonLine ? <Text style={styles.reasonLine}>{reasonLine}</Text> : null}
            <Text style={styles.lastSeen}>{lastInteractionText}</Text>
            <Text style={styles.duration}>Duration: {durationText}</Text>
            {technicalNote ? (
              <View style={styles.techNote}>
                <Feather
                  accessibilityElementsHidden
                  importantForAccessibility="no"
                  name="wifi-off"
                  size={13}
                  color="#8E877C"
                />
                <Text style={styles.techNoteText}>{technicalNote}</Text>
              </View>
            ) : null}
            {shouldLoadRealProfile && !v1Status ? (
              <StatusFallbackNote
                canReadStatus={canReadStatus}
                hasError={Boolean(statusQuery.error)}
                onRetry={() => void statusQuery.refetch()}
              />
            ) : null}
          </View>

          <TouchableOpacity
            accessibilityLabel={`Call ${realProfile.name}`}
            accessibilityRole="button"
            style={styles.callBtn}
            onPress={() => callPatient(realProfile)}
          >
            <Feather
              accessibilityElementsHidden
              importantForAccessibility="no"
              name="phone"
              size={17}
              color={colors.text.onAccent}
            />
            <Text style={styles.callBtnText}>Call {realProfile.name}</Text>
          </TouchableOpacity>

          <View style={styles.card}>
            <View style={styles.summaryHeader}>
              <Text style={styles.cardTitle}>Today's summary</Text>
              <TouchableOpacity
                accessibilityLabel="Generate today's summary"
                accessibilityRole="button"
                accessibilityState={{ busy: summaryMutation.isPending, disabled: summaryMutation.isPending }}
                disabled={summaryMutation.isPending}
                // Small chip in a dense card header: growing it to 44pt would unbalance the row, so the
                // touch area is widened instead of the pill.
                hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
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
            </View>
            {/* polite: the summary lands well after the tap, so it should be read out without yanking focus. */}
            <Text
              accessibilityLiveRegion="polite"
              style={generatedSummary ? styles.summaryText : styles.emptyText}
            >
              {generatedSummary || 'No summary yet.'}
            </Text>
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>This week</Text>
            <WeekTrendBody
              avgDuration={avgDuration}
              hasError={Boolean(trendQuery.error)}
              isEnabled={shouldLoadRealProfile}
              isLoading={trendQuery.isLoading}
              onRetry={() => void trendQuery.refetch()}
              talkedDays={talkedDays}
              trend={realTrend}
            />
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>Caregiver actions</Text>
            <View style={styles.actionRow}>
              <TouchableOpacity
                accessibilityRole="button"
                accessibilityState={{ expanded: activeForm === 'flag' }}
                style={[styles.pillBtn, activeForm === 'flag' && styles.pillBtnActive]}
                onPress={() => setActiveForm(activeForm === 'flag' ? 'none' : 'flag')}
              >
                <Feather
                  accessibilityElementsHidden
                  importantForAccessibility="no"
                  name="flag"
                  size={14}
                  color="#9B5F4E"
                />
                <Text style={styles.pillBtnText}>Flag a concern</Text>
              </TouchableOpacity>
              <TouchableOpacity
                accessibilityRole="button"
                accessibilityState={{ expanded: activeForm === 'away' }}
                style={[styles.pillBtn, activeForm === 'away' && styles.pillBtnActive]}
                onPress={() => setActiveForm(activeForm === 'away' ? 'none' : 'away')}
              >
                <Feather
                  accessibilityElementsHidden
                  importantForAccessibility="no"
                  name="calendar"
                  size={14}
                  color="#596C56"
                />
                <Text style={styles.pillBtnText}>Mark as away</Text>
              </TouchableOpacity>
            </View>

            {activeForm === 'flag' ? (
              <View style={styles.form}>
                <Text style={styles.formLabel}>How would you describe it?</Text>
                <View accessibilityRole="tablist" style={styles.segment}>
                  {(['worth_checking', 'needs_attention'] as const).map((option) => {
                    const optionLabel = option === 'worth_checking' ? 'Worth checking' : 'Needs attention';
                    return (
                      <TouchableOpacity
                        key={option}
                        accessibilityLabel={optionLabel}
                        accessibilityRole="tab"
                        accessibilityState={{ selected: flagSeverity === option }}
                        style={[styles.segmentItem, flagSeverity === option && styles.segmentItemActive]}
                        onPress={() => setFlagSeverity(option)}
                      >
                        <Text style={[styles.segmentText, flagSeverity === option && styles.segmentTextActive]}>
                          {optionLabel}
                        </Text>
                      </TouchableOpacity>
                    );
                  })}
                </View>
                <Text style={styles.formLabel}>What did you notice?</Text>
                <TextInput
                  accessibilityLabel="What did you notice?"
                  style={styles.textArea}
                  multiline
                  placeholder="A short note for your own reference"
                  placeholderTextColor={colors.text.tertiary}
                  value={flagReason}
                  onChangeText={setFlagReason}
                />
                <TouchableOpacity
                  // Named explicitly: the spinner below replaces the <Text> that would otherwise supply
                  // this button's accessible name, right when it matters most.
                  accessibilityLabel={flagMutation.isPending ? 'Flagging concern' : 'Flag concern'}
                  accessibilityRole="button"
                  accessibilityState={{ busy: flagMutation.isPending, disabled: flagMutation.isPending }}
                  style={[styles.submitBtn, flagMutation.isPending && styles.submitBtnDisabled]}
                  disabled={flagMutation.isPending}
                  onPress={submitFlag}
                >
                  {flagMutation.isPending ? (
                    <ActivityIndicator accessibilityElementsHidden importantForAccessibility="no" color={colors.text.onAccent} />
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
                  accessibilityLabel="From"
                  accessibilityHint="Enter the first away day as year dash month dash day"
                  style={styles.input}
                  placeholder="YYYY-MM-DD"
                  placeholderTextColor={colors.text.tertiary}
                  autoCapitalize="none"
                  value={awayStart}
                  onChangeText={setAwayStart}
                />
                <Text style={styles.formLabel}>To</Text>
                <TextInput
                  accessibilityLabel="To"
                  accessibilityHint="Enter the last away day as year dash month dash day"
                  style={styles.input}
                  placeholder="YYYY-MM-DD"
                  placeholderTextColor={colors.text.tertiary}
                  autoCapitalize="none"
                  value={awayEnd}
                  onChangeText={setAwayEnd}
                />
                <Text style={styles.formLabel}>Timezone</Text>
                <TextInput
                  accessibilityLabel="Timezone"
                  style={styles.input}
                  placeholder="Asia/Singapore"
                  placeholderTextColor={colors.text.tertiary}
                  autoCapitalize="none"
                  value={awayTimezone}
                  onChangeText={setAwayTimezone}
                />
                <Text style={styles.formLabel}>Reason (optional)</Text>
                <TextInput
                  accessibilityLabel="Reason (optional)"
                  style={styles.input}
                  placeholder="e.g. Visiting family"
                  placeholderTextColor={colors.text.tertiary}
                  value={awayReason}
                  onChangeText={setAwayReason}
                />
                <TouchableOpacity
                  accessibilityLabel={awayMutation.isPending ? 'Saving away period' : 'Save away period'}
                  accessibilityRole="button"
                  accessibilityState={{ busy: awayMutation.isPending, disabled: awayMutation.isPending }}
                  style={[styles.submitBtn, awayMutation.isPending && styles.submitBtnDisabled]}
                  disabled={awayMutation.isPending}
                  onPress={submitAway}
                >
                  {awayMutation.isPending ? (
                    <ActivityIndicator accessibilityElementsHidden importantForAccessibility="no" color={colors.text.onAccent} />
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
        <Feather
          accessibilityElementsHidden
          importantForAccessibility="no"
          name="user"
          size={28}
          color={colors.accent}
        />
        <Text maxFontSizeMultiplier={1.3} style={styles.placeholderTitle}>Bear with us</Text>
        <Text style={styles.placeholderText}>This profile is not ready to show yet.</Text>
      </View>
    </SafeAreaView>
  );
}

/**
 * Loading / not-loaded-for-this-profile / failed / genuinely-empty are four different situations, and this
 * card used to render one anonymous grey box for all of them — a failed trend request looked exactly like a
 * week with no check-ins. The failure branch never renders the server's error text, only the connection
 * framing: a request that did not come back is not news about the person.
 */
function WeekTrendBody({
  avgDuration,
  hasError,
  isEnabled,
  isLoading,
  onRetry,
  talkedDays,
  trend,
}: {
  avgDuration: number;
  hasError: boolean;
  isEnabled: boolean;
  isLoading: boolean;
  onRetry: () => void;
  talkedDays: number;
  trend: TrendDay[];
}) {
  // Data we already hold wins over a failed refresh — the week is still true, only the retry is pending.
  if (trend.length > 0) {
    return (
      <>
        <MiniSparkline data={trend} days={7} height={scaleSize(52)} />
        <Text style={styles.weekStat}>
          Talked {talkedDays} of 7 days · Avg {formatDuration(avgDuration)}
        </Text>
      </>
    );
  }

  if (isLoading) {
    return <LoadingState title="Loading this week" />;
  }

  if (hasError) {
    return <ErrorState compact onRetry={onRetry} title="We could not load this week" />;
  }

  if (!isEnabled) {
    return (
      <EmptyState
        compact
        icon="bar-chart-2"
        title="Not ready yet"
        message="This week's check-ins will appear here once they start coming through."
      />
    );
  }

  return (
    <EmptyState
      compact
      icon="bar-chart-2"
      title="No check-ins this week yet"
      message="We will fill this in as the daily check-ins come through."
    />
  );
}

/**
 * The status pill keeps its calm wording in every unhappy case; this note is what tells the caregiver which
 * case it is. A failure is framed as a connection problem and never carries the server's error string —
 * that is how a screen in this app once greeted people with the headline "Not found".
 */
function StatusFallbackNote({
  canReadStatus,
  hasError,
  onRetry,
}: {
  canReadStatus: boolean;
  hasError: boolean;
  onRetry: () => void;
}) {
  // Signed-out is checked FIRST, and deliberately so. The commonest way this query ends up in `error` is an
  // expired v1 session: the client clears the session when a token refresh fails, leaving both hasError and
  // !canReadStatus true at once. Checking hasError first told those caregivers "connection problem" and
  // offered a Try again that re-fires without a token and 401s forever — an unbreakable loop over the wrong
  // advice. Signing in again is the only thing that actually fixes it.
  if (!canReadStatus) {
    return (
      <View accessibilityLiveRegion="polite" style={styles.statusNote}>
        <Text style={styles.statusNoteText}>
          Sign in again to see today's status. Everything else on this page still works.
        </Text>
      </View>
    );
  }

  if (hasError) {
    return (
      // polite: this appears after a refresh the caregiver triggered, so it should be read out without
      // pulling focus off whatever they were already reading.
      <View accessibilityLiveRegion="polite" style={styles.statusNote}>
        <Text style={styles.statusNoteText}>
          We could not refresh today's status just now. This is usually a connection problem, not news about
          your loved one.
        </Text>
        <TouchableOpacity
          accessibilityLabel="Try loading today's status again"
          accessibilityRole="button"
          onPress={onRetry}
          style={styles.statusNoteRetry}
        >
          <Text style={styles.statusNoteRetryText}>Try again</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return null;
}

function ActionCard({ icon, label, onPress }: { icon: any; label: string; onPress: () => void }) {
  return (
    <TouchableOpacity
      accessibilityLabel={label}
      accessibilityRole="button"
      style={styles.actionCard}
      onPress={onPress}
      activeOpacity={0.75}
    >
      <Feather
        accessibilityElementsHidden
        importantForAccessibility="no"
        name={icon}
        size={20}
        color={colors.accent}
      />
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
  safe: { flex: 1, backgroundColor: colors.surface.page },
  notFound: { padding: 30, fontSize: 16, color: colors.text.tertiary, textAlign: 'center' },
  placeholder: {
    alignItems: 'center',
    flex: 1,
    gap: spacing.sm,
    justifyContent: 'center',
    paddingHorizontal: spacing.xxl,
  },
  placeholderTitle: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: scaleSize(24), fontWeight: '500' },
  placeholderText: { color: colors.text.secondary, fontSize: fontSize.subheading, lineHeight: 22, textAlign: 'center' },

  content: { paddingHorizontal: spacing.xl, paddingBottom: scaleSize(48), paddingTop: spacing.lg },

  banner: {
    backgroundColor: colors.surface.card,
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border.default,
    padding: spacing.xl,
    marginBottom: 14,
    ...cardShadow,
  },
  bannerTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: spacing.md,
  },
  avatar: {
    width: scaleSize(52),
    height: scaleSize(52),
    borderRadius: radius.pill,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  avatarImage: { height: '100%', width: '100%' },
  avatarText: { fontSize: scaleSize(18), fontWeight: '500', fontFamily: fontFamily.display },
  bannerName: { fontSize: scaleSize(22), fontWeight: '500', color: colors.text.primary, fontFamily: fontFamily.display, marginBottom: spacing.xs },
  reasonLine: { fontSize: fontSize.bodyLarge, color: '#4A433C', lineHeight: 20, marginBottom: spacing.xs },
  lastSeen: { fontSize: fontSize.bodyLarge, color: colors.text.secondary },
  duration: { fontSize: fontSize.body, color: colors.text.tertiary, marginTop: 2 },

  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    flexShrink: 1,
    maxWidth: '78%',
    gap: 7,
    paddingHorizontal: spacing.md,
    paddingVertical: 6,
    borderRadius: radius.pill,
    backgroundColor: colors.surface.muted,
    borderWidth: 1,
    borderColor: colors.border.default,
  },
  statusPillDot: { width: 9, height: 9, borderRadius: radius.pill },
  statusPillText: { fontSize: fontSize.body, fontWeight: '600', color: '#4A433C' },
  techNote: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 7,
    marginTop: spacing.md,
    padding: 10,
    borderRadius: radius.md,
    backgroundColor: colors.surface.muted,
  },
  techNoteText: { flex: 1, fontSize: fontSize.body, color: '#6E6459', lineHeight: 18 },
  statusNote: {
    backgroundColor: colors.surface.muted,
    borderRadius: radius.md,
    gap: 10,
    marginTop: spacing.md,
    padding: 10,
  },
  statusNoteText: { fontSize: fontSize.body, color: '#6E6459', lineHeight: 18 },
  statusNoteRetry: {
    alignItems: 'center',
    alignSelf: 'flex-start',
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: radius.pill,
    borderWidth: 1,
    justifyContent: 'center',
    // 44pt: this is the recovery path, so it must be easy to hit one-handed.
    minHeight: MIN_TOUCH_TARGET,
    paddingHorizontal: spacing.lg,
  },
  statusNoteRetryText: { color: colors.accent, fontSize: fontSize.body, fontWeight: '700' },

  actionRow: { flexDirection: 'row', gap: 10, flexWrap: 'wrap' },
  pillBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: radius.pill,
    backgroundColor: colors.surface.muted,
    borderWidth: 1,
    borderColor: colors.border.default,
    // Was ~40pt tall, under the 44pt minimum for a stressed one-handed tap.
    minHeight: MIN_TOUCH_TARGET,
  },
  pillBtnActive: { borderColor: '#C4B9AF', backgroundColor: '#EFE9E0' },
  pillBtnText: { fontSize: fontSize.body, fontWeight: '600', color: '#4A433C' },
  form: { marginTop: spacing.lg, gap: spacing.sm },
  formLabel: { fontSize: fontSize.body, fontWeight: '600', color: colors.text.secondary, marginTop: 6 },
  formHint: { fontSize: fontSize.body, color: '#736D64', lineHeight: 18 },
  segment: { flexDirection: 'row', gap: spacing.sm },
  segmentItem: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    borderRadius: radius.md,
    backgroundColor: colors.surface.muted,
    borderWidth: 1,
    borderColor: colors.border.default,
    minHeight: MIN_TOUCH_TARGET,
  },
  segmentItemActive: { backgroundColor: '#EFE9E0', borderColor: '#C4B9AF' },
  segmentText: { fontSize: fontSize.body, fontWeight: '600', color: colors.text.secondary },
  segmentTextActive: { color: colors.text.primary },
  input: {
    backgroundColor: colors.surface.input,
    borderColor: colors.border.default,
    borderRadius: radius.md,
    borderWidth: 1,
    color: colors.text.primary,
    fontSize: fontSize.subheading,
    paddingHorizontal: spacing.md,
    paddingVertical: 11,
  },
  textArea: {
    backgroundColor: colors.surface.input,
    borderColor: colors.border.default,
    borderRadius: radius.md,
    borderWidth: 1,
    color: colors.text.primary,
    fontSize: fontSize.subheading,
    minHeight: 72,
    paddingHorizontal: spacing.md,
    paddingVertical: 11,
    textAlignVertical: 'top',
  },
  submitBtn: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.accent,
    borderRadius: scaleSize(12),
    paddingVertical: 13,
    marginTop: spacing.md,
    minHeight: 46,
  },
  submitBtnDisabled: { opacity: 0.65 },
  submitBtnText: { color: colors.text.onAccent, fontSize: fontSize.subheading, fontWeight: '600' },

  callBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    backgroundColor: colors.accent,
    borderRadius: scaleSize(12),
    paddingVertical: 15,
    marginBottom: 14,
  },
  callBtnText: { color: colors.text.onAccent, fontSize: fontSize.subheading, fontWeight: '600' },

  card: {
    backgroundColor: colors.surface.card,
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border.default,
    padding: scaleSize(18),
    marginBottom: scaleSize(14),
    ...cardShadow,
  },
  // 13pt uppercase + letterSpacing: at large system text sizes this title is the widest thing in the
  // summary header row, so it shrinks/wraps instead of overflowing the card (the row already wraps).
  cardTitle: {
    fontSize: fontSize.body,
    fontWeight: '600',
    color: colors.text.tertiary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 10,
    flexShrink: 1,
  },
  summaryHeader: {
    alignItems: 'center',
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
    paddingHorizontal: 10,
    paddingVertical: 7,
  },
  summaryBtnDisabled: {
    opacity: 0.65,
  },
  summaryBtnText: {
    color: colors.accent,
    fontSize: fontSize.caption,
    fontWeight: '700',
    flexShrink: 1,
  },
  summaryText: { fontSize: fontSize.bodyLarge, color: colors.text.secondary, lineHeight: 21 },
  emptyText: { fontSize: fontSize.bodyLarge, color: colors.text.tertiary, lineHeight: 21 },
  topicRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, marginTop: 14 },
  topicChip: {
    backgroundColor: colors.surface.muted,
    paddingHorizontal: spacing.md,
    paddingVertical: 5,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.border.default,
  },
  topicText: { fontSize: fontSize.caption, color: colors.text.secondary, fontWeight: '600' },
  weekStat: { fontSize: fontSize.body, color: colors.text.secondary, marginTop: spacing.md },
  trendPill: {
    alignSelf: 'flex-start',
    marginTop: spacing.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: 5,
    borderRadius: radius.pill,
    backgroundColor: colors.surface.muted,
  },
  trendText: { fontSize: fontSize.caption, fontWeight: '600', color: '#66735D' },

  actionGrid: { flexDirection: 'row', gap: 10 },
  actionCard: {
    flex: 1,
    backgroundColor: colors.surface.card,
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border.default,
    padding: spacing.lg,
    alignItems: 'center',
    gap: spacing.sm,
    ...cardShadow,
  },
  actionLabel: { fontSize: fontSize.caption, color: colors.text.secondary, fontWeight: '600', textAlign: 'center' },
});
