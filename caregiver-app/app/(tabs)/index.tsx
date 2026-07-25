import { useQuery, useQueryClient } from '@tanstack/react-query';
import React, { useCallback } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, Image,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useRouter } from 'expo-router';
import { Feather } from '@expo/vector-icons';
import { EmptyState, ErrorState, LoadingState } from '../../src/components/ScreenState';
import { apiGet } from '../../src/lib/apiClient';
import { getStoredAuthSession } from '../../src/lib/authSession';
import { hasV1Session } from '../../src/lib/v1AuthSession';
import { caregiverConfigKey } from '../../src/lib/queryKeys';
import { invalidatePatientStatuses, usePatientStatusesV1 } from '../../src/lib/v1Client';
import {
  STATUS_META,
  NEUTRAL_STATUS_COLOR,
  getStatusLabel,
  getReasonText,
  getBaselineProgressText,
  formatLastInteraction,
} from '../../src/lib/v1Status';
import {
  colors, spacing, radius, fontSize, fontFamily, cardShadow, MIN_TOUCH_TARGET,
} from '../../src/theme';

// Summary-strip dots use the authoritative Option-1 status palette (baseline §2.9).
const STATUS_DOT: Record<string, string> = {
  green: STATUS_META.doing_well.dot,
  yellow: STATUS_META.worth_checking.dot,
  red: STATUS_META.needs_attention.dot,
};

// The legacy list still reports a coarse status; it is typed here for accuracy but never rendered —
// every status the caregiver sees comes from src/lib/v1Status.ts via the v1 read model.
type PatientStatus = 'doing_well' | 'worth_checking' | 'needs_attention';

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
  const queryClient = useQueryClient();
  const session = getStoredAuthSession();
  const latestConfigQuery = useQuery({
    // The endpoint requires an explicit nurseId — without one it used to return an arbitrary caregiver's
    // record, so the query stays disabled rather than asking for "whoever is newest".
    enabled: Boolean(session?.nurseId),
    queryKey: caregiverConfigKey(session?.nurseId),
    queryFn: () => apiGet<LatestConfigResponse>(
      `/api/nurse-patient-config/latest?nurseId=${encodeURIComponent(session?.nurseId || '')}`,
    ),
  });
  const { refetch: refetchLatestConfig } = latestConfigQuery;

  // Refresh BOTH the loved-one list and the authoritative v1 status on every focus. Status is the whole
  // point of this screen and the mirror moves it behind the app's back; refetching only the legacy list
  // (the previous behaviour) left the dots frozen for as long as the process stayed alive.
  useFocusEffect(
    useCallback(() => {
      if (session?.nurseId) void refetchLatestConfig();
      void invalidatePatientStatuses(queryClient);
    }, [queryClient, refetchLatestConfig, session?.nurseId]),
  );
  const configuredPatients = Array.isArray(latestConfigQuery.data?.patients) ? latestConfigQuery.data.patients : [];
  const caregiverName = typeof latestConfigQuery.data?.caregiverName === 'string' ? latestConfigQuery.data.caregiverName : '';
  const today = new Date().toLocaleDateString('en-SG', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });
  const displayName = getFirstName(caregiverName) || 'there';

  // Authoritative status comes from the v1 read model (baseline §4), keyed by the same id the legacy
  // list returns (the migration reuses the legacy ObjectId hex as the v1 patient _id). We never bucket
  // off the legacy days-since-conversation status, which reports establishing patients as red.
  const patientIds = configuredPatients.map((patient) => patient.patientId || patient.id);
  const statusResults = usePatientStatusesV1(patientIds);
  const statusValues = statusResults.map((result) => result.data?.status);

  const doingWell = statusValues.filter((status) => status === 'doing_well').length;
  const checkIn = statusValues.filter((status) => status === 'worth_checking').length;
  const attention = statusValues.filter((status) => status === 'needs_attention').length;

  // A zero in this strip must mean "nobody is in this bucket", never "we could not find out" — so when no
  // status arrived at all the counts are replaced rather than shown as zeroes.
  //
  // WHY it is missing decides what to say, and the three reasons need three different messages. Calling all
  // of them a connection problem was wrong twice over: v1 login is best-effort by design (CLAUDE.md), so a
  // caregiver with no v1 session was told to check a connection that was fine, and offered a Try again that
  // could never succeed; and a loved one whose mirror is not paired yet has no monitoring record at all,
  // which is a setup step, not a fault.
  const canReadStatuses = hasV1Session();
  const statusesLoading = statusResults.some((result) => result.isLoading);
  const noStatusArrived = configuredPatients.length > 0
    && !statusesLoading
    && statusResults.every((result) => !result.data);
  const everyPatientUnpaired = noStatusArrived && statusResults.every((result) => result.isUnavailable);
  const statusNotice: 'none' | 'signed-out' | 'not-set-up' | 'failed' = !noStatusArrived
    ? 'none'
    : !canReadStatuses
      ? 'signed-out'
      : everyPatientUnpaired
        ? 'not-set-up'
        : 'failed';
  const retryStatuses = () => {
    void invalidatePatientStatuses(queryClient);
  };

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.headerText}>
            <Text style={styles.greeting}>{getGreeting()}, {displayName}</Text>
            <Text style={styles.date}>{today}</Text>
          </View>
          <TouchableOpacity
            accessibilityLabel="Add a loved one"
            accessibilityRole="button"
            // The circle stays 36pt so the header keeps its proportions; hitSlop takes the tappable area
            // past 44pt, which is what a thumb actually needs.
            hitSlop={{ bottom: 8, left: 8, right: 8, top: 8 }}
            onPress={() => router.push('/onboarding?mode=add-patient&returnTo=home')}
            style={styles.addBtn}
          >
            <Feather
              accessibilityElementsHidden
              importantForAccessibility="no"
              name="plus"
              size={16}
              color={colors.text.onAccent}
            />
          </TouchableOpacity>
        </View>

        {/* Status Summary Strip */}
        {statusNotice !== 'none' ? (
          <StatusNotice notice={statusNotice} onRetry={retryStatuses} />
        ) : (
          <View style={styles.summaryStrip}>
            <SummaryChip dot={STATUS_DOT.green} count={doingWell} label="Doing well" pending={statusesLoading} />
            <View style={styles.divider} />
            <SummaryChip dot={STATUS_DOT.yellow} count={checkIn} label="Worth checking" pending={statusesLoading} />
            <View style={styles.divider} />
            <SummaryChip dot={STATUS_DOT.red} count={attention} label="Needs attention" pending={statusesLoading} />
          </View>
        )}

        {/* Loved One Cards */}
        <Text style={styles.sectionTitle}>Your loved ones</Text>
        {configuredPatients.length ? (
          configuredPatients.map((patient, index) => {
            const statusResult = statusResults[index];
            const v1 = statusResult?.data;
            const dotColor = v1 ? (STATUS_META[v1.status]?.dot ?? NEUTRAL_STATUS_COLOR) : NEUTRAL_STATUS_COLOR;
            // Four reasons for "no status", each said plainly. They used to collapse into one vague
            // "Status updating" that a caregiver could stare at forever without learning that their mirror
            // was never connected, or that signing in again would fix it. None of these is phrased as news
            // about the person.
            const label = v1
              ? getStatusLabel(v1.status, patient.name)
              : statusResult?.isLoading
                ? 'Checking in…'
                : statusResult?.isSignedOut
                  ? 'Sign in again to see today'
                  : statusResult?.isUnavailable
                    ? 'Connect a mirror to start check-ins'
                    : statusResult?.isError
                      ? "Today's update did not reach us"
                      : 'Status updating';
            const metaLine = v1
              ? v1.status === 'establishing'
                ? getBaselineProgressText(v1.baselineProgress)
                : getReasonText(v1.primaryReason, patient.name)
              : patient.lastSpokenLabel;
            const lastInteractionLine = v1 ? formatLastInteraction(v1.lastInteractionAt) : '';
            return (
              <TouchableOpacity
                // One label for the whole row. Read as separate elements this card is six fragments —
                // initials, name, a dot, a status, a reason, a chevron — which is unusable by voice.
                accessibilityLabel={[patient.name, label, metaLine, lastInteractionLine]
                  .filter(Boolean)
                  .join('. ')}
                accessibilityRole="button"
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
                <View
                  accessibilityElementsHidden
                  importantForAccessibility="no"
                  style={styles.patientAvatar}
                >
                  {patient.photoUrl ? (
                    <Image source={{ uri: patient.photoUrl }} style={styles.patientAvatarImage} />
                  ) : (
                    // Initials sit in a fixed circle that cannot grow with the system font without
                    // becoming an oval, so this one label is capped.
                    <Text maxFontSizeMultiplier={1.6} style={styles.patientAvatarText}>
                      {getInitials(patient.name)}
                    </Text>
                  )}
                </View>
                <View style={styles.patientInfo}>
                  <Text style={styles.patientName}>{patient.name}</Text>
                  <View style={styles.patientStatusRow}>
                    <View
                      accessibilityElementsHidden
                      importantForAccessibility="no"
                      style={[styles.patientStatusDot, { backgroundColor: dotColor }]}
                    />
                    <Text style={styles.patientStatusText}>{label}</Text>
                  </View>
                  <Text style={styles.patientMeta}>{metaLine}</Text>
                  {v1 ? (
                    <Text style={styles.patientSubMeta}>{lastInteractionLine}</Text>
                  ) : null}
                </View>
                <Text
                  accessibilityElementsHidden
                  importantForAccessibility="no"
                  style={styles.chevron}
                >
                  ›
                </Text>
              </TouchableOpacity>
            );
          })
        ) : (
          // Polite live region so a caregiver using a screen reader hears the outcome of "Try again"
          // instead of being left on the old placeholder.
          <View accessibilityLiveRegion="polite">
            <LovedOnesPlaceholder
              hasError={Boolean(latestConfigQuery.error)}
              isLoading={latestConfigQuery.isLoading}
              isSignedIn={Boolean(session?.nurseId)}
              onRetry={() => void latestConfigQuery.refetch()}
            />
          </View>
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

/**
 * Loading / signed-out / failed / genuinely-empty are four different situations and the caregiver deserves
 * to be told which. Previously there were two branches, so both a disabled query (no nurseId after a
 * dropped session) and a failed request rendered as "no loved one profiles yet" — telling someone their
 * mother is not in the app when the only problem was sign-in or signal. A failure also never renders the
 * server's error text: raw strings are for logs, not for someone wondering whether Mum is alright.
 */
function LovedOnesPlaceholder({
  hasError,
  isLoading,
  isSignedIn,
  onRetry,
}: {
  hasError: boolean;
  isLoading: boolean;
  isSignedIn: boolean;
  onRetry: () => void;
}) {
  if (isLoading) {
    return <LoadingState message="We are loading your loved ones." />;
  }

  if (!isSignedIn) {
    return (
      <EmptyState
        icon="lock"
        title="Sign in again to see your loved ones"
        message="Your loved ones are kept private to your account. Signing out and back in will reconnect them."
      />
    );
  }

  if (hasError) {
    return (
      <ErrorState
        title="We could not load your loved ones"
        message="This is usually a connection problem, not something about your loved one."
        onRetry={onRetry}
      />
    );
  }

  return (
    <EmptyState
      icon="user-plus"
      title="No loved ones yet"
      message="Add someone you care for and their daily check-ins will show up here."
    />
  );
}

function getFirstName(name: string) {
  return name.trim().split(/\s+/)[0] || '';
}

// The greeting used to be hardcoded "Good morning" — read at 11pm by someone checking on their mother it
// makes the app feel like it is not paying attention. Local device hour, no server involvement.
function getGreeting(now: Date = new Date()) {
  const hour = now.getHours();
  if (hour < 12) return 'Good morning';
  if (hour < 18) return 'Good afternoon';
  return 'Good evening';
}

function getInitials(name: string) {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return '?';
  return parts.slice(0, 2).map((part) => part[0]?.toUpperCase()).join('');
}

// Identity and last-interaction only. The legacy list's own `status` is deliberately NOT forwarded: the
// profile screen reads the authoritative v1 status, and the legacy bucket reports a patient who is still
// establishing a baseline as red.
function toProfileRoutePatient(patient: DashboardPatient) {
  return {
    name: patient.name,
    phoneNumber: patient.phoneNumber,
    photoUrl: patient.photoUrl,
    lastSpokenAt: patient.lastSpokenAt,
    lastSpokenLabel: patient.lastSpokenLabel,
    duration: patient.duration,
  };
}

/**
 * Stands in for the summary strip when no status could be read. One message per reason: a retry is only
 * offered where retrying can actually change the outcome, and nothing here is framed as news about a person.
 */
function StatusNotice({
  notice,
  onRetry,
}: {
  notice: 'signed-out' | 'not-set-up' | 'failed';
  onRetry: () => void;
}) {
  const copy = {
    'signed-out': {
      icon: 'lock' as const,
      text: "Sign out and back in to see today's summary. Your loved ones are listed below either way.",
      retry: false,
    },
    'not-set-up': {
      icon: 'link' as const,
      text: 'Daily check-ins start once a mirror is connected. You can set that up from Settings.',
      retry: false,
    },
    failed: {
      icon: 'cloud-off' as const,
      text: "Today's summary is not available yet. This is a connection issue, not a change in how anyone is doing.",
      retry: true,
    },
  }[notice];

  return (
    <View accessibilityLiveRegion="polite" style={styles.statusUnavailable}>
      <Feather
        accessibilityElementsHidden
        importantForAccessibility="no"
        name={copy.icon}
        size={16}
        color={colors.text.secondary}
      />
      <Text style={styles.statusUnavailableText}>{copy.text}</Text>
      {copy.retry ? (
        <TouchableOpacity
          accessibilityLabel="Try loading today's summary again"
          accessibilityRole="button"
          hitSlop={{ bottom: 8, left: 8, right: 8, top: 8 }}
          onPress={onRetry}
          style={styles.statusRetry}
        >
          <Text style={styles.statusRetryText}>Try again</Text>
        </TouchableOpacity>
      ) : null}
    </View>
  );
}

function SummaryChip({ dot, count, label, pending }: { dot: string; count: number; label: string; pending?: boolean }) {
  return (
    // One reading per chip ("2 worth checking"); otherwise the strip announces a bare number, a coloured
    // dot and a label as three unrelated stops. While statuses are still arriving the count is not
    // asserted at all — a half-loaded strip must not be read as a finished tally.
    <View
      accessible
      accessibilityLabel={pending ? `${label}, still loading` : `${count} ${label.toLowerCase()}`}
      style={styles.chip}
    >
      {/* The count sits between two fixed dividers and is at most two digits, so it is the one place a cap
          loses nothing. The label below it stays uncapped and wraps. */}
      <Text maxFontSizeMultiplier={1.6} style={styles.chipCount}>{pending ? '–' : count}</Text>
      <View
        accessibilityElementsHidden
        importantForAccessibility="no"
        style={[styles.chipDot, { backgroundColor: dot }]}
      />
      <Text style={styles.chipLabel}>{label}</Text>
    </View>
  );
}

function QuickLink({ icon, label, sub, onPress }: { icon: any; label: string; sub: string; onPress: () => void }) {
  return (
    <TouchableOpacity accessibilityRole="button" style={styles.quickLink} onPress={onPress} activeOpacity={0.7}>
      <Feather
        accessibilityElementsHidden
        importantForAccessibility="no"
        name={icon}
        size={22}
        color={colors.accent}
      />
      <Text style={styles.quickLabel}>{label}</Text>
      <Text style={styles.quickSub}>{sub}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.surface.page },
  scroll: { flex: 1 },
  content: { paddingHorizontal: spacing.xl, paddingTop: spacing.xl, paddingBottom: 48 },

  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 22,
  },
  // flex (RN does not shrink flex items by default) so a scaled-up greeting wraps instead of pushing the
  // add button off the right edge.
  headerText: { flex: 1, paddingRight: spacing.md },
  greeting: {
    fontSize: fontSize.display, fontWeight: '500', color: colors.text.primary, fontFamily: fontFamily.display,
  },
  date: { fontSize: fontSize.body, color: colors.text.tertiary, marginTop: 3 },
  addBtn: {
    width: 36,
    height: 36,
    borderRadius: radius.pill,
    backgroundColor: colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
  },

  summaryStrip: {
    flexDirection: 'row',
    backgroundColor: colors.surface.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border.default,
    paddingHorizontal: 10,
    paddingVertical: spacing.lg,
    marginBottom: spacing.xxl,
    alignItems: 'center',
    ...cardShadow,
  },
  // Shown in place of the strip when no status could be read at all, so a zero is never mistaken for a
  // finding. Deliberately neutral (no status colour) — it is a connection notice, not a status.
  statusUnavailable: {
    alignItems: 'center',
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: radius.lg,
    borderWidth: 1,
    gap: spacing.sm,
    marginBottom: spacing.xxl,
    paddingHorizontal: 18,
    paddingVertical: 18,
  },
  statusUnavailableText: {
    color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 21, textAlign: 'center',
  },
  statusRetry: {
    alignItems: 'center', justifyContent: 'center', minHeight: MIN_TOUCH_TARGET, paddingHorizontal: spacing.lg,
  },
  statusRetryText: { color: colors.accent, fontSize: fontSize.subheading, fontWeight: '700' },
  chip: { alignItems: 'center', flex: 1, gap: 5, minWidth: 0 },
  chipCount: {
    fontSize: 24, fontWeight: '500', color: colors.text.primary, fontFamily: fontFamily.display, lineHeight: 29,
  },
  chipDot: { width: 7, height: 7, borderRadius: radius.pill },
  chipLabel: {
    color: colors.text.secondary,
    // Was 11px capped at 82pt wide and clipped to two lines: "Needs attention" was the first thing on the
    // dashboard to lose a word at large system text. minHeight only aligns the three chips, it never caps.
    fontSize: fontSize.body,
    lineHeight: 17,
    minHeight: 28,
    textAlign: 'center',
  },
  divider: {
    width: 1, height: 46, backgroundColor: colors.border.default, marginHorizontal: spacing.xs,
  },

  sectionTitle: {
    fontSize: fontSize.heading,
    fontWeight: '600',
    color: colors.text.primary,
    marginBottom: 14,
    marginTop: spacing.xs,
  },
  patientCard: {
    alignItems: 'center',
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: radius.xl,
    borderWidth: 1,
    flexDirection: 'row',
    gap: spacing.md,
    marginBottom: spacing.md,
    padding: spacing.lg,
    ...cardShadow,
  },
  patientAvatar: {
    alignItems: 'center',
    backgroundColor: colors.surface.muted,
    borderRadius: radius.pill,
    height: 56,
    justifyContent: 'center',
    overflow: 'hidden',
    width: 56,
  },
  patientAvatarImage: {
    height: '100%',
    width: '100%',
  },
  patientAvatarText: {
    color: colors.text.secondary,
    fontFamily: fontFamily.display,
    fontSize: 18,
    fontWeight: '500',
  },
  patientInfo: { flex: 1 },
  patientName: {
    color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '400',
  },
  patientStatusRow: { alignItems: 'center', flexDirection: 'row', gap: 7, marginTop: 7 },
  patientStatusDot: { borderRadius: radius.pill, height: 8, width: 8 },
  // flex so a long status label wraps beside the dot instead of running past the card edge.
  patientStatusText: { color: colors.text.secondary, flex: 1, fontSize: fontSize.body, fontWeight: '400' },
  patientMeta: { color: colors.text.secondary, fontSize: fontSize.body, fontWeight: '500', marginTop: 7 },
  patientSubMeta: { color: colors.text.tertiary, fontSize: fontSize.caption, fontWeight: '400', marginTop: 3 },
  chevron: { fontSize: fontSize.title, color: colors.textDecorative, fontWeight: '300' },

  quickGrid: { flexDirection: 'row', gap: spacing.md },
  quickLink: {
    flex: 1,
    backgroundColor: colors.surface.card,
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border.default,
    padding: 18,
    gap: 6,
    ...cardShadow,
  },
  quickLabel: { fontSize: fontSize.bodyLarge, fontWeight: '600', color: colors.text.primary },
  quickSub: { fontSize: fontSize.caption, color: colors.text.tertiary },
});
