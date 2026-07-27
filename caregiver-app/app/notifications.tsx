import { Feather } from '@expo/vector-icons';
import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import React, { useCallback, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Linking,
  RefreshControl,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { useFocusEffect, useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { loadCaregiverHome } from '../src/lib/v1Caregiver';
import { getStoredAuthSession } from '../src/lib/authSession';
import { useTabBarClearance } from '../src/lib/useTabBarClearance';
import { friendlyPushError, registerPushNotificationDevice } from '../src/lib/pushNotifications';
import { caregiverConfigKey, notificationsQueryKey } from '../src/lib/queryKeys';
import { hasV1Session } from '../src/lib/v1AuthSession';
import { STATUS_META } from '../src/lib/v1Status';
import {
  listNotificationsV1,
  markNotificationReadV1,
  type V1Notification,
} from '../src/lib/v1Client';
import { colors, spacing, radius, fontSize, fontFamily, scaleSize, MIN_TOUCH_TARGET } from '../src/theme';

// The alert feed is the authoritative v1 read model (GET /api/v1/notifications), produced by the
// server's end-of-day evaluation. It replaced a legacy `/notifications?nurseId=` endpoint that never
// existed on the server — every request 404'd and the screen rendered the words "Not found" as its
// headline. v1 rows carry only a patientId, so names and phone numbers are joined from the loved-one
// list the dashboard already caches.

const PAGE_SIZE = 12;
type NotificationsTab = 'alerts' | 'device';

type TypeMeta = { color: string; icon: keyof typeof Feather.glyphMap; label: string };

// Keyed on the types the server actually emits (jobs/finalizeDay.ts + v1/notifications/service.ts).
// `technical_issue` is deliberately styled and labelled as a device/connection matter, never as a change
// in the person — same product rule the status screens follow.
// The two review-case types share their name with a caregiver status, so their colour is READ from the
// authoritative palette rather than copied. Copies were already byte-identical here, which is precisely how
// a second status vocabulary starts drifting away from the one the doc fixes (§2.9).
const TYPE_META: Record<string, TypeMeta> = {
  completion: { color: STATUS_META.doing_well.dot, icon: 'check-circle', label: 'Checked in' },
  late_completion: { color: '#8B6A92', icon: 'clock', label: 'Later than usual' },
  missed_7pm: { color: STATUS_META.worth_checking.dot, icon: 'alert-circle', label: 'No check-in yet' },
  red_missed_streak: { color: STATUS_META.needs_attention.dot, icon: 'alert-triangle', label: 'Worth a call' },
  technical_issue: { color: '#6F7F92', icon: 'wifi-off', label: 'Connection issue' },
  worth_checking: { color: STATUS_META.worth_checking.dot, icon: 'eye', label: 'Worth checking' },
  needs_attention: { color: STATUS_META.needs_attention.dot, icon: 'alert-triangle', label: 'Needs attention' },
};

// Neutral, matching the `establishing` tone: an unrecognised type must not borrow an alarm colour.
const FALLBACK_TYPE_META: TypeMeta = { color: STATUS_META.establishing.dot, icon: 'bell', label: 'Update' };

function typeMetaFor(type: string): TypeMeta {
  return TYPE_META[type] || FALLBACK_TYPE_META;
}

type PatientDirectoryEntry = { name: string; phoneNumber: string };
type LatestConfigResponse = {
  patients?: { id?: string; patientId?: string; name?: string; phoneNumber?: string }[];
};

export default function NotificationsScreen() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const session = getStoredAuthSession();
  const bottomClearance = useTabBarClearance();
  const canReadAlerts = hasV1Session();
  const [selectedTab, setSelectedTab] = useState<NotificationsTab>('alerts');
  const [deviceMessage, setDeviceMessage] = useState('');

  const notificationsQuery = useInfiniteQuery({
    enabled: canReadAlerts,
    queryKey: notificationsQueryKey(session?.userId),
    initialPageParam: null as string | null,
    queryFn: ({ pageParam }) => listNotificationsV1({ limit: PAGE_SIZE, cursor: pageParam }),
    getNextPageParam: (lastPage) => lastPage.nextCursor,
    // Alerts are written by a server job, so unlike most screens here a cached page really does go out of
    // date on its own. Bounded rather than the app-wide staleTime: Infinity.
    staleTime: 60_000,
  });

  const notifications = useMemo(
    () => (notificationsQuery.data?.pages || []).flatMap((page) => page.data),
    [notificationsQuery.data],
  );

  // Names/phones for the alert cards. Reuses the dashboard's cache entry, so in practice this costs no
  // extra request; v1 notifications carry only a patientId.
  const directoryQuery = useQuery({
    enabled: Boolean(session?.userId),
    queryKey: caregiverConfigKey(session?.userId),
    queryFn: loadCaregiverHome,
  });

  const directory = useMemo(() => {
    const map = new Map<string, PatientDirectoryEntry>();
    for (const patient of directoryQuery.data?.patients || []) {
      map.set(patient.patientId, {
        name: patient.displayName || '',
        phoneNumber: patient.profile.phoneNumber || '',
      });
    }
    return map;
  }, [directoryQuery.data]);

  const { refetch: refetchNotifications } = notificationsQuery;
  useFocusEffect(
    useCallback(() => {
      if (canReadAlerts) void refetchNotifications();
    }, [canReadAlerts, refetchNotifications]),
  );

  const markReadMutation = useMutation({
    mutationFn: markNotificationReadV1,
    onSuccess: (updated) => {
      // Patch in place so the list does not jump while the caregiver is reading it.
      queryClient.setQueryData<typeof notificationsQuery.data>(notificationsQueryKey(session?.userId), (current) => {
        if (!current) return current;
        return {
          ...current,
          pages: current.pages.map((page) => ({
            ...page,
            data: page.data.map((item) => (item.notificationId === updated.notificationId ? updated : item)),
          })),
        };
      });
    },
  });

  const registerDeviceMutation = useMutation({
    mutationFn: registerPushNotificationDevice,
    onMutate: () => setDeviceMessage('Registering this phone...'),
    onSuccess: (result) => {
      setDeviceMessage(result.ok
        ? 'This phone is registered. Alerts will also appear in the list above.'
        : result.reason || 'Could not register this phone for alerts.');
    },
    onError: (error) => {
      // Never the raw message. registerPushNotificationDevice maps its own failures, but a throw from the
      // server call lands here, and a Firebase/Java string read aloud to a caregiver is alarming and tells
      // them nothing they can act on.
      setDeviceMessage(friendlyPushError(error));
    },
  });

  async function refresh() {
    if (!canReadAlerts) return;
    await queryClient.invalidateQueries({ queryKey: ['notificationsV1'] });
  }

  function loadMore() {
    if (notificationsQuery.hasNextPage && !notificationsQuery.isFetchingNextPage) {
      void notificationsQuery.fetchNextPage();
    }
  }

  const header = (
    <View style={styles.headerBlock}>
      <View style={styles.header}>
        <Text style={styles.title} maxFontSizeMultiplier={1.3}>Alerts</Text>
        <TouchableOpacity
          accessibilityLabel="Refresh alerts"
          accessibilityRole="button"
          disabled={notificationsQuery.isFetching}
          onPress={() => void refresh()}
          style={styles.refreshButton}
        >
          <Feather name="refresh-cw" size={18} color={colors.accent} />
        </TouchableOpacity>
      </View>
      <View style={styles.segmentedControl}>
        <TouchableOpacity
          accessibilityLabel="Notifications list"
          accessibilityRole="tab"
          activeOpacity={0.82}
          onPress={() => setSelectedTab('alerts')}
          style={[styles.segmentButton, selectedTab === 'alerts' && styles.segmentButtonActive]}
        >
          <Text style={[styles.segmentText, selectedTab === 'alerts' && styles.segmentTextActive]}>
            Notifications
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          accessibilityLabel="Alerts on this phone"
          accessibilityRole="tab"
          activeOpacity={0.82}
          onPress={() => setSelectedTab('device')}
          style={[styles.segmentButton, selectedTab === 'device' && styles.segmentButtonActive]}
        >
          <Text style={[styles.segmentText, selectedTab === 'device' && styles.segmentTextActive]}>
            This phone
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  if (selectedTab === 'device') {
    return (
      <SafeAreaView style={styles.safe}>
        <View style={styles.content}>
          {header}
          <View style={styles.testCard}>
            <View style={styles.testIcon}>
              <Feather name="bell" size={24} color={colors.accent} />
            </View>
            <Text style={styles.testTitle} maxFontSizeMultiplier={1.3}>Alerts on this phone</Text>
            <Text style={styles.testText}>
              Register this phone so Reflexion can reach you here. Your alerts always appear in the list on
              the Notifications tab, whether or not this phone is registered.
            </Text>
            <TouchableOpacity
              accessibilityLabel="Register this phone for alerts"
              accessibilityRole="button"
              activeOpacity={0.84}
              disabled={registerDeviceMutation.isPending || !session?.userId}
              onPress={() => registerDeviceMutation.mutate({ nurseId: session?.userId || '' })}
              style={[styles.testButton, (registerDeviceMutation.isPending || !session?.userId) && styles.testButtonDisabled]}
            >
              {registerDeviceMutation.isPending ? (
                <ActivityIndicator color={colors.text.onAccent} />
              ) : (
                <>
                  <Feather name="smartphone" size={16} color={colors.text.onAccent} />
                  <Text style={styles.testButtonText}>Register this phone</Text>
                </>
              )}
            </TouchableOpacity>
            {deviceMessage ? (
              <Text style={styles.testMessage}>{deviceMessage}</Text>
            ) : null}
          </View>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safe}>
      <FlatList
        contentContainerStyle={[styles.content, { paddingBottom: bottomClearance }]}
        data={notifications}
        keyExtractor={(item) => item.notificationId}
        ListHeaderComponent={header}
        ListEmptyComponent={
          <AlertsPlaceholder
            canReadAlerts={canReadAlerts}
            isLoading={notificationsQuery.isLoading}
            hasError={Boolean(notificationsQuery.error)}
            onRetry={() => void refresh()}
          />
        }
        ListFooterComponent={
          notifications.length > 0 ? (
            <View style={styles.footer}>
              {notificationsQuery.isFetchingNextPage ? (
                <ActivityIndicator color={colors.accent} />
              ) : notificationsQuery.hasNextPage ? (
                <TouchableOpacity
                  accessibilityRole="button"
                  onPress={loadMore}
                  style={styles.loadMoreButton}
                >
                  <Text style={styles.loadMoreText}>Load more</Text>
                </TouchableOpacity>
              ) : (
                <Text style={styles.endText}>All caught up</Text>
              )}
            </View>
          ) : null
        }
        onEndReached={loadMore}
        onEndReachedThreshold={0.35}
        refreshControl={
          <RefreshControl
            refreshing={notificationsQuery.isRefetching}
            tintColor={colors.accent}
            onRefresh={() => void refresh()}
          />
        }
        renderItem={({ item }) => (
          <AlertCard
            notification={item}
            patient={directory.get(item.patientId)}
            router={router}
            onMarkRead={() => markReadMutation.mutate(item.notificationId)}
          />
        )}
        showsVerticalScrollIndicator={false}
      />
    </SafeAreaView>
  );
}

/**
 * Loading / signed-out / failed / genuinely-empty are four different situations and the caregiver deserves
 * to be told which. Notably a failure NEVER renders the server's error text as copy — that is how this
 * screen came to greet people with the headline "Not found".
 */
function AlertsPlaceholder({
  canReadAlerts,
  isLoading,
  hasError,
  onRetry,
}: {
  canReadAlerts: boolean;
  isLoading: boolean;
  hasError: boolean;
  onRetry: () => void;
}) {
  if (isLoading) {
    return (
      <View style={styles.emptyState}>
        <ActivityIndicator color={colors.accent} />
        <Text style={styles.emptyTitle} maxFontSizeMultiplier={1.3}>Loading alerts</Text>
      </View>
    );
  }

  if (!canReadAlerts) {
    return (
      <View style={styles.emptyState}>
        <View style={styles.emptyIcon}>
          <Feather name="lock" size={28} color={colors.accent} />
        </View>
        <Text style={styles.emptyTitle} maxFontSizeMultiplier={1.3}>Sign in again to see alerts</Text>
        <Text style={styles.emptyText}>
          Your alerts are kept private to your account. Signing out and back in will reconnect them.
        </Text>
      </View>
    );
  }

  if (hasError) {
    return (
      <View style={styles.emptyState}>
        <View style={styles.emptyIcon}>
          <Feather name="cloud-off" size={28} color={colors.accent} />
        </View>
        <Text style={styles.emptyTitle} maxFontSizeMultiplier={1.3}>We could not load your alerts</Text>
        <Text style={styles.emptyText}>
          This is usually a connection problem, not something about your loved one.
        </Text>
        <TouchableOpacity accessibilityRole="button" onPress={onRetry} style={styles.retryButton}>
          <Text style={styles.retryButtonText}>Try again</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.emptyState}>
      <View style={styles.emptyIcon}>
        <Feather name="bell" size={28} color={colors.accent} />
      </View>
      <Text style={styles.emptyTitle} maxFontSizeMultiplier={1.3}>No alerts yet</Text>
      <Text style={styles.emptyText}>
        Nothing needs your attention right now. We will let you know here if that changes.
      </Text>
    </View>
  );
}

const CALL_TO_ACTION_TYPES = new Set(['missed_7pm', 'red_missed_streak', 'worth_checking', 'needs_attention']);
const DAY_DETAIL_TYPES = new Set(['completion', 'late_completion']);

function AlertCard({
  notification,
  patient,
  router,
  onMarkRead,
}: {
  notification: V1Notification;
  patient?: PatientDirectoryEntry;
  router: ReturnType<typeof useRouter>;
  onMarkRead: () => void;
}) {
  const typeMeta = typeMetaFor(notification.type);
  const isUnread = notification.state !== 'read';
  const patientName = patient?.name || 'Your loved one';
  // A connection problem is never a reason to call the person — the mirror is what needs attention.
  const showPatientActions = CALL_TO_ACTION_TYPES.has(notification.type);
  const showDayDetailAction = DAY_DETAIL_TYPES.has(notification.type) && Boolean(notification.localDate);

  return (
    <TouchableOpacity
      accessibilityLabel={`${notification.title}. ${patientName}. ${notification.body}${isUnread ? '. Unread' : ''}`}
      accessibilityRole="button"
      activeOpacity={0.9}
      disabled={!isUnread}
      onPress={onMarkRead}
      style={[styles.card, { borderLeftColor: typeMeta.color }, !isUnread && styles.cardRead]}
    >
      <View style={styles.cardTopRow}>
        <View style={[styles.iconWrap, { backgroundColor: `${typeMeta.color}18` }]}>
          <Feather name={typeMeta.icon} size={18} color={typeMeta.color} />
        </View>
        <View style={styles.cardTextBlock}>
          <View style={styles.titleLine}>
            <Text style={styles.cardTitle}>{notification.title}</Text>
            {isUnread ? <View style={styles.unreadDot} /> : null}
          </View>
          <Text style={styles.patientName}>{patientName}</Text>
          <Text style={styles.bodyText}>{notification.body}</Text>
          <View style={styles.metaRow}>
            <Text style={styles.metaText}>{typeMeta.label}</Text>
            <View style={styles.metaDot} />
            <Text style={styles.metaText}>{formatAlertTime(notification.createdAt)}</Text>
          </View>
          {showPatientActions ? (
            <View style={styles.actionRow}>
              <TouchableOpacity
                accessibilityLabel={`Call ${patientName}`}
                accessibilityRole="button"
                activeOpacity={0.82}
                onPress={() => void callPatient(patientName, patient?.phoneNumber)}
                style={styles.callButton}
              >
                <Feather name="phone" size={15} color={colors.text.onAccent} />
                <Text style={styles.callButtonText}>Call now</Text>
              </TouchableOpacity>
              <TouchableOpacity
                accessibilityLabel={`View ${patientName}'s profile`}
                accessibilityRole="button"
                activeOpacity={0.82}
                onPress={() => viewPatientProfile(notification, patientName, patient?.phoneNumber, router)}
                style={styles.profileButton}
              >
                <Text style={styles.profileButtonText}>View profile</Text>
              </TouchableOpacity>
            </View>
          ) : null}
          {showDayDetailAction ? (
            <TouchableOpacity
              accessibilityRole="button"
              activeOpacity={0.82}
              onPress={() => router.push(`/session-history/${notification.patientId}/${notification.localDate}`)}
              style={styles.summaryButton}
            >
              <Feather name="book-open" size={15} color={colors.text.onAccent} />
              <Text style={styles.summaryButtonText}>View that day</Text>
            </TouchableOpacity>
          ) : null}
        </View>
      </View>
    </TouchableOpacity>
  );
}

async function callPatient(patientName: string, phoneNumber?: string) {
  if (!phoneNumber?.trim()) {
    Alert.alert('No phone number', `${patientName} does not have a phone number saved.`);
    return;
  }

  try {
    await Linking.openURL(`tel:${phoneNumber.replace(/[^\d+]/g, '')}`);
  } catch {
    Alert.alert('Unable to call', `Could not open the phone app for ${phoneNumber}.`);
  }
}

function viewPatientProfile(
  notification: V1Notification,
  patientName: string,
  phoneNumber: string | undefined,
  router: ReturnType<typeof useRouter>,
) {
  if (!notification.patientId) {
    Alert.alert('Profile unavailable', 'This alert is not linked to a profile.');
    return;
  }

  // Only identity is passed through the route param. Status is deliberately NOT included: the profile
  // screen reads the authoritative v1 status itself, and guessing a colour from an alert type is exactly
  // how a patient still learning their routine could get shown as red.
  router.push({
    pathname: '/profile/[id]',
    params: {
      id: notification.patientId,
      patient: JSON.stringify({
        name: patientName,
        phoneNumber: phoneNumber || '',
        lastSpokenAt: null,
        lastSpokenLabel: '',
        duration: 0,
      }),
    },
  });
}

function formatAlertTime(value: string | null) {
  if (!value) {
    return 'Time unavailable';
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return 'Time unavailable';
  }

  return new Intl.DateTimeFormat('en-SG', {
    day: 'numeric',
    hour: 'numeric',
    hour12: true,
    minute: '2-digit',
    month: 'short',
  }).format(date);
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.surface.page },
  // paddingBottom comes from useTabBarClearance at render time; 104 was a guess that happened to be
  // large enough on one phone and is wrong as soon as the bar grows with the system font size.
  content: { padding: spacing.xl },
  headerBlock: { marginBottom: 18 },
  header: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 14,
  },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.display, fontWeight: '600' },
  refreshButton: {
    alignItems: 'center',
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: 22,
    borderWidth: 1,
    height: MIN_TOUCH_TARGET,
    justifyContent: 'center',
    width: MIN_TOUCH_TARGET,
  },
  segmentedControl: {
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: radius.md,
    borderWidth: 1,
    flexDirection: 'row',
    padding: spacing.xs,
  },
  segmentButton: {
    alignItems: 'center',
    borderRadius: radius.sm,
    flex: 1,
    justifyContent: 'center',
    minHeight: 38,
    paddingHorizontal: 10,
  },
  segmentButtonActive: { backgroundColor: colors.accent },
  segmentText: { color: colors.text.secondary, fontSize: fontSize.body, fontWeight: '700', textAlign: 'center' },
  segmentTextActive: { color: colors.text.onAccent },
  testCard: {
    alignItems: 'center',
    backgroundColor: colors.surface.card,
    borderColor: '#ECE4D9',
    borderRadius: radius.sm,
    borderWidth: 1,
    padding: 22,
    shadowColor: '#3B3028',
    shadowOffset: { height: 3, width: 0 },
    shadowOpacity: 0.07,
    shadowRadius: 12,
  },
  testIcon: {
    alignItems: 'center',
    backgroundColor: '#F3E8ED',
    borderRadius: 28,
    height: 56,
    justifyContent: 'center',
    marginBottom: 14,
    width: 56,
  },
  testTitle: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: scaleSize(21), fontWeight: '600' },
  testText: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: scaleSize(20), marginTop: spacing.sm, textAlign: 'center' },
  testButton: {
    alignItems: 'center',
    backgroundColor: colors.accent,
    borderRadius: radius.sm,
    flexDirection: 'row',
    gap: spacing.sm,
    justifyContent: 'center',
    marginTop: 18,
    minHeight: 46,
    paddingHorizontal: 18,
    width: '100%',
  },
  testButtonDisabled: { opacity: 0.72 },
  testButtonText: { color: colors.text.onAccent, fontSize: fontSize.bodyLarge, fontWeight: '700' },
  testMessage: { color: '#5E554E', fontSize: fontSize.body, lineHeight: scaleSize(19), marginTop: spacing.md, textAlign: 'center' },
  card: {
    backgroundColor: colors.surface.card,
    borderColor: '#ECE4D9',
    borderLeftWidth: 3,
    borderRadius: radius.sm,
    borderWidth: 1,
    marginBottom: 14,
    padding: spacing.lg,
    shadowColor: '#3B3028',
    shadowOffset: { height: 3, width: 0 },
    shadowOpacity: 0.07,
    shadowRadius: 12,
  },
  cardRead: { opacity: 0.72 },
  cardTopRow: { alignItems: 'flex-start', flexDirection: 'row', gap: spacing.md },
  iconWrap: {
    alignItems: 'center',
    borderRadius: 18,
    height: 36,
    justifyContent: 'center',
    width: 36,
  },
  cardTextBlock: { flex: 1 },
  titleLine: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    gap: spacing.sm,
    justifyContent: 'space-between',
  },
  cardTitle: {
    color: colors.text.primary,
    flex: 1,
    fontFamily: fontFamily.display,
    fontSize: scaleSize(18),
    fontWeight: '600',
    lineHeight: scaleSize(23),
  },
  unreadDot: {
    backgroundColor: colors.accent,
    borderRadius: radius.pill,
    height: 9,
    marginTop: 7,
    width: 9,
  },
  patientName: { color: '#5E554E', fontSize: fontSize.body, fontWeight: '700', marginTop: 3 },
  bodyText: { color: '#3C342E', fontSize: fontSize.bodyLarge, lineHeight: scaleSize(20), marginTop: 10 },
  metaRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, marginTop: 14 },
  metaText: { color: '#746B63', fontSize: fontSize.caption, fontWeight: '600' },
  metaDot: { backgroundColor: colors.border.strong, borderRadius: 2, height: 4, width: 4 },
  actionRow: { flexDirection: 'row', gap: spacing.md, marginTop: 18 },
  callButton: {
    alignItems: 'center',
    backgroundColor: colors.accent,
    borderRadius: radius.sm,
    flex: 1,
    flexDirection: 'row',
    gap: spacing.sm,
    justifyContent: 'center',
    minHeight: MIN_TOUCH_TARGET,
    paddingHorizontal: spacing.md,
  },
  callButtonText: { color: colors.text.onAccent, fontSize: fontSize.bodyLarge, fontWeight: '700' },
  profileButton: {
    alignItems: 'center',
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: radius.sm,
    borderWidth: 1,
    flex: 1,
    justifyContent: 'center',
    minHeight: MIN_TOUCH_TARGET,
    paddingHorizontal: spacing.md,
  },
  profileButtonText: { color: '#3C342E', fontSize: fontSize.bodyLarge, fontWeight: '700' },
  summaryButton: {
    alignItems: 'center',
    backgroundColor: '#6F7F92',
    borderRadius: radius.sm,
    flexDirection: 'row',
    gap: spacing.sm,
    justifyContent: 'center',
    marginTop: 18,
    minHeight: MIN_TOUCH_TARGET,
    paddingHorizontal: spacing.md,
  },
  summaryButtonText: { color: colors.text.onAccent, fontSize: fontSize.bodyLarge, fontWeight: '700' },
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 360,
    paddingHorizontal: 24,
  },
  emptyIcon: {
    alignItems: 'center',
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: 34,
    borderWidth: 1,
    height: 68,
    justifyContent: 'center',
    marginBottom: 14,
    width: 68,
  },
  emptyTitle: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: scaleSize(22), fontWeight: '600', marginTop: 10 },
  emptyText: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: scaleSize(20), marginTop: spacing.sm, textAlign: 'center' },
  footer: { alignItems: 'center', minHeight: 58, paddingTop: 10 },
  loadMoreButton: {
    backgroundColor: colors.accent,
    borderRadius: radius.sm,
    paddingHorizontal: 18,
    paddingVertical: 11,
  },
  loadMoreText: { color: colors.text.onAccent, fontSize: fontSize.bodyLarge, fontWeight: '700' },
  endText: { color: '#746B63', fontSize: fontSize.caption, fontWeight: '700', paddingVertical: spacing.md },
  retryButton: {
    alignItems: 'center',
    backgroundColor: colors.accent,
    borderRadius: radius.sm,
    justifyContent: 'center',
    marginTop: 18,
    minHeight: 46,
    paddingHorizontal: 24,
  },
  retryButtonText: { color: colors.text.onAccent, fontSize: fontSize.bodyLarge, fontWeight: '700' },
});
