import { Feather } from '@expo/vector-icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import React, { useCallback, useEffect, useState } from 'react';
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
import { apiGet, apiSend } from '../src/lib/apiClient';
import { getStoredAuthSession } from '../src/lib/authSession';
import { registerPushNotificationDevice } from '../src/lib/pushNotifications';

type NotificationStatus = 'queued' | 'sent' | 'failed';
type NotificationType =
  | 'completed_session'
  | 'missed_7pm'
  | 'red_missed_streak'
  | 'late_completion'
  | 'daily_summary';

type CaregiverNotification = {
  id: string;
  patientId: string;
  patientName: string;
  patientPhoneNumber: string;
  date: string;
  type: NotificationType;
  title: string;
  body: string;
  status: NotificationStatus;
  error?: string;
  createdAt: string | null;
};

type NotificationsResponse = {
  notifications?: CaregiverNotification[];
  hasMore?: boolean;
  deviceCount?: number;
};

const PAGE_SIZE = 8;
type NotificationsTab = 'alerts' | 'test';

const TYPE_META: Record<NotificationType, { color: string; icon: keyof typeof Feather.glyphMap; label: string }> = {
  completed_session: { color: '#6F806A', icon: 'check-circle', label: 'Completed' },
  late_completion: { color: '#8B6A92', icon: 'clock', label: 'Late completion' },
  daily_summary: { color: '#6F7F92', icon: 'book-open', label: 'Daily summary' },
  missed_7pm: { color: '#D29A52', icon: 'alert-circle', label: 'Missed' },
  red_missed_streak: { color: '#9B5361', icon: 'alert-triangle', label: 'Needs attention' },
};

const STATUS_META: Record<NotificationStatus, { color: string; label: string }> = {
  queued: { color: '#8B7C70', label: 'Queued' },
  sent: { color: '#6F806A', label: 'Sent' },
  failed: { color: '#9B5361', label: 'Failed' },
};

export default function NotificationsScreen() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const session = getStoredAuthSession();
  const [notifications, setNotifications] = useState<CaregiverNotification[]>([]);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(false);
  const [selectedTab, setSelectedTab] = useState<NotificationsTab>('alerts');
  const [testMessage, setTestMessage] = useState('');
  const notificationsQuery = useQuery({
    enabled: Boolean(session?.nurseId),
    queryKey: ['notifications', session?.nurseId || '', page],
    queryFn: () => apiGet<NotificationsResponse>(
      `/api/notifications?nurseId=${encodeURIComponent(session?.nurseId || '')}&page=${page}&limit=${PAGE_SIZE}`,
    ),
  });
  const { refetch: refetchNotifications } = notificationsQuery;
  useFocusEffect(
    useCallback(() => {
      if (session?.nurseId) {
        void refetchNotifications();
      }
    }, [refetchNotifications, session?.nurseId]),
  );
  const registerDeviceMutation = useMutation({
    mutationFn: registerPushNotificationDevice,
  });
  const testNotificationMutation = useMutation({
    mutationFn: async () => {
      if (!session?.nurseId) throw new Error('Sign in before sending a test notification.');
      setTestMessage('Registering this device...');
      const registration = await registerPushNotificationDevice({ nurseId: session.nurseId });
      if (!registration.ok) {
        throw new Error(registration.reason || 'Unable to register this device for push notifications.');
      }
      setTestMessage('Sending test notification...');
      return apiSend<NotificationsResponse>('/api/notifications/test', {
        method: 'POST',
        body: JSON.stringify({ nurseId: session.nurseId }),
      });
    },
    onSuccess: async (body) => {
      setTestMessage(`Test notification sent to ${body?.deviceCount || 1} device${body?.deviceCount === 1 ? '' : 's'}.`);
      setPage(1);
      await queryClient.invalidateQueries({ queryKey: ['notifications', session?.nurseId || ''] });
    },
    onError: (err) => {
      setTestMessage(err instanceof Error ? err.message : 'Unable to send test notification.');
    },
  });

  useEffect(() => {
    if (!session?.nurseId) {
      setNotifications([]);
      setHasMore(false);
      return;
    }
    if (!notificationsQuery.data) return;
    setNotifications((current) =>
      page === 1
        ? notificationsQuery.data?.notifications || []
        : mergeNotifications(current, notificationsQuery.data?.notifications || []),
    );
    setHasMore(Boolean(notificationsQuery.data.hasMore));
  }, [notificationsQuery.data, page, session?.nurseId]);

  useEffect(() => {
    if (!session?.nurseId) {
      return;
    }

    registerDeviceMutation.mutate({ nurseId: session.nurseId });
  }, [session?.nurseId]);

  async function refresh() {
    setPage(1);
    await queryClient.invalidateQueries({ queryKey: ['notifications', session?.nurseId || ''] });
  }

  async function loadMore() {
    if (!hasMore || notificationsQuery.isFetching) {
      return;
    }

    setPage((current) => current + 1);
  }

  async function sendTestNotification() {
    testNotificationMutation.mutate();
  }

  const header = (
    <View style={styles.headerBlock}>
      <View style={styles.header}>
        <Text style={styles.title}>Alerts</Text>
        <TouchableOpacity
          accessibilityLabel="Refresh alerts"
          disabled={notificationsQuery.isFetching}
          onPress={() => void refresh()}
          style={styles.refreshButton}
        >
          <Feather name="refresh-cw" size={18} color="#87566A" />
        </TouchableOpacity>
      </View>
      <View style={styles.segmentedControl}>
        <TouchableOpacity
          activeOpacity={0.82}
          onPress={() => setSelectedTab('alerts')}
          style={[styles.segmentButton, selectedTab === 'alerts' && styles.segmentButtonActive]}
        >
          <Text style={[styles.segmentText, selectedTab === 'alerts' && styles.segmentTextActive]}>
            Notifications
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          activeOpacity={0.82}
          onPress={() => setSelectedTab('test')}
          style={[styles.segmentButton, selectedTab === 'test' && styles.segmentButtonActive]}
        >
          <Text style={[styles.segmentText, selectedTab === 'test' && styles.segmentTextActive]}>
            Test notification
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  if (selectedTab === 'test') {
    return (
      <SafeAreaView style={styles.safe}>
        <View style={styles.content}>
          {header}
          <View style={styles.testCard}>
            <View style={styles.testIcon}>
              <Feather name="bell" size={24} color="#87566A" />
            </View>
            <Text style={styles.testTitle}>Send a test notification</Text>
            <Text style={styles.testText}>
              This registers the current phone as an enabled Expo push device, then sends one test push.
            </Text>
            <TouchableOpacity
              activeOpacity={0.84}
              disabled={testNotificationMutation.isPending}
              onPress={() => void sendTestNotification()}
              style={[styles.testButton, testNotificationMutation.isPending && styles.testButtonDisabled]}
            >
              {testNotificationMutation.isPending ? (
                <ActivityIndicator color="#FFFFFF" />
              ) : (
                <>
                  <Feather name="send" size={16} color="#FFFFFF" />
                  <Text style={styles.testButtonText}>Send test notification</Text>
                </>
              )}
            </TouchableOpacity>
            {testMessage ? (
              <Text style={styles.testMessage}>{testMessage}</Text>
            ) : null}
          </View>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safe}>
      <FlatList
        contentContainerStyle={styles.content}
        data={notifications}
        keyExtractor={(item) => item.id}
        ListHeaderComponent={header}
        ListEmptyComponent={
          notificationsQuery.isLoading ? (
            <View style={styles.emptyState}>
              <ActivityIndicator color="#87566A" />
              <Text style={styles.emptyTitle}>Loading alerts</Text>
            </View>
          ) : (
            <View style={styles.emptyState}>
              <View style={styles.emptyIcon}>
                <Feather name="bell" size={28} color="#87566A" />
              </View>
              <Text style={styles.emptyTitle}>{notificationsQuery.error instanceof Error ? notificationsQuery.error.message : 'No alerts yet'}</Text>
              <Text style={styles.emptyText}>
                New caregiver alerts will appear here after backend checks run.
              </Text>
            </View>
          )
        }
        ListFooterComponent={
          notifications.length > 0 ? (
            <View style={styles.footer}>
              {notificationsQuery.isFetching && page > 1 ? (
                <ActivityIndicator color="#87566A" />
              ) : hasMore ? (
                <TouchableOpacity onPress={() => void loadMore()} style={styles.loadMoreButton}>
                  <Text style={styles.loadMoreText}>Load more</Text>
                </TouchableOpacity>
              ) : (
                <Text style={styles.endText}>All caught up</Text>
              )}
            </View>
          ) : null
        }
        onEndReached={() => void loadMore()}
        onEndReachedThreshold={0.35}
        refreshControl={
          <RefreshControl refreshing={notificationsQuery.isFetching && page === 1} tintColor="#87566A" onRefresh={() => void refresh()} />
        }
        renderItem={({ item }) => <AlertCard notification={item} router={router} />}
        showsVerticalScrollIndicator={false}
      />
    </SafeAreaView>
  );
}

function AlertCard({
  notification,
  router,
}: {
  notification: CaregiverNotification;
  router: ReturnType<typeof useRouter>;
}) {
  const typeMeta = TYPE_META[notification.type];
  const statusMeta = STATUS_META[notification.status];
  const showPatientActions = notification.type === 'missed_7pm' || notification.type === 'red_missed_streak';
  const showDailySummaryAction = notification.type === 'daily_summary';

  return (
    <View style={[styles.card, { borderLeftColor: typeMeta.color }]}>
      <View style={styles.cardTopRow}>
        <View style={[styles.iconWrap, { backgroundColor: `${typeMeta.color}18` }]}>
          <Feather name={typeMeta.icon} size={18} color={typeMeta.color} />
        </View>
        <View style={styles.cardTextBlock}>
          <View style={styles.titleLine}>
            <Text style={styles.cardTitle} numberOfLines={2}>{notification.title}</Text>
            <View style={[styles.statusPill, { borderColor: statusMeta.color }]}>
              <Text style={[styles.statusText, { color: statusMeta.color }]}>{statusMeta.label}</Text>
            </View>
          </View>
          <Text style={styles.patientName}>{notification.patientName}</Text>
          <Text style={styles.bodyText}>{notification.body}</Text>
          {notification.status === 'failed' && notification.error ? (
            <Text style={styles.errorText}>{notification.error}</Text>
          ) : null}
          <View style={styles.metaRow}>
            <Text style={styles.metaText}>{typeMeta.label}</Text>
            <View style={styles.metaDot} />
            <Text style={styles.metaText}>{formatAlertTime(notification.createdAt)}</Text>
          </View>
          {showPatientActions ? (
            <View style={styles.actionRow}>
              <TouchableOpacity
                activeOpacity={0.82}
                onPress={() => void callPatient(notification)}
                style={styles.callButton}
              >
                <Feather name="phone" size={15} color="#FFFFFF" />
                <Text style={styles.callButtonText}>Call now</Text>
              </TouchableOpacity>
              <TouchableOpacity
                activeOpacity={0.82}
                onPress={() => viewPatientProfile(notification, router)}
                style={styles.profileButton}
              >
                <Text style={styles.profileButtonText}>View profile</Text>
              </TouchableOpacity>
            </View>
          ) : null}
          {showDailySummaryAction ? (
            <TouchableOpacity
              activeOpacity={0.82}
              onPress={() => viewDailySummary(notification, router)}
              style={styles.summaryButton}
            >
              <Feather name="book-open" size={15} color="#FFFFFF" />
              <Text style={styles.summaryButtonText}>View daily summary</Text>
            </TouchableOpacity>
          ) : null}
        </View>
      </View>
    </View>
  );
}

function viewDailySummary(notification: CaregiverNotification, router: ReturnType<typeof useRouter>) {
  if (!notification.patientId || !notification.date) {
    Alert.alert('Summary unavailable', 'This alert is not linked to a daily summary.');
    return;
  }

  router.push(`/session-history/${notification.patientId}/${notification.date}`);
}

async function callPatient(notification: CaregiverNotification) {
  if (!notification.patientPhoneNumber.trim()) {
    Alert.alert('No phone number', `${notification.patientName} does not have a phone number saved.`);
    return;
  }

  const phoneNumber = notification.patientPhoneNumber.replace(/[^\d+]/g, '');
  try {
    await Linking.openURL(`tel:${phoneNumber}`);
  } catch {
    Alert.alert('Unable to call', `Could not open the phone app for ${notification.patientPhoneNumber}.`);
  }
}

function viewPatientProfile(notification: CaregiverNotification, router: ReturnType<typeof useRouter>) {
  if (!notification.patientId) {
    Alert.alert('Profile unavailable', 'This alert is not linked to a patient profile.');
    return;
  }

  router.push({
    pathname: '/profile/[id]',
    params: {
      id: notification.patientId,
      patient: JSON.stringify({
        name: notification.patientName,
        phoneNumber: notification.patientPhoneNumber,
        status: notification.type === 'red_missed_streak' ? 'needs_attention' : 'worth_checking',
        statusLabel: notification.type === 'red_missed_streak' ? 'Needs attention' : 'Worth checking',
        lastSpokenAt: null,
        lastSpokenLabel: 'No completed check-in',
        duration: 0,
      }),
    },
  });
}

function mergeNotifications(current: CaregiverNotification[], next: CaregiverNotification[]) {
  const existing = new Set(current.map((item) => item.id));
  return [...current, ...next.filter((item) => !existing.has(item.id))];
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
  safe: { flex: 1, backgroundColor: '#F8F3EC' },
  content: { padding: 20, paddingBottom: 104 },
  headerBlock: { marginBottom: 18 },
  header: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 14,
  },
  title: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 26, fontWeight: '600' },
  refreshButton: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 22,
    borderWidth: 1,
    height: 44,
    justifyContent: 'center',
    width: 44,
  },
  segmentedControl: {
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 10,
    borderWidth: 1,
    flexDirection: 'row',
    padding: 4,
  },
  segmentButton: {
    alignItems: 'center',
    borderRadius: 8,
    flex: 1,
    justifyContent: 'center',
    minHeight: 38,
    paddingHorizontal: 10,
  },
  segmentButtonActive: { backgroundColor: '#87566A' },
  segmentText: { color: '#756C64', fontSize: 13, fontWeight: '700', textAlign: 'center' },
  segmentTextActive: { color: '#FFFFFF' },
  testCard: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderColor: '#ECE4D9',
    borderRadius: 8,
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
  testTitle: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 21, fontWeight: '600' },
  testText: { color: '#756C64', fontSize: 14, lineHeight: 20, marginTop: 8, textAlign: 'center' },
  testButton: {
    alignItems: 'center',
    backgroundColor: '#87566A',
    borderRadius: 8,
    flexDirection: 'row',
    gap: 8,
    justifyContent: 'center',
    marginTop: 18,
    minHeight: 46,
    paddingHorizontal: 18,
    width: '100%',
  },
  testButtonDisabled: { opacity: 0.72 },
  testButtonText: { color: '#FFFFFF', fontSize: 14, fontWeight: '700' },
  testMessage: { color: '#5E554E', fontSize: 13, lineHeight: 19, marginTop: 12, textAlign: 'center' },
  card: {
    backgroundColor: '#FFFFFF',
    borderColor: '#ECE4D9',
    borderLeftWidth: 3,
    borderRadius: 8,
    borderWidth: 1,
    marginBottom: 14,
    padding: 16,
    shadowColor: '#3B3028',
    shadowOffset: { height: 3, width: 0 },
    shadowOpacity: 0.07,
    shadowRadius: 12,
  },
  cardTopRow: { alignItems: 'flex-start', flexDirection: 'row', gap: 12 },
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
    gap: 8,
    justifyContent: 'space-between',
  },
  cardTitle: {
    color: '#2B2522',
    flex: 1,
    fontFamily: 'Georgia',
    fontSize: 18,
    fontWeight: '600',
    lineHeight: 23,
  },
  statusPill: {
    borderRadius: 999,
    borderWidth: 1,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  statusText: { fontSize: 11, fontWeight: '700' },
  patientName: { color: '#5E554E', fontSize: 13, fontWeight: '700', marginTop: 3 },
  bodyText: { color: '#3C342E', fontSize: 14, lineHeight: 20, marginTop: 10 },
  errorText: { color: '#9B5361', fontSize: 12, lineHeight: 17, marginTop: 8 },
  metaRow: { alignItems: 'center', flexDirection: 'row', gap: 8, marginTop: 14 },
  metaText: { color: '#8B8177', fontSize: 12, fontWeight: '600' },
  metaDot: { backgroundColor: '#D8CFC3', borderRadius: 2, height: 4, width: 4 },
  actionRow: { flexDirection: 'row', gap: 12, marginTop: 18 },
  callButton: {
    alignItems: 'center',
    backgroundColor: '#87566A',
    borderRadius: 8,
    flex: 1,
    flexDirection: 'row',
    gap: 8,
    justifyContent: 'center',
    minHeight: 44,
    paddingHorizontal: 12,
  },
  callButtonText: { color: '#FFFFFF', fontSize: 14, fontWeight: '700' },
  profileButton: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 8,
    borderWidth: 1,
    flex: 1,
    justifyContent: 'center',
    minHeight: 44,
    paddingHorizontal: 12,
  },
  profileButtonText: { color: '#3C342E', fontSize: 14, fontWeight: '700' },
  summaryButton: {
    alignItems: 'center',
    backgroundColor: '#6F7F92',
    borderRadius: 8,
    flexDirection: 'row',
    gap: 8,
    justifyContent: 'center',
    marginTop: 18,
    minHeight: 44,
    paddingHorizontal: 12,
  },
  summaryButtonText: { color: '#FFFFFF', fontSize: 14, fontWeight: '700' },
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 360,
    paddingHorizontal: 24,
  },
  emptyIcon: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 34,
    borderWidth: 1,
    height: 68,
    justifyContent: 'center',
    marginBottom: 14,
    width: 68,
  },
  emptyTitle: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 22, fontWeight: '600', marginTop: 10 },
  emptyText: { color: '#756C64', fontSize: 14, lineHeight: 20, marginTop: 8, textAlign: 'center' },
  footer: { alignItems: 'center', minHeight: 58, paddingTop: 10 },
  loadMoreButton: {
    backgroundColor: '#87566A',
    borderRadius: 8,
    paddingHorizontal: 18,
    paddingVertical: 11,
  },
  loadMoreText: { color: '#FFFFFF', fontSize: 14, fontWeight: '700' },
  endText: { color: '#8B8177', fontSize: 12, fontWeight: '700', paddingVertical: 12 },
});
