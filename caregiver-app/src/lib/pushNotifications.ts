import Constants from 'expo-constants';
import { Platform } from 'react-native';
import { registerNotificationDeviceV1 } from './v1Client';
import { hasV1Session } from './v1AuthSession';

type NotificationsModule = typeof import('expo-notifications');

// `nurseId` is retained so callers keep their existing shape, but the device is registered against the
// v1 human token (identity comes from the bearer, not the body). The legacy `/caregiver-devices` route
// this used to POST to never existed on the server — every registration 404'd and was swallowed.
type RegisterPushNotificationDeviceInput = {
  nurseId: string;
};

export type PushNotificationDeviceRegistration = {
  expoPushToken: string;
  platform: 'ios' | 'android' | 'web' | 'unknown';
  appVersion?: string;
};

type RegisterPushNotificationDeviceResult = {
  ok: boolean;
  reason?: string;
  deviceId?: string;
};

let notificationHandlerConfigured = false;

/**
 * Installs the foreground display handler and the Android channel, with NO dependency on being signed in
 * or on a registration round-trip succeeding.
 *
 * This used to happen only as a side effect of getPushNotificationDeviceRegistration, which sits behind an
 * early `hasV1Session()` return. Since v1 login is best-effort by design, a caregiver whose v1 token had
 * not come up got no handler — and on Android a delivered push is only shown while the app is FOREGROUND
 * if a handler asks for it. So the alert arrived, the phone stayed silent, and the caregiver saw it only
 * after opening the Notifications tab: exactly the "push doesn't work" report, with nothing wrong in the
 * delivery chain. Call this at app start, before anything can be received.
 */
export async function prepareNotificationDisplay(): Promise<void> {
  if (Platform.OS === 'web') return;
  const Notifications = await loadNotifications();
  if (!Notifications) return;
  try {
    configureNotificationHandler(Notifications);
    await configureAndroidChannel(Notifications);
  } catch (error) {
    // Display setup is best-effort like the rest of push: never let it break app start.
    console.warn('[pushNotifications] display setup unavailable', error);
  }
}

export async function registerPushNotificationDevice({
  nurseId,
}: RegisterPushNotificationDeviceInput): Promise<RegisterPushNotificationDeviceResult> {
  if (!nurseId) {
    return { ok: false, reason: 'Missing nurse id.' };
  }

  // The v1 route is authenticated. v1 login is best-effort by design, so a caregiver signed in only via
  // legacy simply has no push device this session — that must not surface as a sign-in failure.
  if (!hasV1Session()) {
    return { ok: false, reason: 'Sign in again to enable alerts on this phone.' };
  }

  const registration = await getPushNotificationDeviceRegistration();
  if (!registration.ok || !registration.device) {
    return { ok: false, reason: registration.reason };
  }

  try {
    const body = await registerNotificationDeviceV1(registration.device);
    return { ok: true, deviceId: body?.deviceId };
  } catch (error) {
    return {
      ok: false,
      reason: error instanceof Error ? error.message : 'Unable to register push device.',
    };
  }
}

export async function getPushNotificationDeviceRegistration(): Promise<{
  ok: boolean;
  reason?: string;
  device?: PushNotificationDeviceRegistration;
}> {
  if (Platform.OS === 'web') {
    return { ok: false, reason: 'Push notifications are not supported on web.' };
  }

  const Notifications = await loadNotifications();
  if (!Notifications) {
    return { ok: false, reason: 'Expo notifications are unavailable.' };
  }

  // Push is a best-effort enhancement and must NEVER block sign-in / onboarding. Without an FCM
  // config (no google-services.json / googleServicesFile) getExpoPushTokenAsync throws
  // "Default FirebaseApp is not initialized" — swallow that (and any permission/channel error) and
  // degrade to "no push this session" instead of surfacing it as a sign-in failure.
  try {
    configureNotificationHandler(Notifications);
    await configureAndroidChannel(Notifications);

    const permission = await requestNotificationPermission(Notifications);
    if (!permission) {
      return { ok: false, reason: 'Notification permission was not granted.' };
    }

    const projectId = Constants.expoConfig?.extra?.eas?.projectId || Constants.easConfig?.projectId;
    const tokenResponse = await Notifications.getExpoPushTokenAsync(projectId ? { projectId } : undefined);
    const expoPushToken = tokenResponse.data;
    if (!isExpoPushToken(expoPushToken)) {
      return { ok: false, reason: 'Expo push token is invalid.' };
    }

    return {
      ok: true,
      device: {
        expoPushToken,
        platform: normalizePlatform(Platform.OS),
        appVersion: Constants.expoConfig?.version,
      },
    };
  } catch (error) {
    console.warn('[pushNotifications] push registration unavailable', error);
    return { ok: false, reason: friendlyPushError(error) };
  }
}

// FCM token retrieval fails on devices WITHOUT Google Play Services (many Android emulators, some
// Huawei / China-ROM phones) with MISSING_INSTANCEID_SERVICE / SERVICE_NOT_AVAILABLE — a device
// limitation, not a bug or a build problem. Surfacing the raw Java/Firebase stack trace to a caregiver
// is alarming and unhelpful, so map the known cases to a plain, reassuring line. Alerts still land in the
// Notifications tab regardless of whether this phone can register for push.
export function friendlyPushError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error ?? '');
  if (/MISSING_INSTANCEID_SERVICE|SERVICE_NOT_AVAILABLE|play[\s_-]*services|google[\s_-]*play/i.test(message)) {
    return "This device doesn't have Google Play Services, so push can't be enabled here. Your alerts still appear in the Notifications tab.";
  }
  if (/FirebaseApp is not initialized|google-services|googleServicesFile/i.test(message)) {
    return 'Push is not set up in this build.';
  }
  return 'Push is unavailable on this device right now. Your alerts still appear in the Notifications tab.';
}

/**
 * Calls `onOpen` when the caregiver taps an alert, including the cold-start case where the tap is what
 * launched the app. Returns a teardown function.
 *
 * Without this a tapped alert just opened Home, so the one thing a caregiver does with a push — read the
 * alert it is about — took three more taps.
 */
export function subscribeToNotificationTaps(onOpen: (data: Record<string, unknown>) => void): () => void {
  if (Platform.OS === 'web') return () => {};
  let subscription: { remove: () => void } | null = null;
  let cancelled = false;

  void (async () => {
    const Notifications = await loadNotifications();
    if (!Notifications || cancelled) return;
    try {
      // A tap that launched the app is already spent by the time we subscribe, so ask for it directly.
      const initial = await Notifications.getLastNotificationResponseAsync();
      if (initial && !cancelled) {
        onOpen((initial.notification.request.content.data ?? {}) as Record<string, unknown>);
      }
      subscription = Notifications.addNotificationResponseReceivedListener((response) => {
        onOpen((response.notification.request.content.data ?? {}) as Record<string, unknown>);
      });
    } catch (error) {
      console.warn('[pushNotifications] tap listener unavailable', error);
    }
  })();

  return () => {
    cancelled = true;
    subscription?.remove();
  };
}

async function loadNotifications() {
  try {
    return await import('expo-notifications');
  } catch (error) {
    console.warn('[pushNotifications] expo-notifications unavailable', error);
    return null;
  }
}

function configureNotificationHandler(Notifications: NotificationsModule) {
  if (notificationHandlerConfigured) {
    return;
  }

  Notifications.setNotificationHandler({
    handleNotification: async () => ({
      shouldPlaySound: true,
      shouldSetBadge: false,
      shouldShowBanner: true,
      shouldShowList: true,
    }),
  });
  notificationHandlerConfigured = true;
}

async function configureAndroidChannel(Notifications: NotificationsModule) {
  if (Platform.OS !== 'android') {
    return;
  }

  await Notifications.setNotificationChannelAsync('reflexion-caregiver', {
    importance: Notifications.AndroidImportance.MAX,
    name: 'Reflexion caregiver alerts',
    vibrationPattern: [0, 250, 250, 250],
  });
}

async function requestNotificationPermission(Notifications: NotificationsModule) {
  const existingPermissions = await Notifications.getPermissionsAsync();
  if (existingPermissions.status === 'granted') {
    return true;
  }

  const requestedPermissions = await Notifications.requestPermissionsAsync();
  return requestedPermissions.status === 'granted';
}

function isExpoPushToken(token: string) {
  return token.startsWith('ExponentPushToken[') || token.startsWith('ExpoPushToken[');
}

function normalizePlatform(value: string): PushNotificationDeviceRegistration['platform'] {
  return value === 'ios' || value === 'android' || value === 'web' ? value : 'unknown';
}
