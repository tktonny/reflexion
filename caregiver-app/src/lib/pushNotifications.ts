import Constants from 'expo-constants';
import { Platform } from 'react-native';
import { apiSend } from './apiClient';

type NotificationsModule = typeof import('expo-notifications');

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

export async function registerPushNotificationDevice({
  nurseId,
}: RegisterPushNotificationDeviceInput): Promise<RegisterPushNotificationDeviceResult> {
  if (!nurseId) {
    return { ok: false, reason: 'Missing nurse id.' };
  }

  const registration = await getPushNotificationDeviceRegistration();
  if (!registration.ok) {
    return { ok: false, reason: registration.reason };
  }

  try {
    const body = await apiSend<{ deviceId?: string }>('/api/caregiver-devices', {
      method: 'POST',
      body: JSON.stringify({
        nurseId,
        ...registration.device,
      }),
    });

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
    console.warn('[pushNotifications] push registration unavailable (FCM not configured?)', error);
    return { ok: false, reason: error instanceof Error ? error.message : 'Push registration unavailable.' };
  }
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
