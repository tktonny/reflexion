import type { ReactNode } from 'react';
import { QueryClientProvider, useMutation } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { Stack, useGlobalSearchParams, usePathname, useRouter } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { View } from 'react-native';
import { getStoredAuthSession, loadStoredAuthSession } from '../src/lib/authSession';
import { loadV1Session } from '../src/lib/v1AuthSession';
import { registerPushNotificationDevice } from '../src/lib/pushNotifications';
import { queryClient } from '../src/lib/queryClient';

export default function RootLayout() {
  return (
    <QueryClientProvider client={queryClient}>
      <StatusBar style="dark" />
      <AuthGate>
        <Stack screenOptions={{ headerShown: false }}>
          <Stack.Screen name="index" />
          <Stack.Screen name="sign-in" />
          <Stack.Screen name="forgot-password" />
          <Stack.Screen name="reset-password" />
          <Stack.Screen name="(tabs)" />
          <Stack.Screen name="onboarding" />
          <Stack.Screen name="profile/[id]" options={{ headerShown: true, title: 'Profile', headerBackTitle: 'Back' }} />
          <Stack.Screen name="trend/[id]" options={{ headerShown: true, title: 'Trend History', headerBackTitle: 'Back' }} />
          <Stack.Screen name="session/[id]" options={{ headerShown: true, title: 'Session Replay', headerBackTitle: 'Back' }} />
          <Stack.Screen name="session-history/[id]" options={{ headerShown: true, title: 'Session History', headerBackTitle: 'Back' }} />
          <Stack.Screen name="session-history/[id]/[date]" options={{ headerShown: true, title: 'Daily Sessions', headerBackTitle: 'Back' }} />
          <Stack.Screen name="mirror-management" options={{ headerShown: false }} />
          <Stack.Screen name="mirror-management/add" options={{ headerShown: false }} />
          <Stack.Screen name="notifications" options={{ headerShown: false }} />
          <Stack.Screen name="faq" options={{ headerShown: true, title: 'FAQ & Guide', headerBackTitle: 'Back' }} />
          <Stack.Screen name="chatbot" options={{ headerShown: true, title: 'Support', headerBackTitle: 'Back' }} />
        </Stack>
      </AuthGate>
    </QueryClientProvider>
  );
}

function AuthGate({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { mode } = useGlobalSearchParams<{ mode?: string }>();
  const [isHydrated, setIsHydrated] = useState(false);
  const { mutate: registerDevice } = useMutation({
    mutationFn: registerPushNotificationDevice,
  });

  // Hydrate the stored sessions ONCE on mount (legacy session gates routing; v1 token feeds
  // status reads). Critically, the navigator (<Stack>) is only withheld on this first boot —
  // never on later navigations. Re-checking the session on every navigation (the previous
  // behaviour) flipped this back to a loading <View/>, which unmounted and remounted <Stack>,
  // re-resolving it to `index` -> <Redirect to="/(tabs)"> and bouncing the user to Home — and
  // racing the mid-transition unmount into "Reflexion keeps stopping".
  useEffect(() => {
    let isMounted = true;
    void Promise.all([loadStoredAuthSession(), loadV1Session()]).then(() => {
      if (!isMounted) return;
      const session = getStoredAuthSession();
      if (session?.nurseId) {
        registerDevice({ nurseId: session.nurseId });
      }
      setIsHydrated(true);
    });
    return () => {
      isMounted = false;
    };
  }, [registerDevice]);

  // Route guard: runs on every navigation, but ONLY issues a redirect — it never gates whether
  // <Stack> renders, so navigating never remounts the navigator. Reads the in-memory session
  // synchronously (already hydrated above).
  useEffect(() => {
    if (!isHydrated) return;
    const session = getStoredAuthSession();
    const isSignUpRoute = pathname === '/onboarding' && mode !== 'add-patient';
    const isPasswordRoute = pathname === '/forgot-password' || pathname === '/reset-password';
    const isPublicRoute = pathname === '/sign-in' || isSignUpRoute || isPasswordRoute;

    if (!session && !isPublicRoute) {
      router.replace('/sign-in');
    } else if (session && pathname === '/sign-in') {
      router.replace('/(tabs)');
    }
  }, [isHydrated, pathname, mode, router]);

  if (!isHydrated) {
    return <View style={{ flex: 1 }} />;
  }

  return <View style={{ flex: 1 }}>{children}</View>;
}
