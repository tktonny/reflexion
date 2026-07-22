import type { ReactNode } from 'react';
import { QueryClientProvider, useMutation } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { Stack, useGlobalSearchParams, usePathname, useRouter } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { View } from 'react-native';
import { loadStoredAuthSession } from '../src/lib/authSession';
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
  const [hasCheckedSession, setHasCheckedSession] = useState(false);
  const { mutate: registerDevice } = useMutation({
    mutationFn: registerPushNotificationDevice,
  });

  useEffect(() => {
    let isMounted = true;

    const checkSession = async () => {
      setHasCheckedSession(false);
      const session = await loadStoredAuthSession();
      const isSignUpRoute = pathname === '/onboarding' && mode !== 'add-patient';
      const isPasswordRoute = pathname === '/forgot-password' || pathname === '/reset-password';
      const isPublicRoute = pathname === '/sign-in' || isSignUpRoute || isPasswordRoute;

      if (!isMounted) {
        return;
      }

      if (!session && !isPublicRoute) {
        router.replace('/sign-in');
      } else if (session && pathname === '/sign-in') {
        router.replace('/(tabs)');
      }
      if (session?.nurseId && pathname !== '/sign-in' && !isSignUpRoute) {
        registerDevice({ nurseId: session.nurseId });
      }

      setHasCheckedSession(true);
    };

    void checkSession();
    return () => {
      isMounted = false;
    };
  }, [mode, pathname, registerDevice, router]);

  if (!hasCheckedSession) {
    return <View style={{ flex: 1 }} />;
  }

  return <View style={{ flex: 1 }}>{children}</View>;
}
