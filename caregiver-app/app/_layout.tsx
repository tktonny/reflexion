import { QueryClientProvider } from '@tanstack/react-query';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { StyleSheet, View } from 'react-native';

import { CaregiverProvider } from '../src/architecture/CaregiverContext';
import { DemoModeIndicator } from '../src/components/DemoModeIndicator';
import { MotionProvider } from '../src/components/Motion';
import { queryClient } from '../src/lib/queryClient';

/**
 * Replacement product shell. Authentication transport and the future backend schema are intentionally
 * separate from this UX tree; no legacy session gate can redirect into an obsolete screen.
 */
export default function RootLayout() {
  return (
    <MotionProvider>
      <QueryClientProvider client={queryClient}>
        <CaregiverProvider>
          <StatusBar style="dark" />
          <View style={styles.root}>
            <Stack screenOptions={{ animation: 'slide_from_right', gestureEnabled: true, headerShown: false }}>
          <Stack.Screen name="index" />
          <Stack.Screen name="sign-in" />
          <Stack.Screen name="forgot-password" />
          <Stack.Screen name="reset-verification" />
          <Stack.Screen name="reset-password" />
          <Stack.Screen name="create-account" />
          <Stack.Screen name="welcome" />
          <Stack.Screen name="setup" />
          <Stack.Screen name="(tabs)" />
          <Stack.Screen name="loved-one/[id]" />
          <Stack.Screen name="activity/[eventId]" />
          <Stack.Screen name="chat/[id]" />
          <Stack.Screen name="device/[id]" />
          <Stack.Screen name="settings/[section]" />
            </Stack>
            <DemoModeIndicator />
          </View>
        </CaregiverProvider>
      </QueryClientProvider>
    </MotionProvider>
  );
}

const styles = StyleSheet.create({ root: { flex: 1, minWidth: 0 } });
