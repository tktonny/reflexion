import { Redirect } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, StyleSheet, View } from 'react-native';

import { BrandLockup } from '../src/components/BrandLockup';
import { loadV1Session } from '../src/lib/v1AuthSession';
import { colors, spacing } from '../src/theme';

export default function SplashScreen() {
  const [destination, setDestination] = useState<string | null>(null);
  useEffect(() => {
    let active = true;
    void loadV1Session().then((session) => {
      if (!active) return;
      const next = session ? '/(tabs)' : '/sign-in';
      setTimeout(() => { if (active) setDestination(next); }, 450);
    });
    return () => { active = false; };
  }, []);
  if (destination) return <Redirect href={destination} />;
  return <View style={styles.screen}><BrandLockup /><ActivityIndicator color={colors.accent} style={styles.spinner} /></View>;
}

const styles = StyleSheet.create({ screen: { alignItems: 'center', backgroundColor: colors.surface.page, flex: 1, justifyContent: 'center', padding: spacing.xl }, spinner: { marginTop: spacing.xl } });
