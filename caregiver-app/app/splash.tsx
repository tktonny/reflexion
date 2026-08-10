import { Redirect } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Image, StyleSheet, Text, View, useWindowDimensions } from 'react-native';

import { loadV1Session } from '../src/lib/v1AuthSession';
import { colors, fontFamily, fontSize, scaleSize } from '../src/theme';

export default function SplashScreen() {
  const { width } = useWindowDimensions();
  const [destination, setDestination] = useState<string | null>(null);
  useEffect(() => {
    let active = true;
    void loadV1Session().then((session) => {
      if (!active) return;
      const next = session ? '/(tabs)' : '/sign-in';
      setTimeout(() => {
        if (active) setDestination(next);
      }, 450);
    });
    return () => { active = false; };
  }, []);
  if (destination) return <Redirect href={destination} />;
  const logoWidth = Math.min(width * 0.62, scaleSize(230));
  const illustrationWidth = Math.min(width * 0.59, scaleSize(224));
  const branchWidth = Math.min(width * 0.35, scaleSize(136));
  return (
    <View style={styles.screen}>
      <View pointerEvents="none" style={StyleSheet.absoluteFill}>
        <Image
          accessibilityElementsHidden
          importantForAccessibility="no-hide-descendants"
          resizeMode="contain"
          source={require('../assets/auth/botanical-left.png')}
          style={[styles.botanicalLeft, { height: branchWidth * 2, width: branchWidth }]}
        />
        <Image
          accessibilityElementsHidden
          importantForAccessibility="no-hide-descendants"
          resizeMode="contain"
          source={require('../assets/auth/reflexion-logo.png')}
          style={[styles.logo, { height: logoWidth * (240 / 360), width: logoWidth }]}
        />
        <Image
          accessibilityElementsHidden
          importantForAccessibility="no-hide-descendants"
          resizeMode="contain"
          source={require('../assets/auth/mirror-bear-cleaned.png')}
          style={[styles.mirrorBear, { height: illustrationWidth * (420 / 410), width: illustrationWidth }]}
        />
        <Image
          accessibilityElementsHidden
          importantForAccessibility="no-hide-descendants"
          resizeMode="contain"
          source={require('../assets/auth/botanical-right.png')}
          style={[styles.botanicalRight, { height: branchWidth * 2.06, width: branchWidth }]}
        />
      </View>
      <View accessible accessibilityLabel="Loading Reflexion" style={styles.loading}>
        <ActivityIndicator color={colors.accent} size="large" />
        <Text style={styles.loadingText}>Loading...</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { backgroundColor: colors.surface.page, flex: 1, minWidth: 0, overflow: 'hidden' },
  logo: { alignSelf: 'center', position: 'absolute', top: '28%' },
  mirrorBear: { alignSelf: 'center', position: 'absolute', top: '48%' },
  botanicalLeft: { left: -scaleSize(4), position: 'absolute', top: '13%' },
  botanicalRight: { bottom: -scaleSize(7), position: 'absolute', right: -scaleSize(5) },
  loading: { alignItems: 'center', bottom: '13%', left: 0, position: 'absolute', right: 0 },
  loadingText: { color: colors.text.primary, fontFamily: fontFamily.ui, fontSize: fontSize.heading, lineHeight: 28, marginTop: scaleSize(12) },
});
