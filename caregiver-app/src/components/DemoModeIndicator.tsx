import { useRouter } from 'expo-router';
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { isDemoFeatureEnabled, useDemoMode } from '../demo/demoMode';
import { MotionPressable } from './Motion';
import { colors, fontFamily, fontSize, radius, spacing } from '../theme';

/** Small persistent development affordance; it is absent from production builds. */
export function DemoModeIndicator() {
  const router = useRouter();
  const active = useDemoMode();
  if (!isDemoFeatureEnabled() || !active) return null;
  return <View pointerEvents="box-none" style={styles.overlay}>
    <MotionPressable accessibilityLabel="Demo mode. Open demo controls." accessibilityRole="button" onPress={() => router.push('/demo')} style={styles.pill} feedback="button">
      <View style={styles.dot} />
      <Text style={styles.text}>DEMO MODE</Text>
    </MotionPressable>
  </View>;
}

const styles = StyleSheet.create({
  overlay: { alignItems: 'flex-end', left: 0, pointerEvents: 'box-none', position: 'absolute', right: 0, top: 8, zIndex: 1000 },
  pill: { alignItems: 'center', backgroundColor: '#EAF4E7', borderColor: colors.textDecorative, borderRadius: radius.pill, borderWidth: 1, flexDirection: 'row', gap: spacing.xs, marginRight: spacing.md, minHeight: 30, paddingHorizontal: spacing.md, paddingVertical: spacing.xs },
  dot: { backgroundColor: colors.textDecorative, borderRadius: radius.pill, height: 7, width: 7 },
  text: { color: colors.accent, fontFamily: fontFamily.ui, fontSize: fontSize.caption, fontWeight: '800', letterSpacing: 0.4 },
});
