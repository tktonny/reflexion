import { Feather } from '@expo/vector-icons';
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { colors, fontFamily, scaleSize } from '../theme';

/** A small, reusable brand mark for the calm, editorial setup screens. */
export function BrandLockup({ compact = false }: { compact?: boolean }) {
  return (
    <View accessible accessibilityLabel="Reflexion, care connected" style={[styles.wrap, compact && styles.compact]}>
      <Feather name="feather" size={compact ? 19 : 24} color={colors.textDecorative} />
      <Text style={[styles.name, compact && styles.nameCompact]}>Reflexion</Text>
      <Text style={styles.tagline}>care · connected</Text>
    </View>
  );
}

export function BotanicalCorner({ side = 'right' }: { side?: 'left' | 'right' }) {
  const flip = side === 'left' ? styles.left : styles.right;
  return (
    <View accessibilityElementsHidden importantForAccessibility="no-hide-descendants" pointerEvents="none" style={[styles.botanical, flip]}>
      <Feather name="feather" size={scaleSize(52)} color="#BBC6AB" style={styles.leafOne} />
      <Feather name="feather" size={scaleSize(41)} color="#C9D0BA" style={styles.leafTwo} />
      <Feather name="feather" size={scaleSize(34)} color="#D8DCCB" style={styles.leafThree} />
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { alignItems: 'center', gap: 1 },
  compact: { alignItems: 'flex-start' },
  name: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: scaleSize(27), fontWeight: '500', lineHeight: scaleSize(32) },
  nameCompact: { fontSize: scaleSize(22), lineHeight: scaleSize(26) },
  tagline: { color: colors.accent, fontSize: scaleSize(9), fontWeight: '700', letterSpacing: 2.1, textTransform: 'lowercase' },
  botanical: { bottom: -12, height: 128, opacity: 0.14, position: 'absolute', width: 110 },
  right: { right: -18 },
  left: { left: -18, transform: [{ scaleX: -1 }] },
  leafOne: { bottom: 5, position: 'absolute', right: 7, transform: [{ rotate: '-26deg' }] },
  leafTwo: { bottom: 46, position: 'absolute', right: 39, transform: [{ rotate: '-68deg' }] },
  leafThree: { bottom: 77, position: 'absolute', right: 7, transform: [{ rotate: '-14deg' }] },
});
