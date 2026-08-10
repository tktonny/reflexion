import { Feather } from '@expo/vector-icons';
import React from 'react';
import { Image, StyleSheet, Text, View, type ImageSourcePropType, type StyleProp, type ViewStyle } from 'react-native';

import { colors, fontFamily, fontSize, radius, scaleSize, spacing } from '../theme';

const logoSource = require('../../assets/chat/reflexion-logo.png') as ImageSourcePropType;
const berriesSource = require('../../assets/loved/botanical-berries.png') as ImageSourcePropType;
const paleSource = require('../../assets/loved/botanical-pale.png') as ImageSourcePropType;
const sprigSource = require('../../assets/loved/botanical-sprig.png') as ImageSourcePropType;
const mumPortraitSource = require('../../assets/act02-mum-mary.png') as ImageSourcePropType;

export type LovedBotanicalVariant = 'berries' | 'pale' | 'sprig';

export function LovedBrandLockup({ compact = false }: { compact?: boolean }) {
  return (
    <Image
      accessibilityLabel="Reflexion, Care. Connected."
      accessibilityRole="image"
      resizeMode="contain"
      source={logoSource}
      style={[styles.logo, compact && styles.logoCompact]}
    />
  );
}

export function LovedBotanical({ variant = 'berries', style }: { variant?: LovedBotanicalVariant; style?: StyleProp<ViewStyle> }) {
  const source = variant === 'pale' ? paleSource : variant === 'sprig' ? sprigSource : berriesSource;
  return (
    <View accessibilityElementsHidden importantForAccessibility="no-hide-descendants" pointerEvents="none" style={[styles.botanical, style]}>
      <Image resizeMode="contain" source={source} style={styles.botanicalImage} />
    </View>
  );
}

export function LovedAvatar({ name, photoUrl, size = 96 }: { name: string; photoUrl?: string | null; size?: number }) {
  const normalized = name.toLowerCase();
  const useCanonicalPortrait = !photoUrl && (normalized.includes('mum') || normalized.includes('mary') || normalized.includes('margaret'));
  if (photoUrl || useCanonicalPortrait) {
    return <Image accessibilityLabel={`${name} photo`} source={photoUrl ? { uri: photoUrl } : mumPortraitSource} style={[styles.avatar, { height: size, width: size }]} />;
  }
  return (
    <View accessible accessibilityLabel={`${name} initials`} style={[styles.avatar, styles.initialAvatar, { height: size, width: size }]}>
      <Text style={[styles.initial, { fontSize: Math.round(size * 0.36) }]}>{name.trim().slice(0, 1).toUpperCase() || '?'}</Text>
    </View>
  );
}

export function LovedIconCircle({ icon, size = 50, tone = 'teal' }: { icon: keyof typeof Feather.glyphMap; size?: number; tone?: 'teal' | 'soft' | 'amber' }) {
  const backgroundColor = tone === 'amber' ? '#FFF0D9' : tone === 'soft' ? '#EEF5F1' : '#E3F0ED';
  return <View accessibilityElementsHidden importantForAccessibility="no-hide-descendants" style={[styles.iconCircle, { backgroundColor, height: size, width: size }]}><Feather color={tone === 'amber' ? colors.status.amber : colors.accent} name={icon} size={Math.round(size * 0.45)} /></View>;
}

export function LovedChip({ children }: { children: React.ReactNode }) {
  return <View style={styles.chip}><Text style={styles.chipText}>{children}</Text></View>;
}

export function LovedDot({ color = colors.accent }: { color?: string }) {
  return <View accessibilityElementsHidden importantForAccessibility="no-hide-descendants" style={[styles.dot, { backgroundColor: color }]} />;
}

const styles = StyleSheet.create({
  logo: { alignSelf: 'flex-start', height: scaleSize(104), marginLeft: -spacing.sm, width: scaleSize(222) },
  logoCompact: { height: scaleSize(80), width: scaleSize(176) },
  botanical: { height: scaleSize(276), position: 'absolute', right: -scaleSize(38), top: -scaleSize(28), width: scaleSize(185), zIndex: 0 },
  botanicalImage: { height: '100%', width: '100%' },
  avatar: { borderRadius: radius.pill, flexShrink: 0, overflow: 'hidden' },
  initialAvatar: { alignItems: 'center', backgroundColor: '#E5EEE9', justifyContent: 'center' },
  initial: { color: colors.accent, fontFamily: fontFamily.display, fontWeight: '500' },
  iconCircle: { alignItems: 'center', borderRadius: radius.pill, flexShrink: 0, justifyContent: 'center' },
  chip: { alignItems: 'center', backgroundColor: '#E7F1EE', borderRadius: radius.pill, minHeight: 34, paddingHorizontal: spacing.md, paddingVertical: spacing.xs },
  chipText: { color: '#155B58', fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 22 },
  dot: { borderRadius: radius.pill, height: 9, width: 9 },
});
