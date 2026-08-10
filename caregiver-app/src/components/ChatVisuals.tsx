import { Feather } from '@expo/vector-icons';
import React from 'react';
import { Image, StyleSheet, Text, View, type ImageSourcePropType, type StyleProp, type ViewStyle } from 'react-native';

import type { V1FamilyMessage } from '../lib/v1Caregiver';
import { MotionPressable } from './Motion';
import { cardShadow, colors, fontFamily, fontSize, radius, scaleSize, spacing } from '../theme';

const logoSource = require('../../assets/chat/reflexion-logo.png') as ImageSourcePropType;
const berriesSource = require('../../assets/chat/botanical-berries.png') as ImageSourcePropType;
const budSource = require('../../assets/chat/botanical-bud.png') as ImageSourcePropType;
const pinkSource = require('../../assets/chat/botanical-pink.png') as ImageSourcePropType;
const mumPortraitSource = require('../../assets/act02-mum-mary.png') as ImageSourcePropType;

export function ChatBrandLockup({ compact = false }: { compact?: boolean }) {
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

export function ChatBotanical({ variant = 'berries', style }: { variant?: 'berries' | 'bud' | 'pink'; style?: StyleProp<ViewStyle> }) {
  const source = variant === 'bud' ? budSource : variant === 'pink' ? pinkSource : berriesSource;
  return (
    <View accessibilityElementsHidden importantForAccessibility="no-hide-descendants" pointerEvents="none" style={[styles.botanical, style]}>
      <Image resizeMode="contain" source={source} style={styles.botanicalImage} />
    </View>
  );
}

export function ChatAvatar({ name, photoUrl, size = 58 }: { name: string; photoUrl?: string | null; size?: number }) {
  const normalized = name.toLowerCase();
  const isGroup = normalized.includes('group');
  const usesCanonicalPortrait = !photoUrl && (normalized.includes('mum') || normalized.includes('mary') || normalized.includes('margaret'));
  if (isGroup) {
    return (
      <View accessible accessibilityLabel="Family group" style={[styles.avatar, styles.groupAvatar, { height: size, width: size }]}>
        <Feather color={colors.accent} name="users" size={Math.round(size * 0.48)} />
      </View>
    );
  }
  if (photoUrl || usesCanonicalPortrait) {
    return <Image accessibilityLabel={`${name} photo`} source={photoUrl ? { uri: photoUrl } : mumPortraitSource} style={[styles.avatar, { height: size, width: size }]} />;
  }
  return (
    <View accessible accessibilityLabel={name} style={[styles.avatar, styles.initialAvatar, { height: size, width: size }]}>
      <Text style={[styles.initial, { fontSize: Math.round(size * 0.38) }]}>{name.trim().slice(0, 1).toUpperCase() || '?'}</Text>
    </View>
  );
}

export function ChatSurfaceCard({ children, style }: { children: React.ReactNode; style?: StyleProp<ViewStyle> }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

export function ChatRecipientCard({ name = 'Mum', photoUrl, onEdit, showLabel = true }: { name?: string; photoUrl?: string | null; onEdit?: () => void; showLabel?: boolean }) {
  return (
    <ChatSurfaceCard style={styles.recipientCard}>
      {showLabel ? <Text style={styles.recipientLabel}>To</Text> : null}
      <View style={styles.recipientRow}>
        <ChatAvatar name={name} photoUrl={photoUrl} size={58} />
        <Text style={styles.recipientName}>{displayRelationship(name)}</Text>
        {onEdit ? <MotionPressable accessibilityLabel="Edit recipient" accessibilityRole="button" onPress={onEdit} style={styles.editButton}><Text style={styles.editText}>Edit</Text></MotionPressable> : null}
      </View>
    </ChatSurfaceCard>
  );
}

export function ChatIconCircle({ icon, size = 56 }: { icon: keyof typeof Feather.glyphMap; size?: number }) {
  return <View accessibilityElementsHidden importantForAccessibility="no-hide-descendants" style={[styles.iconCircle, { height: size, width: size }]}><Feather color={colors.accent} name={icon} size={Math.round(size * 0.43)} /></View>;
}

export function ChatStatusBadge({ state, label }: { state: V1FamilyMessage['state']; label?: string }) {
  const tone = state === 'failed' ? 'red' : state === 'delivered' || state === 'opened' ? 'blue' : 'green';
  const copy = label || messageStateCopy(state);
  return (
    <View accessible accessibilityLabel={copy} style={styles.statusRow}>
      {tone === 'blue' ? <View style={styles.checkPair}><Feather color="#3A7CC4" name="check" size={14} /><Feather color="#3A7CC4" name="check" size={14} style={styles.overlapCheck} /></View> : <View style={[styles.statusDot, tone === 'red' && styles.statusDotRed]} />}
      <Text style={[styles.statusText, tone === 'red' && styles.statusTextRed, tone === 'blue' && styles.statusTextBlue]}>{copy}</Text>
    </View>
  );
}

export function messageStateCopy(state: V1FamilyMessage['state']) {
  return ({ scheduled: 'Scheduled', queued: 'Queued', delivered: 'Delivered', opened: 'Opened', expired: 'Expired', failed: 'Failed' } as Record<V1FamilyMessage['state'], string>)[state];
}

export function displayRelationship(name: string) {
  const normalized = name.toLowerCase();
  if (normalized.includes('margaret') || normalized.includes('mary')) return 'Mum';
  if (normalized.includes('robert')) return name === 'Robert' ? 'Robert' : 'Dad';
  if (normalized.includes('joan')) return 'Grandma';
  return name || 'Loved one';
}

const styles = StyleSheet.create({
  logo: { alignSelf: 'flex-start', height: scaleSize(104), marginLeft: -spacing.sm, width: scaleSize(222) },
  logoCompact: { height: scaleSize(74), width: scaleSize(164) },
  botanical: { height: scaleSize(252), position: 'absolute', right: -scaleSize(30), top: -scaleSize(12), width: scaleSize(190), zIndex: 0 },
  botanicalImage: { height: '100%', width: '100%' },
  avatar: { borderRadius: radius.pill, flexShrink: 0, overflow: 'hidden' },
  groupAvatar: { alignItems: 'center', backgroundColor: '#E7F1EC', justifyContent: 'center' },
  initialAvatar: { alignItems: 'center', backgroundColor: '#E7F0EA', justifyContent: 'center' },
  initial: { color: colors.accent, fontFamily: fontFamily.display, fontWeight: '500' },
  card: { ...cardShadow, backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, minWidth: 0 },
  recipientCard: { gap: spacing.sm, padding: spacing.lg },
  recipientLabel: { color: colors.text.primary, fontSize: fontSize.body, fontWeight: '600', lineHeight: 20 },
  recipientRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.lg, minWidth: 0 },
  recipientName: { color: colors.text.primary, flex: 1, flexShrink: 1, fontFamily: fontFamily.display, fontSize: scaleSize(29), fontStyle: 'italic', lineHeight: scaleSize(36), minWidth: 0 },
  editButton: { alignItems: 'center', justifyContent: 'center', minHeight: 44, paddingHorizontal: spacing.xs },
  editText: { color: colors.accent, fontSize: fontSize.bodyLarge, fontWeight: '700' },
  iconCircle: { alignItems: 'center', backgroundColor: '#E5F0EC', borderRadius: radius.pill, flexShrink: 0, justifyContent: 'center' },
  statusRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.xs, minWidth: 0 },
  statusDot: { backgroundColor: '#2BA46C', borderRadius: radius.pill, height: 8, width: 8 },
  statusDotRed: { backgroundColor: colors.status.red },
  statusText: { color: '#24925E', flexShrink: 1, fontSize: fontSize.body, lineHeight: 20 },
  statusTextRed: { color: colors.status.red },
  statusTextBlue: { color: '#3A7CC4' },
  checkPair: { flexDirection: 'row', height: 16, width: 19 },
  overlapCheck: { marginLeft: -8 },
});
