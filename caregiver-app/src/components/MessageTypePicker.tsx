import { Feather } from '@expo/vector-icons';
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import type { IconName } from './AppUI';
import { MotionPressable } from './Motion';
import { colors, fontSize, radius, spacing } from '../theme';

export type MessageType = 'text' | 'photo' | 'voice';

const OPTIONS: { value: MessageType; label: string; icon: IconName }[] = [
  { value: 'text', label: 'Text', icon: 'message-square' },
  { value: 'photo', label: 'Photo', icon: 'image' },
  { value: 'voice', label: 'Voice', icon: 'mic' },
];

export function MessageTypePicker({ selected, onSelect }: { selected: MessageType; onSelect: (type: MessageType) => void }) {
  return <View accessibilityLabel="Message type" style={styles.wrap}>{OPTIONS.map((option) => { const active = option.value === selected; return <MotionPressable key={option.value} accessibilityLabel={`${option.label} message${active ? ', selected' : ''}`} accessibilityRole="button" accessibilityState={{ selected: active }} feedback="card" haptic="selection" onPress={() => onSelect(option.value)} style={[styles.option, active && styles.optionSelected]}><Feather color={active ? colors.text.onAccent : colors.accent} name={active ? 'check-circle' : option.icon} size={19} /><Text style={[styles.optionText, active && styles.optionTextSelected]}>{option.label}</Text></MotionPressable>; })}</View>;
}

const styles = StyleSheet.create({
  wrap: { flexDirection: 'row', gap: spacing.sm, minWidth: 0, width: '100%' },
  option: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.md, borderWidth: 1, flex: 1, flexDirection: 'row', gap: spacing.xs, justifyContent: 'center', minHeight: 48, minWidth: 0, paddingHorizontal: spacing.sm },
  optionSelected: { backgroundColor: colors.accent, borderColor: colors.accent },
  optionText: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.body, fontWeight: '700', minWidth: 0 },
  optionTextSelected: { color: colors.text.onAccent },
});
