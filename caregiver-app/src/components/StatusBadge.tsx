import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import type { Status } from '../data/mockData';

const COLORS: Record<Status, { bg: string; text: string; dot: string }> = {
  green:  { bg: '#F0F3ED', text: '#66735D', dot: '#66735D' },
  yellow: { bg: '#F6EFE5', text: '#B2844B', dot: '#B2844B' },
  red:    { bg: '#F3E8ED', text: '#87566A', dot: '#87566A' },
};

interface Props {
  status: Status;
  label: string;
  large?: boolean;
}

export default function StatusBadge({ status, label, large }: Props) {
  const c = COLORS[status];
  return (
    <View style={[styles.badge, { backgroundColor: c.bg }, large && styles.large]}>
      <View style={[styles.dot, { backgroundColor: c.dot }, large && styles.dotLarge]} />
      <Text style={[styles.label, { color: c.text }, large && styles.labelLarge]}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 999,
    gap: 6,
  },
  large: {
    paddingHorizontal: 14,
    paddingVertical: 7,
  },
  dot: {
    width: 7,
    height: 7,
    borderRadius: 999,
  },
  dotLarge: {
    width: 9,
    height: 9,
  },
  label: { fontSize: 12, fontWeight: '600' },
  labelLarge: { fontSize: 14, fontWeight: '600' },
});
