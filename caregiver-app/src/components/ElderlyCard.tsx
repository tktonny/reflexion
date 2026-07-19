import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import type { ElderlyProfile, Status } from '../data/mockData';
import { getStatusLabel, getLastSeen, getInitials } from '../data/mockData';

const DOT_COLOR: Record<Status, string> = {
  green: '#66735D',
  yellow: '#B2844B',
  red: '#87566A',
};

const BORDER_COLOR: Record<Status, string> = {
  green: '#66735D',
  yellow: '#B2844B',
  red: '#87566A',
};

const AVATAR_BG: Record<Status, string> = {
  green: '#F0F3ED',
  yellow: '#F6EFE5',
  red: '#F3E8ED',
};

const AVATAR_TEXT: Record<Status, string> = {
  green: '#4A5745',
  yellow: '#7A5C30',
  red: '#6B3D50',
};

interface Props {
  profile: ElderlyProfile;
  status: Status;
  onPress: () => void;
}

export default function ElderlyCard({ profile, status, onPress }: Props) {
  const initials = getInitials(profile.nickname);
  return (
    <TouchableOpacity
      style={[styles.card, { borderLeftColor: BORDER_COLOR[status] }]}
      onPress={onPress}
      activeOpacity={0.8}
    >
      <View style={[styles.avatar, { backgroundColor: AVATAR_BG[status] }]}>
        <Text style={[styles.avatarText, { color: AVATAR_TEXT[status] }]}>{initials}</Text>
      </View>
      <View style={styles.info}>
        <Text style={styles.name}>{profile.nickname}</Text>
        <View style={styles.statusRow}>
          <View style={[styles.dot, { backgroundColor: DOT_COLOR[status] }]} />
          <Text style={[styles.statusText, { color: DOT_COLOR[status] }]}>{getStatusLabel(status)}</Text>
        </View>
        <Text style={styles.lastSeen}>{getLastSeen(profile.id)}</Text>
      </View>
      <Text style={styles.chevron}>›</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderWidth: 1,
    borderColor: '#E7DED2',
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.035,
    shadowRadius: 10,
    elevation: 2,
    gap: 12,
  },
  avatar: {
    width: 48,
    height: 48,
    borderRadius: 999,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: { fontSize: 16, fontWeight: '500', fontFamily: 'Georgia' },
  info: { flex: 1 },
  name: { fontSize: 16, fontWeight: '600', color: '#2B2522', marginBottom: 4 },
  statusRow: { flexDirection: 'row', alignItems: 'center', gap: 5, marginBottom: 2 },
  dot: { width: 7, height: 7, borderRadius: 999 },
  statusText: { fontSize: 13, fontWeight: '600' },
  lastSeen: { fontSize: 12, color: '#A69C92', marginTop: 2 },
  chevron: { fontSize: 20, color: '#C4B9AF', fontWeight: '300' },
});
