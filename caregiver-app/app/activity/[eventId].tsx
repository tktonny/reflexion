import { useLocalSearchParams, useRouter } from 'expo-router';
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { AppHeader, PrimaryButton, ProvenanceSection, ScreenLayout, TertiaryButton } from '../../src/components/AppUI';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';

export default function ActivityDetailScreen() {
  const { eventId, title = 'Activity detail', detail = 'This activity is shown as a factual record.', time = 'Time unavailable', patientId = '' } = useLocalSearchParams<{ eventId: string; title?: string; detail?: string; time?: string; patientId?: string }>();
  const router = useRouter();
  const decodedTitle = decodeParam(title);
  const decodedDetail = decodeParam(detail);
  const decodedTime = decodeParam(time);
  const sessionId = eventId?.startsWith('session-') ? eventId.slice('session-'.length) : '';
  return <ScreenLayout><AppHeader title="Activity" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>{decodedTitle}</Text><View style={styles.card}><ProvenanceSection label="Observed">{decodedTime}</ProvenanceSection><ProvenanceSection label="Detail">{decodedDetail}</ProvenanceSection><ProvenanceSection label="Limitations">This record reports an observed, reported or technical event. It does not make a health or safety claim.</ProvenanceSection></View>{sessionId && patientId ? <PrimaryButton label="Open session" onPress={() => router.push(`/loved-one/${patientId}/sessions/${sessionId}`)} /> : null}<TertiaryButton label="Back to activity" onPress={() => router.back()} /></ScreenLayout>;
}

function decodeParam(value: string | string[]) {
  const raw = Array.isArray(value) ? value[0] || '' : value;
  try { return decodeURIComponent(raw); } catch { return raw; }
}

const styles = StyleSheet.create({ title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', marginTop: spacing.xl }, card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, overflow: 'hidden', paddingHorizontal: spacing.lg } });
