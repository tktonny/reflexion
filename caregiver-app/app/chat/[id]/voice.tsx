import { useLocalSearchParams, useRouter } from 'expo-router';
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, SecondaryButton } from '../../../src/components/AppUI';
import { MessageTypePicker } from '../../../src/components/MessageTypePicker';
import { colors, fontFamily, fontSize, radius, spacing } from '../../../src/theme';

export default function VoiceMessageRecorderScreen() {
  const router = useRouter();
  const { id } = useLocalSearchParams<{ id: string }>();
  const explainUnavailable = () => router.replace(`/chat/${id}/compose`);
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Voice message" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Voice message</Text><Text style={styles.subtitle}>Voice messages can be up to two minutes and never autoplay on the Mirror.</Text><MessageTypePicker selected="voice" onSelect={(type) => { if (type === 'text') router.push(`/chat/${id}/compose`); if (type === 'photo') router.push(`/chat/${id}/photo`); }} /><View style={styles.card}><Text style={styles.cardTitle}>Voice messages are not available yet</Text><Text style={styles.copy}>You can use a text message instead. Nothing is recorded or sent from this screen.</Text></View><PrimaryButton disabled label="Voice messages unavailable" onPress={explainUnavailable} /><SecondaryButton label="Use text message instead" onPress={explainUnavailable} /></ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, subtitle: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 }, card: { backgroundColor: colors.status.greyBg, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, padding: spacing.xl }, cardTitle: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 22 }, copy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 } });
