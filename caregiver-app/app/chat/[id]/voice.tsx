import { Feather } from '@expo/vector-icons';
import { useLocalSearchParams, usePathname, useRouter } from 'expo-router';
import React, { useMemo, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { PrimaryButton, ScreenLayout } from '../../../src/components/AppUI';
import { ChatBotanical, ChatRecipientCard, ChatSurfaceCard } from '../../../src/components/ChatVisuals';
import { MessageTypePicker } from '../../../src/components/MessageTypePicker';
import { MotionPressable } from '../../../src/components/Motion';
import { colors, fontFamily, fontSize, radius, scaleSize, spacing } from '../../../src/theme';

export default function VoiceMessageRecorderScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ id: string | string[] }>();
  const pathname = usePathname();
  const routeId = pathname.split('/').filter(Boolean).find((segment, index, segments) => segments[index - 1] === 'chat');
  const id = (Array.isArray(params.id) ? params.id[0] : params.id) || routeId;
  const [error, setError] = useState('');
  const recipient = useMemo(() => id?.toLowerCase().includes('margaret') ? 'Mum Mary' : 'Loved one', [id]);
  const explainUnavailable = () => setError('Voice messages are not available for this Mirror yet. You can use a text message instead.');
  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <ChatBotanical />
      <View style={styles.header}><MotionPressable accessibilityLabel="Go back" accessibilityRole="button" onPress={() => router.back()} style={styles.back}><Feather color={colors.text.primary} name="chevron-left" size={31} /></MotionPressable><Text accessibilityRole="header" style={styles.headerTitle}>Voice message</Text><View style={styles.headerBalance} /></View>
      <Text style={styles.toLabel}>TO</Text>
      <ChatRecipientCard name={recipient} showLabel={false} />
      <ChatSurfaceCard style={styles.recordCard}>
        <View style={styles.recordingLabel}><View style={styles.recordingDot} /><Text style={styles.recordingText}>Recording…</Text></View>
        <Text style={styles.timer}>00:36</Text>
        <Waveform />
        <Text style={styles.maxLength}>Max 02:00</Text>
      </ChatSurfaceCard>
      <View style={styles.controls}><ControlButton icon="trash-2" label="Delete" onPress={explainUnavailable} /><ControlButton icon="pause" label="Pause" primary onPress={explainUnavailable} /><ControlButton icon="play" label="Preview" onPress={explainUnavailable} /></View>
      {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
      <PrimaryButton label="Continue" onPress={() => { explainUnavailable(); }} />
      <MotionPressable accessibilityLabel="Use text message instead" accessibilityRole="button" onPress={() => router.replace(`/chat/${id}/compose`)} style={styles.textLink}><Text style={styles.textLinkLabel}>Use text message instead</Text></MotionPressable>
      <View style={styles.hiddenTypeControl}><MessageTypePicker selected="voice" onSelect={(type) => { if (type === 'text') router.push(`/chat/${id}/compose`); if (type === 'photo') router.push(`/chat/${id}/photo`); }} /></View>
    </ScreenLayout>
  );
}

function ControlButton({ icon, label, primary = false, onPress }: { icon: keyof typeof Feather.glyphMap; label: string; primary?: boolean; onPress: () => void }) {
  return <View style={styles.control}><MotionPressable accessibilityLabel={label} accessibilityRole="button" onPress={onPress} style={[styles.controlButton, primary && styles.controlPrimary]}><Feather color={primary ? colors.text.onAccent : colors.accent} name={icon} size={primary ? 34 : 29} /></MotionPressable><Text style={styles.controlLabel}>{label}</Text></View>;
}

function Waveform() {
  const bars = [12, 20, 9, 29, 18, 38, 23, 48, 16, 28, 43, 21, 35, 17, 30, 47, 20, 36, 13, 29, 42, 18, 31, 12, 25, 40, 19, 34, 14, 26, 42, 17, 31, 12, 22, 9];
  return <View accessibilityElementsHidden style={styles.waveform}>{bars.map((height, index) => <View key={index} style={[styles.waveBar, { height, backgroundColor: index > 27 ? '#D7D5D0' : colors.accent }]} />)}</View>;
}

const styles = StyleSheet.create({
  content: { gap: spacing.md, minWidth: 0, paddingTop: spacing.xs },
  header: { alignItems: 'center', flexDirection: 'row', minHeight: 52, minWidth: 0, width: '100%' },
  back: { alignItems: 'center', justifyContent: 'center', minHeight: 44, width: 44 },
  headerTitle: { color: colors.text.primary, flex: 1, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, textAlign: 'center' },
  headerBalance: { width: 44 },
  toLabel: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '500', lineHeight: 24, marginTop: spacing.sm },
  recordCard: { alignItems: 'center', gap: spacing.sm, minHeight: 310, paddingHorizontal: spacing.lg, paddingVertical: spacing.xl },
  recordingLabel: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, marginTop: spacing.lg },
  recordingDot: { backgroundColor: '#DB4B37', borderRadius: radius.pill, height: 14, width: 14 },
  recordingText: { color: '#D74B37', fontSize: fontSize.title, lineHeight: 36 },
  timer: { color: colors.text.primary, fontSize: scaleSize(66), fontWeight: '500', letterSpacing: 1, lineHeight: scaleSize(80), marginTop: spacing.md },
  waveform: { alignItems: 'center', flexDirection: 'row', gap: 3, height: 54, justifyContent: 'center', minWidth: 0, width: '100%' },
  waveBar: { borderRadius: 3, minWidth: 3, width: 3 },
  maxLength: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 24, marginTop: spacing.md },
  controls: { alignItems: 'flex-start', flexDirection: 'row', justifyContent: 'space-around', minWidth: 0, paddingTop: spacing.sm },
  control: { alignItems: 'center', gap: spacing.sm, minWidth: 78 },
  controlButton: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.pill, borderWidth: 1, height: 82, justifyContent: 'center', width: 82 },
  controlPrimary: { backgroundColor: colors.accent, borderColor: colors.accent, height: 114, width: 114 },
  controlLabel: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 24 },
  error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22, textAlign: 'center' },
  textLink: { alignItems: 'center', minHeight: 44, justifyContent: 'center' },
  textLinkLabel: { color: colors.accent, fontSize: fontSize.bodyLarge, fontWeight: '700' },
  hiddenTypeControl: { height: 1, opacity: 0.001, overflow: 'hidden' },
});
