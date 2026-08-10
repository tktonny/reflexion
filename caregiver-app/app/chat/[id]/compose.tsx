import { useLocalSearchParams, usePathname, useRouter } from 'expo-router';
import React, { useMemo, useState } from 'react';
import { Alert, StyleSheet, Text, View } from 'react-native';

import { PrimaryButton, ScreenLayout } from '../../../src/components/AppUI';
import { Field } from '../../../src/components/Field';
import { ChatBotanical, ChatBrandLockup, ChatRecipientCard, ChatSurfaceCard } from '../../../src/components/ChatVisuals';
import { MessageTypePicker } from '../../../src/components/MessageTypePicker';
import { MotionPressable } from '../../../src/components/Motion';
import { colors, fontFamily, fontSize, radius, spacing } from '../../../src/theme';

const QUICK_EMOJIS = ['😊', '❤️', '🤗', '🌸', '👍', '💙'];

export default function MessageComposerScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ id: string | string[] }>();
  const pathname = usePathname();
  const routeId = pathname.split('/').filter(Boolean).find((segment, index, segments) => segments[index - 1] === 'chat');
  const id = (Array.isArray(params.id) ? params.id[0] : params.id) || routeId;
  const [schedule, setSchedule] = useState<'now' | 'specific-date-and-time'>('now');
  const [message, setMessage] = useState('');
  const [scheduledFor, setScheduledFor] = useState('');
  const recipient = useMemo(() => id?.toLowerCase().includes('margaret') ? 'Mum Mary' : 'Loved one', [id]);
  const preview = () => {
    if (!message.trim()) { Alert.alert('Write a message first', 'Your message will be shown only when your loved one opens the Mirror notification.'); return; }
    if (schedule === 'specific-date-and-time' && Number.isNaN(new Date(scheduledFor).getTime())) { Alert.alert('Add a date and time', 'Use a date such as 2026-08-04T09:00.'); return; }
    router.push({ pathname: `/chat/${id}/preview`, params: { message: message.trim(), scheduledFor: schedule === 'now' ? '' : new Date(scheduledFor).toISOString() } });
  };
  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <View style={styles.hero}><ChatBotanical /><ChatBrandLockup /><Text accessibilityRole="header" style={styles.title}>New message</Text><Text style={styles.subtitle}>CHAT-03 · Text message composer</Text></View>
      <ChatRecipientCard name={recipient} onEdit={() => router.back()} />
      <ChatSurfaceCard style={styles.messageCard}><Field label="Message" multiline maxLength={500} onChangeText={setMessage} placeholder="Good morning Mum! Just checking in to see how you’re doing today." style={styles.messageInput} value={message} /><Text style={styles.counter}>{message.length}/500</Text></ChatSurfaceCard>
      <ChatSurfaceCard style={styles.quickCard}><Text style={styles.sectionLabel}>Quick emojis</Text><View style={styles.emojiRow}>{QUICK_EMOJIS.map((emoji) => <MotionPressable key={emoji} accessibilityLabel={`Add ${emoji}`} accessibilityRole="button" haptic="selection" onPress={() => setMessage((current) => `${current}${current ? ' ' : ''}${emoji}`)} style={styles.emojiButton}><Text style={styles.emoji}>{emoji}</Text></MotionPressable>)}</View></ChatSurfaceCard>
      <ChatSurfaceCard style={styles.scheduleCard}><Text style={styles.sectionLabel}>Send options</Text><ScheduleOption label="Send now" selected={schedule === 'now'} onPress={() => setSchedule('now')} /><ScheduleOption label="Schedule" detail="Choose a date and time" selected={schedule === 'specific-date-and-time'} onPress={() => setSchedule('specific-date-and-time')} showChevron />{schedule === 'specific-date-and-time' ? <Field label="Delivery date and time" autoCapitalize="none" onChangeText={setScheduledFor} placeholder="2026-08-04T09:00" value={scheduledFor} /> : null}</ChatSurfaceCard>
      <PrimaryButton label="Review message" onPress={preview} />
      <View style={styles.hiddenTypeControl}><MessageTypePicker selected="text" onSelect={(type) => { if (type === 'photo') router.push(`/chat/${id}/photo`); if (type === 'voice') router.push(`/chat/${id}/voice`); }} /></View>
    </ScreenLayout>
  );
}

function ScheduleOption({ label, detail, selected, showChevron = false, onPress }: { label: string; detail?: string; selected: boolean; showChevron?: boolean; onPress: () => void }) {
  return <MotionPressable accessibilityLabel={detail ? `${label}. ${detail}` : label} accessibilityRole="button" accessibilityState={{ selected }} onPress={onPress} style={styles.scheduleOption}><View style={[styles.radio, selected && styles.radioSelected]}>{selected ? <View style={styles.radioInner} /> : null}</View><View style={styles.optionCopy}><Text style={styles.optionLabel}>{label}</Text>{detail ? <Text style={styles.optionDetail}>{detail}</Text> : null}</View>{showChevron ? <FeatherChevron /> : null}</MotionPressable>;
}

function FeatherChevron() { return <Text accessibilityElementsHidden style={styles.chevron}>›</Text>; }

const styles = StyleSheet.create({
  content: { gap: spacing.md, minWidth: 0, paddingTop: spacing.xs },
  hero: { minHeight: 176, minWidth: 0, position: 'relative', zIndex: 1 },
  title: { color: colors.text.primary, fontFamily: fontFamily.ui, fontSize: fontSize.display, fontWeight: '700', lineHeight: 42, marginTop: spacing.sm, minWidth: 0 },
  subtitle: { color: '#405A73', fontSize: fontSize.bodyLarge, lineHeight: 24, marginTop: spacing.sm, minWidth: 0 },
  messageCard: { gap: spacing.xs, padding: spacing.lg },
  messageInput: { borderRadius: radius.xl, fontSize: fontSize.bodyLarge, lineHeight: 24, minHeight: 294, paddingHorizontal: spacing.lg, paddingTop: spacing.lg },
  counter: { alignSelf: 'flex-end', color: colors.text.secondary, fontSize: fontSize.caption, marginRight: spacing.sm },
  quickCard: { gap: spacing.md, padding: spacing.lg },
  sectionLabel: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '600', lineHeight: 24 },
  emojiRow: { flexDirection: 'row', gap: spacing.sm, minWidth: 0 },
  emojiButton: { alignItems: 'center', backgroundColor: '#F1F7F4', borderColor: '#D8E7E0', borderRadius: radius.pill, flex: 1, height: 58, justifyContent: 'center', minWidth: 0 },
  emoji: { fontSize: 25 },
  scheduleCard: { gap: spacing.sm, padding: spacing.lg },
  scheduleOption: { alignItems: 'center', borderBottomColor: colors.border.subtle, borderBottomWidth: 1, flexDirection: 'row', gap: spacing.md, minHeight: 58, minWidth: 0, paddingVertical: spacing.sm },
  radio: { alignItems: 'center', borderColor: colors.text.secondary, borderRadius: radius.pill, borderWidth: 2, height: 25, justifyContent: 'center', width: 25 },
  radioSelected: { borderColor: colors.accent },
  radioInner: { backgroundColor: colors.accent, borderRadius: radius.pill, height: 13, width: 13 },
  optionCopy: { flex: 1, flexShrink: 1, minWidth: 0 },
  optionLabel: { color: colors.text.primary, fontSize: fontSize.bodyLarge, lineHeight: 23 },
  optionDetail: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 21 },
  chevron: { color: colors.text.secondary, fontSize: 33, lineHeight: 33, paddingHorizontal: spacing.xs },
  hiddenTypeControl: { marginTop: -spacing.xs, opacity: 0.001 },
});
