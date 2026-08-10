import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useLocalSearchParams, usePathname, useRouter } from 'expo-router';
import React, { useCallback, useMemo, useState } from 'react';
import { ActivityIndicator, Alert, StyleSheet, Text, View } from 'react-native';

import { PrimaryButton, SecondaryButton, ScreenLayout } from '../../../src/components/AppUI';
import { ChatAvatar, ChatBotanical, ChatBrandLockup, ChatIconCircle, ChatSurfaceCard, displayRelationship } from '../../../src/components/ChatVisuals';
import { MotionPressable } from '../../../src/components/Motion';
import { sendFamilyMessageV1, loadCaregiverHome, type CaregiverHomePatient } from '../../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, scaleSize, spacing } from '../../../src/theme';

export default function MessagePreviewScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ id: string | string[]; message?: string | string[]; scheduledFor?: string | string[] }>();
  const pathname = usePathname();
  const routeId = pathname.split('/').filter(Boolean).find((segment, index, segments) => segments[index - 1] === 'chat');
  const id = (Array.isArray(params.id) ? params.id[0] : params.id) || routeId;
  const message = Array.isArray(params.message) ? params.message[0] || '' : params.message || '';
  const scheduledFor = Array.isArray(params.scheduledFor) ? params.scheduledFor[0] || '' : params.scheduledFor || '';
  const [person, setPerson] = useState<CaregiverHomePatient | null>(null);
  const [sending, setSending] = useState(false);
  const recipient = useMemo(() => id?.toLowerCase().includes('margaret') ? 'Mum Mary' : 'Loved one', [id]);
  useFocusEffect(useCallback(() => {
    if (!id) return;
    void loadCaregiverHome().then((home) => setPerson(home.patients.find((item) => item.patientId === id) || null)).catch(() => undefined);
  }, [id]));
  const send = async () => {
    if (!id || !message.trim()) return;
    setSending(true);
    try {
      await sendFamilyMessageV1({ patientId: id, body: message.trim(), ...(scheduledFor ? { scheduledFor } : {}) });
      router.replace(`/chat/${id}`);
    } catch (cause) { Alert.alert('Message was not sent', cause instanceof Error ? cause.message : 'Please try again.'); }
    finally { setSending(false); }
  };
  const relationship = displayRelationship(recipient);
  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <View style={styles.hero}><ChatBotanical /><ChatBrandLockup /><View style={styles.titleRow}><MotionBack onPress={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Message preview</Text></View><Text style={styles.subtitle}>Review your message before sending</Text></View>
      <ChatSurfaceCard style={styles.previewCard}>
        <View style={styles.recipientRow}><ChatAvatar name={recipient} photoUrl={person?.profile.photoUrl} size={72} /><View style={styles.recipientCopy}><Text style={styles.toLabel}>To</Text><Text style={styles.recipientName}>{relationship}</Text><View style={styles.onlinePill}><View style={styles.onlineDot} /><Text style={styles.onlineText}>{person?.deviceId ? 'Device online' : 'No device paired'}</Text></View></View></View>
        <View style={styles.divider} />
        <PreviewRow icon="message-square" label="Message type" value="Text message" />
        <View style={styles.divider} />
        <View style={styles.messageSection}><Text style={styles.rowLabel}>Your message</Text><View style={styles.messageBox}><Text style={styles.messageText}>{message || 'Your message will appear here.'}</Text></View></View>
        <View style={styles.divider} />
        <PreviewRow icon="clock" label="Delivery timing" value={scheduledFor ? formatDate(scheduledFor) : 'Send now'} detail={scheduledFor ? 'The message will be queued for this time' : 'Message will be delivered immediately'} />
        <View style={styles.divider} />
        <PreviewRow icon="clock" label="Recipient time" value={`${formatTime(new Date())} (your time)`} detail={`${formatTime(new Date())} (${relationship}’s time)`} />
        <View style={styles.divider} />
        <PreviewRow icon="smartphone" label="Devices" value={person?.deviceId ? 'Reflexion Mirror' : 'No paired device'} detail={person?.deviceId ? 'Online now' : 'Pair a device to send'} />
      </ChatSurfaceCard>
      {sending ? <ActivityIndicator color={colors.accent} /> : <PrimaryButton icon="send" label={scheduledFor ? 'Schedule message' : 'Send message'} onPress={() => void send()} />}
      <SecondaryButton label="Go back and edit" onPress={() => router.back()} />
    </ScreenLayout>
  );
}

function MotionBack({ onPress }: { onPress: () => void }) { return <MotionPressable accessibilityLabel="Go back" accessibilityRole="button" onPress={onPress} style={styles.backWrap}><Feather color={colors.text.primary} name="arrow-left" size={29} /></MotionPressable>; }

function PreviewRow({ icon, label, value, detail }: { icon: keyof typeof Feather.glyphMap; label: string; value: string; detail?: string }) {
  return <View style={styles.infoRow}><ChatIconCircle icon={icon} size={56} /><View style={styles.infoCopy}><Text style={styles.rowLabel}>{label}</Text><Text style={styles.rowValue}>{value}</Text>{detail ? <Text style={styles.rowDetail}>{detail}</Text> : null}</View></View>;
}

function formatDate(value: string) { return new Intl.DateTimeFormat('en-SG', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value)); }
function formatTime(value: Date) { return new Intl.DateTimeFormat('en-SG', { hour: 'numeric', minute: '2-digit' }).format(value); }

const styles = StyleSheet.create({
  content: { gap: spacing.md, minWidth: 0, paddingTop: spacing.xs },
  hero: { minHeight: 184, minWidth: 0, position: 'relative', zIndex: 1 },
  titleRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, marginTop: spacing.sm, minWidth: 0 },
  backWrap: { alignItems: 'center', justifyContent: 'center', minHeight: 44, width: 42 },
  title: { color: colors.text.primary, flex: 1, fontFamily: fontFamily.ui, fontSize: fontSize.display, fontWeight: '700', lineHeight: 42, minWidth: 0 },
  subtitle: { color: '#405A73', fontSize: fontSize.bodyLarge, lineHeight: 24, marginLeft: scaleSize(44), marginTop: spacing.xs, minWidth: 0 },
  previewCard: { gap: spacing.lg, padding: 0 },
  recipientRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.lg, minWidth: 0, padding: spacing.lg },
  recipientCopy: { flex: 1, flexShrink: 1, gap: 2, minWidth: 0 },
  toLabel: { color: colors.text.primary, fontSize: fontSize.body, lineHeight: 20 },
  recipientName: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: scaleSize(30), fontStyle: 'italic', lineHeight: scaleSize(37) },
  onlinePill: { alignItems: 'center', alignSelf: 'flex-start', backgroundColor: '#DCEFE8', borderRadius: radius.pill, flexDirection: 'row', gap: spacing.sm, marginTop: spacing.xs, minHeight: 34, paddingHorizontal: spacing.md },
  onlineDot: { backgroundColor: '#1A966B', borderRadius: radius.pill, height: 9, width: 9 },
  onlineText: { color: colors.text.primary, fontSize: fontSize.body },
  divider: { borderTopColor: colors.border.subtle, borderTopWidth: 1 },
  infoRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.lg, minWidth: 0, paddingHorizontal: spacing.lg, paddingVertical: spacing.md },
  infoCopy: { flex: 1, flexShrink: 1, gap: spacing.xs, minWidth: 0 },
  rowLabel: { color: '#2A2F34', fontSize: fontSize.body, lineHeight: 21 },
  rowValue: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, lineHeight: 24, minWidth: 0 },
  rowDetail: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 21, minWidth: 0 },
  messageSection: { gap: spacing.sm, minWidth: 0, paddingHorizontal: spacing.lg, paddingVertical: spacing.md },
  messageBox: { backgroundColor: colors.surface.input, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, minWidth: 0, padding: spacing.lg },
  messageText: { color: colors.text.primary, fontSize: fontSize.bodyLarge, lineHeight: 25, minWidth: 0 },
});
