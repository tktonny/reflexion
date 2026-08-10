import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useLocalSearchParams, usePathname, useRouter } from 'expo-router';
import React, { createElement, useCallback, useState } from 'react';
import { ActivityIndicator, Image, Linking, Platform, StyleSheet, Text, View } from 'react-native';

import { ScreenLayout } from '../../../src/components/AppUI';
import { ChatAvatar, ChatStatusBadge, messageStateCopy } from '../../../src/components/ChatVisuals';
import { MotionPressable } from '../../../src/components/Motion';
import { listFamilyMessagesV1, loadCaregiverHome, type CaregiverHomePatient, type V1FamilyMessage, type V1FamilyVoiceReply } from '../../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, scaleSize, spacing } from '../../../src/theme';

export default function ChatThreadScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ id: string | string[] }>();
  const pathname = usePathname();
  const routeId = pathname.split('/').filter(Boolean).find((segment, index, segments) => segments[index - 1] === 'chat');
  const id = (Array.isArray(params.id) ? params.id[0] : params.id) || routeId;
  const [person, setPerson] = useState<CaregiverHomePatient | null>(null);
  const [messages, setMessages] = useState<V1FamilyMessage[]>([]);
  const [loading, setLoading] = useState(true);
  const refresh = useCallback(async () => {
    if (!id) return;
    setLoading(true);
    try {
      const [home, feed] = await Promise.all([loadCaregiverHome(), listFamilyMessagesV1(id)]);
      setPerson(home.patients.find((item) => item.patientId === id) || null);
      setMessages(feed);
    } finally {
      setLoading(false);
    }
  }, [id]);
  useFocusEffect(useCallback(() => {
    void refresh();
  }, [refresh]));
  const name = person?.displayName || 'Loved one';
  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <MotionPressable accessibilityLabel="Go back" accessibilityRole="button" onPress={() => router.back()} style={styles.headerButton}>
          <Feather color={colors.text.primary} name="arrow-left" size={30} />
        </MotionPressable>
        <ChatAvatar name={name} photoUrl={person?.profile.photoUrl} size={58} />
        <View style={styles.headerCopy}>
          <Text accessibilityRole="header" numberOfLines={1} style={styles.personName}>{name}</Text>
          <View style={styles.onlineRow}><View style={styles.onlineDot} /><Text style={styles.onlineText}>{person?.deviceId ? 'Online' : 'Offline'}</Text></View>
        </View>
        <MotionPressable accessibilityLabel={`Call ${name}`} accessibilityRole="button" onPress={() => { if (person?.profile.phoneNumber) void Linking.openURL(`tel:${person.profile.phoneNumber}`); }} style={styles.headerButton}>
          <Feather color={colors.text.primary} name="phone" size={24} />
        </MotionPressable>
        <MotionPressable accessibilityLabel="More conversation options" accessibilityRole="button" onPress={() => undefined} style={styles.moreButton}>
          <Feather color={colors.text.primary} name="more-horizontal" size={27} />
        </MotionPressable>
      </View>

      <View style={styles.dayRule}><View style={styles.rule} /><Text style={styles.dayLabel}>Today</Text><View style={styles.rule} /></View>
      {loading ? <ActivityIndicator color={colors.accent} /> : null}
      <View style={styles.messageList}>
        {messages.slice().reverse().map((message) => <MessageBubble key={message.messageId} message={message} />)}
      </View>
      {!loading && !messages.length ? <Text style={styles.empty}>No messages have been sent yet.</Text> : null}
      <View style={styles.composer}>
        <MotionPressable accessibilityLabel="Add message type" accessibilityRole="button" onPress={() => router.push(`/chat/${id}/compose`)} style={styles.composerIcon}><Feather color={colors.text.primary} name="plus" size={29} /></MotionPressable>
        <MotionPressable accessibilityLabel="Type a message" accessibilityRole="button" onPress={() => router.push(`/chat/${id}/compose`)} style={styles.composerField}><Text style={styles.composerPlaceholder}>Type a message…</Text></MotionPressable>
        <MotionPressable accessibilityLabel="Send a photo" accessibilityRole="button" onPress={() => router.push(`/chat/${id}/photo`)} style={styles.composerIcon}><Feather color={colors.text.primary} name="image" size={24} /></MotionPressable>
        <MotionPressable accessibilityLabel="Send a voice message" accessibilityRole="button" onPress={() => router.push(`/chat/${id}/voice`)} style={styles.composerIcon}><Feather color={colors.text.primary} name="mic" size={26} /></MotionPressable>
        <View accessible accessibilityLabel="Emoji picker" style={styles.composerIcon}><Feather color={colors.text.primary} name="smile" size={26} /></View>
      </View>
    </ScreenLayout>
  );
}

function MessageBubble({ message }: { message: V1FamilyMessage }) {
  const outgoing = message.senderName === 'Your family' || message.senderName.toLowerCase().includes('family');
  const text = message.text || message.caption || message.body || (message.kind === 'voice' ? 'Voice message' : 'Family message');
  return (
    <View style={[styles.messageBlock, outgoing ? styles.outgoingBlock : styles.incomingBlock]}>
      <View style={[styles.bubble, outgoing ? styles.outgoingBubble : styles.incomingBubble]}>
        {message.kind === 'voice' ? <VoiceBubble outgoing={outgoing} /> : <Text style={[styles.bubbleText, outgoing ? styles.outgoingText : styles.incomingText]}>{text}</Text>}
        {message.kind === 'photo' && message.mediaUrl ? <Image accessibilityLabel="Photo message" source={{ uri: message.mediaUrl }} style={styles.messagePhoto} /> : null}
        <View style={styles.bubbleMeta}><Text style={[styles.bubbleTime, outgoing ? styles.outgoingMeta : styles.incomingMeta]}>{formatTime(message.createdAt)}</Text>{outgoing ? <CheckPair /> : null}</View>
      </View>
      <ChatStatusBadge label={message.kind === 'voice' ? (message.interactionState ? 'Played' : messageStateCopy(message.state)) : outgoing ? (message.state === 'failed' ? 'Failed' : 'Delivered to device') : 'Opened'} state={message.state} />
      {message.voiceReplies?.map((reply) => <VoiceReply key={reply.replyId} reply={reply} />)}
    </View>
  );
}

function VoiceBubble({ outgoing }: { outgoing: boolean }) {
  return <View style={styles.voiceBubble}><MotionPressable accessibilityLabel="Play voice message" accessibilityRole="button" onPress={() => undefined} style={styles.voicePlay}><Feather color={outgoing ? colors.accent : colors.text.primary} name="play" size={21} /></MotionPressable><Waveform color={outgoing ? 'rgba(255,255,255,0.82)' : '#C9C5BE'} /><Text style={[styles.voiceDuration, outgoing ? styles.outgoingText : styles.incomingText]}>0:16</Text></View>;
}

function Waveform({ color }: { color: string }) {
  return <View accessibilityElementsHidden style={styles.waveform}>{[10, 20, 14, 28, 17, 25, 12, 21, 15, 26, 11, 18, 13, 22, 10].map((height, index) => <View key={index} style={[styles.waveBar, { backgroundColor: color, height }]} />)}</View>;
}

function CheckPair() {
  return <View style={styles.checks}><Feather color={colors.text.onAccent} name="check" size={15} /><Feather color={colors.text.onAccent} name="check" size={15} style={styles.overlapCheck} /></View>;
}

function VoiceReply({ reply }: { reply: V1FamilyVoiceReply }) {
  return <View style={styles.reply}><Text style={styles.replyTitle}>{reply.senderName} replied by voice</Text><Text style={styles.replyMeta}>{reply.state} · {formatDate(reply.sentAt || reply.createdAt)}</Text>{reply.audioUrl ? <VoicePlayer url={reply.audioUrl} /> : <Text style={styles.replyUnavailable}>Audio is still becoming available.</Text>}</View>;
}

function VoicePlayer({ url }: { url: string }) {
  if (Platform.OS === 'web') return createElement('audio', { controls: true, preload: 'metadata', src: url, style: { marginTop: spacing.sm, width: '100%' } });
  return <MotionPressable accessibilityRole="button" onPress={() => void Linking.openURL(url)} style={styles.playButton}><Text style={styles.playText}>Play voice reply</Text></MotionPressable>;
}

function formatTime(value: string) { return new Intl.DateTimeFormat('en-SG', { hour: 'numeric', minute: '2-digit' }).format(new Date(value)); }
function formatDate(value: string) { return new Intl.DateTimeFormat('en-SG', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value)); }

const styles = StyleSheet.create({
  content: { gap: spacing.md, minWidth: 0, paddingTop: spacing.xs },
  header: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, minHeight: 72, minWidth: 0 },
  headerButton: { alignItems: 'center', justifyContent: 'center', minHeight: 44, width: 44 },
  moreButton: { alignItems: 'center', justifyContent: 'center', minHeight: 44, width: 38 },
  headerCopy: { flex: 1, flexShrink: 1, gap: 2, minWidth: 0 },
  personName: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.display, fontSize: scaleSize(26), fontStyle: 'italic', lineHeight: scaleSize(32), minWidth: 0 },
  onlineRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.xs },
  onlineDot: { backgroundColor: colors.accent, borderRadius: radius.pill, height: 8, width: 8 },
  onlineText: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 20 },
  dayRule: { alignItems: 'center', flexDirection: 'row', gap: spacing.md, minWidth: 0, paddingHorizontal: spacing.sm },
  rule: { borderTopColor: colors.border.default, borderTopWidth: 1, flex: 1 },
  dayLabel: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 },
  messageList: { gap: spacing.lg, minWidth: 0 },
  messageBlock: { gap: spacing.xs, maxWidth: '90%', minWidth: 0 },
  outgoingBlock: { alignSelf: 'flex-end', alignItems: 'flex-end' },
  incomingBlock: { alignSelf: 'flex-start', alignItems: 'flex-start' },
  bubble: { borderRadius: radius.xl, gap: spacing.xs, minWidth: 0, padding: spacing.md },
  outgoingBubble: { backgroundColor: colors.accent, borderBottomRightRadius: 5, maxWidth: '100%' },
  incomingBubble: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderWidth: 1, borderBottomLeftRadius: 5, maxWidth: '100%' },
  bubbleText: { flexShrink: 1, fontSize: fontSize.bodyLarge, lineHeight: 24, minWidth: 0 },
  outgoingText: { color: colors.text.onAccent },
  incomingText: { color: '#141C22' },
  bubbleMeta: { alignItems: 'center', flexDirection: 'row', gap: spacing.xs, justifyContent: 'flex-end' },
  bubbleTime: { fontSize: fontSize.caption, lineHeight: 18 },
  outgoingMeta: { color: 'rgba(255,255,255,0.9)' },
  incomingMeta: { color: colors.text.secondary },
  checks: { flexDirection: 'row', height: 16, width: 18 },
  overlapCheck: { marginLeft: -8 },
  messagePhoto: { borderRadius: radius.md, height: 168, marginTop: spacing.xs, width: 268 },
  voiceBubble: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, minWidth: 0 },
  voicePlay: { alignItems: 'center', justifyContent: 'center', minHeight: 36, width: 30 },
  waveform: { alignItems: 'center', flex: 1, flexDirection: 'row', gap: 2, height: 30, justifyContent: 'center', minWidth: 0 },
  waveBar: { borderRadius: 2, minWidth: 2, width: 2 },
  voiceDuration: { fontSize: fontSize.body, lineHeight: 20 },
  reply: { backgroundColor: colors.surface.muted, borderColor: colors.border.subtle, borderRadius: radius.lg, borderWidth: 1, gap: spacing.xs, marginTop: spacing.xs, padding: spacing.md, width: '100%' },
  replyTitle: { color: colors.text.primary, fontSize: fontSize.body, fontWeight: '700' },
  replyMeta: { color: colors.text.tertiary, fontSize: fontSize.caption },
  replyUnavailable: { color: colors.text.secondary, fontSize: fontSize.caption, marginTop: spacing.xs },
  playButton: { alignItems: 'center', alignSelf: 'flex-start', backgroundColor: colors.accent, borderRadius: radius.md, minHeight: 44, justifyContent: 'center', marginTop: spacing.sm, paddingHorizontal: spacing.md },
  playText: { color: colors.text.onAccent, fontSize: fontSize.body, fontWeight: '700' },
  empty: { color: colors.text.secondary, fontSize: fontSize.body, textAlign: 'center' },
  composer: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, flexDirection: 'row', gap: spacing.xs, minHeight: 68, minWidth: 0, paddingHorizontal: spacing.sm, paddingVertical: spacing.sm },
  composerIcon: { alignItems: 'center', justifyContent: 'center', minHeight: 44, width: 36 },
  composerField: { alignItems: 'center', backgroundColor: colors.surface.input, borderColor: colors.border.strong, borderRadius: radius.pill, borderWidth: 1, flex: 1, justifyContent: 'center', minHeight: 48, minWidth: 0, paddingHorizontal: spacing.lg },
  composerPlaceholder: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.bodyLarge, width: '100%' },
});
