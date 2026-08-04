import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { AppHeader, PrimaryButton, ScreenLayout } from '../../../src/components/AppUI';
import { listFamilyMessagesV1, loadCaregiverHome, type V1FamilyMessage } from '../../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, spacing } from '../../../src/theme';

export default function ChatThreadScreen() {
  const router = useRouter(); const { id } = useLocalSearchParams<{ id: string }>(); const [name, setName] = useState('Loved one'); const [messages, setMessages] = useState<V1FamilyMessage[]>([]); const [loading, setLoading] = useState(true);
  const refresh = useCallback(async () => { if (!id) return; setLoading(true); try { const [home, feed] = await Promise.all([loadCaregiverHome(), listFamilyMessagesV1(id)]); setName(home.patients.find((person) => person.patientId === id)?.displayName || 'Loved one'); setMessages(feed); } finally { setLoading(false); } }, [id]);
  useFocusEffect(useCallback(() => { void refresh(); }, [refresh]));
  return <ScreenLayout><AppHeader title={name} onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Messages for {name}</Text><Text style={styles.subtitle}>The paired Mirror shows a notification first. Your message is recorded as opened only after {name} chooses to view it.</Text>{loading ? <ActivityIndicator color={colors.accent} /> : null}{messages.map((message) => <View key={message.messageId} style={styles.outgoing}><Text style={styles.message}>{message.body}</Text><Text style={styles.status}>{messageState(message.state)} · {new Intl.DateTimeFormat('en-SG', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(message.createdAt))}</Text></View>)}{!loading && !messages.length ? <Text style={styles.empty}>No messages have been sent yet.</Text> : null}<PrimaryButton label="Send a message" onPress={() => router.push(`/chat/${id}/compose`)} /></ScreenLayout>;
}
function messageState(state: V1FamilyMessage['state']) { return ({ scheduled: 'Scheduled', queued: 'Queued', delivered: 'Delivered to device', opened: 'Opened' })[state]; }
const styles = StyleSheet.create({ title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', marginTop: spacing.xl }, subtitle: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 }, outgoing: { alignSelf: 'flex-end', backgroundColor: '#E7F3F0', borderRadius: radius.xl, maxWidth: '88%', padding: spacing.lg }, message: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, lineHeight: 23 }, status: { color: colors.status.green, flexShrink: 1, fontSize: fontSize.caption, fontWeight: '700', marginTop: spacing.sm }, empty: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body } });
