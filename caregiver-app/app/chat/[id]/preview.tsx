import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useState } from 'react';
import { ActivityIndicator, Alert, StyleSheet, Text, View } from 'react-native';
import { AppHeader, PrimaryButton, ScreenLayout, SecondaryButton } from '../../../src/components/AppUI';
import { sendFamilyMessageV1 } from '../../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, spacing } from '../../../src/theme';

export default function MessagePreviewScreen() {
  const router = useRouter(); const { id, message = '', scheduledFor = '' } = useLocalSearchParams<{ id: string; message?: string; scheduledFor?: string }>(); const [sending, setSending] = useState(false);
  const send = async () => {
    if (!id || !message.trim()) return;
    setSending(true);
    try {
      const result = await sendFamilyMessageV1({ patientId: id, body: message.trim(), ...(scheduledFor ? { scheduledFor } : {}) });
      router.replace({ pathname: `/chat/${id}/status/${result.messageId}`, params: { state: result.state } });
    } catch (cause) { Alert.alert('Message was not sent', cause instanceof Error ? cause.message : 'Please try again.'); }
    finally { setSending(false); }
  };
  return <ScreenLayout><AppHeader title="Preview" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Review message</Text><View style={styles.card}><Text style={styles.label}>Delivery</Text><Text style={styles.value}>{scheduledFor ? new Intl.DateTimeFormat('en-SG', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(scheduledFor)) : 'Send now'}</Text><Text style={styles.label}>Message</Text><Text style={styles.message}>{message}</Text></View>{sending ? <ActivityIndicator color={colors.accent} /> : <PrimaryButton label={scheduledFor ? 'Schedule message' : 'Send message'} onPress={() => void send()} />}<SecondaryButton label="Keep editing" onPress={() => router.back()} /></ScreenLayout>;
}
const styles = StyleSheet.create({ title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', marginTop: spacing.xl }, card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: 4, padding: spacing.xl }, label: { color: colors.text.secondary, fontSize: fontSize.caption, fontWeight: '700', marginTop: spacing.md }, value: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge }, message: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, lineHeight: 24, marginTop: 2 } });
