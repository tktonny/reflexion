import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { AppHeader, PrimaryButton, ScreenLayout } from '../../../../src/components/AppUI';
import { listFamilyMessagesV1, type V1FamilyMessage } from '../../../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, spacing } from '../../../../src/theme';

export default function MessageStatusScreen() {
  const router = useRouter();
  const { id, messageId, state: initialState = 'queued' } = useLocalSearchParams<{ id: string; messageId: string; state?: string }>();
  const [message, setMessage] = useState<V1FamilyMessage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    if (!id || !messageId) return;
    try {
      const messages = await listFamilyMessagesV1(id);
      setMessage(messages.find((item) => item.messageId === messageId) || null);
      setError('');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Message status could not be refreshed.');
    } finally {
      setLoading(false);
    }
  }, [id, messageId]);

  useFocusEffect(useCallback(() => {
    void refresh();
    const timer = setInterval(() => void refresh(), 10_000);
    return () => clearInterval(timer);
  }, [refresh]));

  const currentState = message?.state || initialState;
  const title = currentState === 'scheduled'
    ? 'Message scheduled'
    : currentState === 'opened'
      ? 'Message opened'
      : currentState === 'delivered'
        ? 'Delivered to device'
        : 'Message queued';
  const detail = currentState === 'scheduled'
    ? 'The message will be queued at the time you chose.'
    : currentState === 'opened'
      ? 'Your loved one chose to open the message on the Mirror.'
      : currentState === 'delivered'
        ? 'The paired Mirror has retrieved the notification. It remains private until it is opened.'
        : 'The paired Mirror will receive a notification the next time it checks for messages.';

  return <ScreenLayout>
    <AppHeader title="Message status" onBack={() => router.back()} />
    <Text accessibilityRole="header" style={styles.title}>{title}</Text>
    <Text style={styles.subtitle}>{detail}</Text>
    {loading ? <ActivityIndicator color={colors.accent} /> : null}
    {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
    <View style={styles.card}><Text style={styles.label}>Current status</Text><Text style={styles.value}>{statusLabel(currentState)}</Text><Text style={styles.note}>This status is read from the paired Mirror delivery record; it is not inferred from the send action.</Text></View>
    <PrimaryButton label="Back to chat" onPress={() => router.replace(`/chat/${id}`)} />
  </ScreenLayout>;
}

function statusLabel(state: string) {
  return ({ scheduled: 'Scheduled', queued: 'Queued for the Mirror', delivered: 'Delivered to device', opened: 'Opened on the Mirror' } as Record<string, string>)[state] || 'Status unavailable';
}

const styles = StyleSheet.create({
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', marginTop: spacing.lg },
  subtitle: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 },
  error: { color: colors.error.text, flexShrink: 1, fontSize: fontSize.body, lineHeight: 21 },
  card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.sm, padding: spacing.xl },
  label: { color: colors.text.secondary, fontSize: fontSize.caption, fontWeight: '700' },
  value: { color: colors.accent, fontSize: fontSize.heading, fontWeight: '700' },
  note: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 21, marginTop: spacing.sm },
});
