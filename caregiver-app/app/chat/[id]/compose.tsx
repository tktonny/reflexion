import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useState } from 'react';
import { Alert, StyleSheet, Text, View } from 'react-native';
import { AppHeader, PrimaryButton, ScreenLayout, SecondaryButton } from '../../../src/components/AppUI';
import { Field } from '../../../src/components/Field';
import { colors, fontFamily, fontSize, radius, spacing } from '../../../src/theme';

export default function MessageComposerScreen() {
  const router = useRouter(); const { id } = useLocalSearchParams<{ id: string }>();
  const [schedule, setSchedule] = useState<'now' | 'specific-date-and-time'>('now'); const [message, setMessage] = useState(''); const [scheduledFor, setScheduledFor] = useState('');
  const preview = () => {
    if (!message.trim()) { Alert.alert('Write a message first', 'Your message will be shown only when your loved one opens the Mirror notification.'); return; }
    if (schedule === 'specific-date-and-time' && Number.isNaN(new Date(scheduledFor).getTime())) { Alert.alert('Add a date and time', 'Use a date such as 2026-08-04T09:00.'); return; }
    router.push({ pathname: `/chat/${id}/preview`, params: { message: message.trim(), scheduledFor: schedule === 'now' ? '' : new Date(scheduledFor).toISOString() } });
  };
  return <ScreenLayout><AppHeader title="New message" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Send a message</Text><Text style={styles.subtitle}>A notification appears on the paired Mirror. The message itself remains private until it is opened.</Text><Field label="Message" multiline onChangeText={setMessage} placeholder="Write something warm and familiar" value={message} /><Text style={styles.label}>When should it arrive?</Text><View style={styles.schedule}><SecondaryButton label="Send now" onPress={() => setSchedule('now')} /><SecondaryButton label="Choose date and time" onPress={() => setSchedule('specific-date-and-time')} /></View>{schedule === 'specific-date-and-time' ? <Field label="Delivery date and time" autoCapitalize="none" onChangeText={setScheduledFor} placeholder="2026-08-04T09:00" value={scheduledFor} /> : <Text style={styles.scheduleNote}>It will be queued for delivery now and shown when the paired Mirror checks for messages.</Text>}<PrimaryButton label="Preview message" onPress={preview} /></ScreenLayout>;
}
const styles = StyleSheet.create({ title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', marginTop: spacing.lg }, subtitle: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 }, label: { color: colors.text.primary, fontSize: fontSize.body, fontWeight: '700', marginTop: spacing.md }, schedule: { gap: spacing.sm }, scheduleNote: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.caption, lineHeight: 18 } });
