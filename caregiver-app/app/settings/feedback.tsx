import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { ActivityIndicator, StyleSheet, Text } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout } from '../../src/components/AppUI';
import { Field } from '../../src/components/Field';
import { submitFeedbackV1 } from '../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, spacing } from '../../src/theme';

export default function Feedback() {
  const router = useRouter();
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const send = async () => { if (!message.trim()) { setError('Add feedback before sending.'); return; } setBusy(true); setError(''); try { await submitFeedbackV1(message.trim(), 'pilot_feedback'); setError('Your feedback has been sent.'); setMessage(''); } catch { setError('We could not send feedback. Check your connection and try again.'); } finally { setBusy(false); } };
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Pilot Feedback" onBack={() => router.back()}/><Text accessibilityRole="header" style={styles.title}>Pilot Feedback</Text><Text style={styles.copy}>Tell us what worked, what was unclear, or what needs attention.</Text><Field label="Your feedback" multiline value={message} onChangeText={(value) => { setMessage(value); setError(''); }} />{error ? <Text accessibilityRole="alert" style={[styles.message, error.startsWith('Your') && styles.success]}>{error}</Text> : null}{busy ? <ActivityIndicator color={colors.accent}/> : <PrimaryButton label="Send feedback" onPress={() => void send()}/>}</ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, copy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 }, message: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 }, success: { color: colors.accent } });
