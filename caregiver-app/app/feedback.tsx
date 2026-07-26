import React, { useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TextInput, TouchableOpacity, ActivityIndicator, Alert,
  KeyboardAvoidingView, Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useMutation } from '@tanstack/react-query';
import { apiSend } from '../src/lib/apiClient';
import { getStoredAuthSession } from '../src/lib/authSession';
import { colors, spacing, radius, fontSize, fontFamily, cardShadow } from '../src/theme';

export default function FeedbackScreen() {
  const router = useRouter();
  const session = getStoredAuthSession();
  const [message, setMessage] = useState('');

  const submit = useMutation({
    mutationFn: () => apiSend<{ feedbackId: string }>('/api/feedback', {
      method: 'POST',
      body: JSON.stringify({ nurseId: session?.nurseId, message: message.trim() }),
    }),
    onSuccess: () => {
      setMessage('');
      Alert.alert('Thank you', 'Your feedback has been sent. We read every message.', [
        { text: 'OK', onPress: () => router.back() },
      ]);
    },
    onError: (error) => {
      Alert.alert('Unable to send', error instanceof Error ? error.message : 'Please try again in a moment.');
    },
  });

  const canSubmit = message.trim().length > 0 && !submit.isPending && Boolean(session?.nurseId);

  return (
    <SafeAreaView style={styles.safe}>
      <KeyboardAvoidingView style={styles.flex} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
          <Text style={styles.title}>Send feedback</Text>
          <Text style={styles.intro}>
            Tell us what is working well or what could be better. This goes straight to the Reflexion team.
          </Text>
          <View style={styles.card}>
            <TextInput
              style={styles.input}
              placeholder="Write your feedback…"
              placeholderTextColor={colors.placeholder}
              value={message}
              onChangeText={setMessage}
              multiline
              textAlignVertical="top"
              maxLength={5000}
              editable={!submit.isPending}
            />
          </View>
          <TouchableOpacity
            accessibilityRole="button"
            disabled={!canSubmit}
            onPress={() => submit.mutate()}
            style={[styles.submitBtn, !canSubmit && styles.submitBtnDisabled]}
          >
            {submit.isPending
              ? <ActivityIndicator color={colors.text.onAccent} />
              : <Text style={styles.submitText}>Send</Text>}
          </TouchableOpacity>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.surface.page },
  flex: { flex: 1 },
  content: { paddingHorizontal: spacing.xl, paddingTop: spacing.xl, paddingBottom: spacing.xxl, gap: spacing.md },
  title: { fontSize: fontSize.display, fontWeight: '500', color: colors.text.primary, fontFamily: fontFamily.display },
  intro: { fontSize: fontSize.bodyLarge, color: colors.text.secondary, lineHeight: 20 },
  card: {
    backgroundColor: colors.surface.card, borderRadius: radius.lg, padding: spacing.md,
    borderWidth: 1, borderColor: colors.border.default, ...cardShadow,
  },
  input: { minHeight: 160, fontSize: fontSize.bodyLarge, color: colors.text.primary },
  submitBtn: {
    backgroundColor: colors.accent, borderRadius: radius.lg, paddingVertical: spacing.lg,
    alignItems: 'center', justifyContent: 'center', marginTop: spacing.sm,
  },
  submitBtnDisabled: { opacity: 0.5 },
  submitText: { color: colors.text.onAccent, fontSize: fontSize.subheading, fontWeight: '700' },
});
