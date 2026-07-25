import { useLocalSearchParams, useRouter } from 'expo-router';
import { useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { apiSend } from '../src/lib/apiClient';
import { passwordResetMessage } from '../src/lib/authMessages';
import { colors, fontFamily, fontSize, MIN_TOUCH_TARGET, radius, spacing } from '../src/theme';

// Reset-completion screen. Reached from the emailed link caregiver-app://reset-password?token=... (or
// the CAREGIVER_APP_URL/reset-password?token=... web link). Sets a new password via the reserved endpoint.
export default function ResetPasswordScreen() {
  const router = useRouter();
  const { token } = useLocalSearchParams<{ token?: string }>();
  const [password, setPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  async function submit() {
    if (submitting) return;
    if (!token) { setError('This reset link is missing its token. Open the link from your email again.'); return; }
    if (password.length < 8) { setError('Password must be at least 8 characters.'); return; }
    setSubmitting(true);
    setError('');
    try {
      await apiSend('/api/auth/password-resets', {
        method: 'POST',
        body: JSON.stringify({ token, newPassword: password }),
      });
      Alert.alert('Password updated', 'You can now sign in with your new password.');
      router.replace('/sign-in');
    } catch (err) {
      // Never the server's own text — this box is a live region, so a raw 502 HTML preview would be read
      // aloud. See src/lib/authMessages.ts.
      setError(passwordResetMessage(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <SafeAreaView style={styles.safe}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.keyboard}>
        <View style={styles.card}>
          <Text style={styles.title}>New password</Text>
          <Text style={styles.subtitle}>Choose a new password for your caregiver account.</Text>
          <Text style={styles.label}>New password</Text>
          <TextInput
            accessibilityLabel="New password"
            autoCapitalize="none"
            // new-password/newPassword is what makes the keychain offer to generate and then SAVE the
            // password; with plain "password" it tries to fill the old one instead.
            autoComplete="new-password"
            onChangeText={setPassword}
            placeholder="At least 8 characters"
            placeholderTextColor={colors.placeholder}
            secureTextEntry
            style={styles.input}
            textContentType="newPassword"
            value={password}
          />
          {/* Announced on Android — the two local checks (missing token, too short) never move the screen,
              so silence is the only other feedback. */}
          {error ? <Text accessibilityLiveRegion="polite" style={styles.error}>{error}</Text> : null}
          <TouchableOpacity
            // Named explicitly because the spinner takes the visible text away while submitting.
            accessibilityLabel="Set new password"
            accessibilityRole="button"
            accessibilityState={{ busy: submitting, disabled: submitting }}
            disabled={submitting}
            onPress={submit}
            style={styles.primaryBtn}
          >
            {submitting ? <ActivityIndicator color={colors.text.onAccent} /> : <Text style={styles.primaryText}>Set new password</Text>}
          </TouchableOpacity>
          <TouchableOpacity
            accessibilityRole="button"
            onPress={() => router.replace('/sign-in')}
            style={styles.linkBtn}
          >
            <Text style={styles.linkText}>Back to sign in</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.surface.page },
  keyboard: { flex: 1, justifyContent: 'center', padding: 24 },
  card: {
    backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: 18,
    borderWidth: 1, padding: 24,
  },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: 34, fontWeight: '500' },
  subtitle: { color: colors.text.secondary, fontSize: 16, lineHeight: 23, marginBottom: 24, marginTop: spacing.sm },
  label: {
    color: colors.text.secondary, fontSize: fontSize.bodyLarge, fontWeight: '700',
    marginBottom: spacing.sm, marginTop: 14,
  },
  input: {
    backgroundColor: colors.surface.input, borderColor: colors.border.default, borderRadius: 12, borderWidth: 1,
    color: colors.text.primary, fontSize: 16, paddingHorizontal: 14, paddingVertical: spacing.md,
  },
  // Form-rejection red — not a status colour (src/lib/v1Status.ts owns those) and not in the theme.
  error: { color: colors.error.text, fontSize: fontSize.bodyLarge, lineHeight: 20, marginTop: spacing.md },
  primaryBtn: {
    alignItems: 'center', backgroundColor: colors.accent, borderRadius: radius.lg, justifyContent: 'center',
    marginTop: 24, minHeight: 50,
  },
  primaryText: { color: colors.text.onAccent, fontSize: 16, fontWeight: '700' },
  // 44pt: the text link is only ~20pt tall on its own, which is an easy miss one-handed.
  linkBtn: { alignItems: 'center', justifyContent: 'center', marginTop: 18, minHeight: MIN_TOUCH_TARGET },
  linkText: { color: colors.accent, fontSize: fontSize.subheading, fontWeight: '700' },
});
