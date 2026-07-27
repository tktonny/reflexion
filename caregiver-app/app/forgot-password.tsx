import { useRouter } from 'expo-router';
import { useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { v1Post } from '../src/lib/v1Client';
import { colors, fontFamily, fontSize, MIN_TOUCH_TARGET, radius, scaleSize, spacing } from '../src/theme';

// Reserved forgot-password request screen. The backend always accepts (no account enumeration); the
// reset email itself is dormant until Postmark is configured on the server (launch-time).
export default function ForgotPasswordScreen() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [sent, setSent] = useState(false);

  async function submit() {
    if (submitting || !email.trim()) return;
    setSubmitting(true);
    try {
      await v1Post('/auth/password-reset-requests', { email: email.trim() });
    } catch {
      // The request is designed to always succeed; ignore transient errors so we don't reveal accounts.
    } finally {
      setSubmitting(false);
      setSent(true);
    }
  }

  return (
    <SafeAreaView style={styles.safe}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.keyboard}>
        <View style={styles.card}>
          <Text maxFontSizeMultiplier={1.3} style={styles.title}>Reset password</Text>
          {sent ? (
            <>
              {/* The confirmation replaces the form in place, so it is announced rather than left silent. */}
              <Text accessibilityLiveRegion="polite" style={styles.subtitle}>
                If an account exists for that email, we’ve sent reset instructions. Open the link in the
                email to set a new password.
              </Text>
              <TouchableOpacity
                accessibilityRole="button"
                onPress={() => router.replace('/sign-in')}
                style={styles.primaryBtn}
              >
                <Text style={styles.primaryText}>Back to sign in</Text>
              </TouchableOpacity>
            </>
          ) : (
            <>
              <Text style={styles.subtitle}>Enter your caregiver email and we’ll send a link to reset your password.</Text>
              <Text style={styles.label}>Email</Text>
              <TextInput
                accessibilityLabel="Email"
                autoCapitalize="none"
                autoComplete="email"
                keyboardType="email-address"
                onChangeText={setEmail}
                placeholder="you@email.com"
                placeholderTextColor={colors.placeholder}
                style={styles.input}
                // Autofill matters here: this screen is reached by someone who has already forgotten
                // something, on a phone keyboard.
                textContentType="emailAddress"
                value={email}
              />
              <TouchableOpacity
                // Named explicitly because the spinner takes the visible text away while submitting.
                accessibilityLabel="Send reset link"
                accessibilityRole="button"
                accessibilityState={{ busy: submitting, disabled: submitting }}
                disabled={submitting}
                onPress={submit}
                style={styles.primaryBtn}
              >
                {submitting ? <ActivityIndicator color={colors.text.onAccent} /> : <Text style={styles.primaryText}>Send reset link</Text>}
              </TouchableOpacity>
              <TouchableOpacity
                accessibilityRole="button"
                onPress={() => router.replace('/sign-in')}
                style={styles.linkBtn}
              >
                <Text style={styles.linkText}>Back to sign in</Text>
              </TouchableOpacity>
            </>
          )}
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.surface.page },
  keyboard: { flex: 1, justifyContent: 'center', padding: scaleSize(24) },
  card: {
    backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: 18,
    borderWidth: 1, padding: scaleSize(24),
  },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: scaleSize(34), fontWeight: '500' },
  subtitle: { color: colors.text.secondary, fontSize: scaleSize(16), lineHeight: scaleSize(23), marginBottom: scaleSize(24), marginTop: spacing.sm },
  label: {
    color: colors.text.secondary, fontSize: fontSize.bodyLarge, fontWeight: '700',
    marginBottom: spacing.sm, marginTop: scaleSize(14),
  },
  input: {
    backgroundColor: colors.surface.input, borderColor: colors.border.default, borderRadius: 12, borderWidth: 1,
    color: colors.text.primary, fontSize: scaleSize(16), paddingHorizontal: scaleSize(14), paddingVertical: spacing.md,
  },
  primaryBtn: {
    alignItems: 'center', backgroundColor: colors.accent, borderRadius: radius.lg, justifyContent: 'center',
    marginTop: scaleSize(24), minHeight: scaleSize(50),
  },
  primaryText: { color: colors.text.onAccent, fontSize: scaleSize(16), fontWeight: '700' },
  // 44pt: the text link is only ~20pt tall on its own, which is an easy miss one-handed.
  linkBtn: { alignItems: 'center', justifyContent: 'center', marginTop: scaleSize(18), minHeight: MIN_TOUCH_TARGET },
  linkText: { color: colors.accent, fontSize: fontSize.subheading, fontWeight: '700' },
});
