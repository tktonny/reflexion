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

import { apiSend } from '../src/lib/apiClient';

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
      await apiSend('/api/auth/password-reset-requests', {
        method: 'POST',
        body: JSON.stringify({ email: email.trim() }),
      });
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
          <Text style={styles.title}>Reset password</Text>
          {sent ? (
            <>
              <Text style={styles.subtitle}>
                If an account exists for that email, we’ve sent reset instructions. Open the link in the
                email to set a new password.
              </Text>
              <TouchableOpacity onPress={() => router.replace('/sign-in')} style={styles.primaryBtn}>
                <Text style={styles.primaryText}>Back to sign in</Text>
              </TouchableOpacity>
            </>
          ) : (
            <>
              <Text style={styles.subtitle}>Enter your caregiver email and we’ll send a link to reset your password.</Text>
              <Text style={styles.label}>Email</Text>
              <TextInput
                autoCapitalize="none"
                autoComplete="email"
                keyboardType="email-address"
                onChangeText={setEmail}
                placeholder="you@email.com"
                placeholderTextColor="#B7ACA1"
                style={styles.input}
                value={email}
              />
              <TouchableOpacity disabled={submitting} onPress={submit} style={styles.primaryBtn}>
                {submitting ? <ActivityIndicator color="#FFFFFF" /> : <Text style={styles.primaryText}>Send reset link</Text>}
              </TouchableOpacity>
              <TouchableOpacity onPress={() => router.replace('/sign-in')} style={styles.linkBtn}>
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
  safe: { flex: 1, backgroundColor: '#F8F3EC' },
  keyboard: { flex: 1, justifyContent: 'center', padding: 24 },
  card: { backgroundColor: '#FFFFFF', borderColor: '#E7DED2', borderRadius: 18, borderWidth: 1, padding: 24 },
  title: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 34, fontWeight: '500' },
  subtitle: { color: '#8F867D', fontSize: 16, lineHeight: 23, marginBottom: 24, marginTop: 8 },
  label: { color: '#756C64', fontSize: 14, fontWeight: '700', marginBottom: 8, marginTop: 14 },
  input: {
    backgroundColor: '#FBF8F4', borderColor: '#E7DED2', borderRadius: 12, borderWidth: 1,
    color: '#2B2522', fontSize: 16, paddingHorizontal: 14, paddingVertical: 12,
  },
  primaryBtn: {
    alignItems: 'center', backgroundColor: '#87566A', borderRadius: 14, justifyContent: 'center',
    marginTop: 24, minHeight: 50,
  },
  primaryText: { color: '#FFFFFF', fontSize: 16, fontWeight: '700' },
  linkBtn: { alignItems: 'center', marginTop: 18 },
  linkText: { color: '#87566A', fontSize: 15, fontWeight: '700' },
});
