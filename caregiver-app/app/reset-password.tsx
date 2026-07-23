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
      setError(err instanceof Error ? err.message : 'Unable to reset the password.');
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
            autoCapitalize="none"
            onChangeText={setPassword}
            placeholder="At least 8 characters"
            placeholderTextColor="#B7ACA1"
            secureTextEntry
            style={styles.input}
            value={password}
          />
          {error ? <Text style={styles.error}>{error}</Text> : null}
          <TouchableOpacity disabled={submitting} onPress={submit} style={styles.primaryBtn}>
            {submitting ? <ActivityIndicator color="#FFFFFF" /> : <Text style={styles.primaryText}>Set new password</Text>}
          </TouchableOpacity>
          <TouchableOpacity onPress={() => router.replace('/sign-in')} style={styles.linkBtn}>
            <Text style={styles.linkText}>Back to sign in</Text>
          </TouchableOpacity>
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
  error: { color: '#8A2E2E', fontSize: 14, marginTop: 12 },
  primaryBtn: {
    alignItems: 'center', backgroundColor: '#87566A', borderRadius: 14, justifyContent: 'center',
    marginTop: 24, minHeight: 50,
  },
  primaryText: { color: '#FFFFFF', fontSize: 16, fontWeight: '700' },
  linkBtn: { alignItems: 'center', marginTop: 18 },
  linkText: { color: '#87566A', fontSize: 15, fontWeight: '700' },
});
