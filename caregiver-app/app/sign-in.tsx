import React, { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
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
import { useRouter } from 'expo-router';
import { apiSend } from '../src/lib/apiClient';
import { clearStoredAuthSession, setStoredAuthSession } from '../src/lib/authSession';
import { registerPushNotificationDevice } from '../src/lib/pushNotifications';

type SignInResponse = {
  nurseId: string;
  name?: string;
  email?: string;
};

export default function SignInScreen() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const signInMutation = useMutation({
    mutationFn: () => apiSend<SignInResponse>('/api/auth/sign-in', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),
    onSuccess: async (body) => {
      await setStoredAuthSession({
        nurseId: body.nurseId,
        name: body.name || '',
        email: body.email || email.trim().toLowerCase(),
      });
      await queryClient.invalidateQueries({ queryKey: ['latestConfig'] });
      const registration = await registerPushNotificationDevice({ nurseId: body.nurseId });
      if (!registration.ok) {
        console.warn('[SignInScreen] push registration failed', registration.reason);
      }
      router.replace('/(tabs)');
    },
    onError: (err) => {
      setError(err instanceof Error ? err.message : 'Unable to sign in.');
    },
  });

  async function signIn() {
    if (signInMutation.isPending) {
      return;
    }

    setError('');
    if (!email.trim() || !password) {
      setError('Enter your email and password.');
      return;
    }

    signInMutation.mutate();
  }

  async function goToSignUp() {
    setError('');
    await clearStoredAuthSession();
    router.replace('/onboarding');
  }

  return (
    <SafeAreaView style={styles.safe}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={styles.keyboard}
      >
        <View style={styles.card}>
          <Text style={styles.title}>Sign in</Text>
          <Text style={styles.subtitle}>Use your caregiver account to continue.</Text>

          {error ? (
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          ) : null}

          <Text style={styles.label}>Email</Text>
          <TextInput
            autoCapitalize="none"
            autoComplete="email"
            keyboardType="email-address"
            onChangeText={setEmail}
            placeholder="you@email.com"
            style={styles.input}
            value={email}
          />

          <Text style={styles.label}>Password</Text>
          <TextInput
            autoCapitalize="none"
            onChangeText={setPassword}
            onSubmitEditing={signIn}
            placeholder="Password"
            secureTextEntry
            style={styles.input}
            value={password}
          />

          <TouchableOpacity disabled={signInMutation.isPending} onPress={signIn} style={styles.signInBtn}>
            {signInMutation.isPending ? (
              <ActivityIndicator color="#FFFFFF" />
            ) : (
              <Text style={styles.signInText}>Sign in</Text>
            )}
          </TouchableOpacity>

          <TouchableOpacity onPress={() => void goToSignUp()} style={styles.signUpBtn}>
            <Text style={styles.signUpText}>If you don't have an account, sign up!</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: '#F8F3EC',
  },
  keyboard: {
    flex: 1,
    justifyContent: 'center',
    padding: 24,
  },
  card: {
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 18,
    borderWidth: 1,
    padding: 24,
  },
  title: {
    color: '#2B2522',
    fontFamily: 'Georgia',
    fontSize: 34,
    fontWeight: '500',
  },
  subtitle: {
    color: '#8F867D',
    fontSize: 16,
    marginBottom: 24,
    marginTop: 8,
  },
  errorBox: {
    backgroundColor: '#FDECEC',
    borderColor: '#F2C7C7',
    borderRadius: 12,
    borderWidth: 1,
    marginBottom: 18,
    padding: 12,
  },
  errorText: {
    color: '#8A2E2E',
    fontSize: 14,
  },
  label: {
    color: '#756C64',
    fontSize: 14,
    fontWeight: '700',
    marginBottom: 8,
    marginTop: 14,
  },
  input: {
    backgroundColor: '#FBF8F4',
    borderColor: '#E7DED2',
    borderRadius: 12,
    borderWidth: 1,
    color: '#2B2522',
    fontSize: 16,
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  signInBtn: {
    alignItems: 'center',
    backgroundColor: '#87566A',
    borderRadius: 14,
    justifyContent: 'center',
    marginTop: 24,
    minHeight: 50,
  },
  signInText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
  signUpBtn: {
    alignItems: 'center',
    marginTop: 18,
  },
  signUpText: {
    color: '#87566A',
    fontSize: 15,
    fontWeight: '700',
  },
});
