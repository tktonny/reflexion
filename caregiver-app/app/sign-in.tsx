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
import { signInMessage } from '../src/lib/authMessages';
import { clearStoredAuthSession, setStoredAuthSession } from '../src/lib/authSession';
import { registerPushNotificationDevice } from '../src/lib/pushNotifications';
import { v1Login } from '../src/lib/v1Client';
import { V1ApiError } from '../src/lib/v1Errors';
import { clearV1Session } from '../src/lib/v1AuthSession';
import { clearCaregiverCache } from '../src/lib/queryKeys';
import { colors, fontFamily, fontSize, MIN_TOUCH_TARGET, radius, spacing } from '../src/theme';

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
    mutationFn: async () => {
      // v1 is now the primary session — it is what every screen reads. Legacy sign-in is kept only as a
      // SELF-HEALING fallback for an account that predates the migration: the server bridges a caregiver
      // into v1 during legacy sign-in, so falling back once and retrying v1 turns a would-be lockout into a
      // slightly slower first login. Without that fallback, anyone the migration missed could never get in.
      try {
        const session = await v1Login(email, password);
        return { userId: session.actor.userId, name: session.actor.name || '', email: session.actor.email || email };
      } catch (v1Error) {
        if (!isCredentialRejection(v1Error)) throw v1Error;
        // Wrong password, or an account with no v1 user yet. Legacy tells the two apart, and signing in
        // there creates the v1 user as a side effect.
        console.warn('[SignInScreen] v1 sign-in refused; trying the legacy bridge once', v1Error);
        await apiSend<SignInResponse>('/api/auth/sign-in', {
          method: 'POST',
          body: JSON.stringify({ email, password }),
        });
        const session = await v1Login(email, password);
        return { userId: session.actor.userId, name: session.actor.name || '', email: session.actor.email || email };
      }
    },
    onSuccess: async (body) => {
      // nurseId is retained under its old name while the legacy surface is still mounted; for a v1-native
      // account it is the v1 userId, which is what every remaining nurseId consumer needs it to be.
      await setStoredAuthSession({
        nurseId: body.userId,
        name: body.name,
        email: body.email || email.trim().toLowerCase(),
      });
      // Defensive: the previous session may have ended without a clean sign-out (app killed, or the
      // sign-up path), which would otherwise leave that caregiver's data cached under gcTime: Infinity.
      clearCaregiverCache(queryClient);
      const registration = await registerPushNotificationDevice({ nurseId: body.userId });
      if (!registration.ok) {
        console.warn('[SignInScreen] push registration failed', registration.reason);
      }
      router.replace('/(tabs)');
    },
    onError: (err) => {
      // Never the server's own text: see src/lib/authMessages.ts.
      setError(signInMessage(err));
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
    await Promise.all([clearStoredAuthSession(), clearV1Session()]);
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
            // Announced on Android: the button returns to its idle state on failure, so without a live
            // region a screen-reader user is left with no signal that the sign-in was rejected at all.
            <View accessibilityLiveRegion="polite" style={styles.errorBox}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          ) : null}

          <Text style={styles.label}>Email</Text>
          <TextInput
            accessibilityLabel="Email"
            autoCapitalize="none"
            autoComplete="email"
            keyboardType="email-address"
            onChangeText={setEmail}
            placeholder="you@email.com"
            style={styles.input}
            // textContentType lets iOS Keychain / Android autofill fill these two fields, which is the
            // difference between one tap and typing an address on a phone keyboard.
            textContentType="emailAddress"
            value={email}
          />

          <Text style={styles.label}>Password</Text>
          <TextInput
            accessibilityLabel="Password"
            autoCapitalize="none"
            autoComplete="password"
            onChangeText={setPassword}
            onSubmitEditing={signIn}
            placeholder="Password"
            secureTextEntry
            style={styles.input}
            textContentType="password"
            value={password}
          />

          <TouchableOpacity
            // The label is spelled out because the spinner replaces the visible text while signing in —
            // otherwise the button loses its name at exactly the moment someone is waiting on it.
            accessibilityLabel="Sign in"
            accessibilityRole="button"
            accessibilityState={{ busy: signInMutation.isPending, disabled: signInMutation.isPending }}
            disabled={signInMutation.isPending}
            onPress={signIn}
            style={styles.signInBtn}
          >
            {signInMutation.isPending ? (
              <ActivityIndicator color={colors.text.onAccent} />
            ) : (
              <Text style={styles.signInText}>Sign in</Text>
            )}
          </TouchableOpacity>

          <TouchableOpacity
            accessibilityRole="button"
            onPress={() => router.push('/forgot-password')}
            style={styles.signUpBtn}
          >
            <Text style={styles.signUpText}>Forgot password?</Text>
          </TouchableOpacity>

          <TouchableOpacity
            accessibilityRole="button"
            onPress={() => void goToSignUp()}
            style={styles.signUpBtn}
          >
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
    backgroundColor: colors.surface.page,
  },
  keyboard: {
    flex: 1,
    justifyContent: 'center',
    padding: 24,
  },
  card: {
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: 18,
    borderWidth: 1,
    padding: 24,
  },
  title: {
    color: colors.text.primary,
    fontFamily: fontFamily.display,
    fontSize: 34,
    fontWeight: '500',
  },
  subtitle: {
    color: colors.text.secondary,
    fontSize: 16,
    marginBottom: 24,
    marginTop: spacing.sm,
  },
  errorBox: {
    // Form-rejection red. Not a status colour (those live in src/lib/v1Status.ts) and not in the theme,
    // so it stays literal here.
    backgroundColor: colors.error.surface,
    borderColor: colors.error.border,
    borderRadius: 12,
    borderWidth: 1,
    marginBottom: 18,
    padding: spacing.md,
  },
  errorText: {
    color: colors.error.text,
    fontSize: fontSize.bodyLarge,
    lineHeight: 20,
  },
  label: {
    color: colors.text.secondary,
    fontSize: fontSize.bodyLarge,
    fontWeight: '700',
    marginBottom: spacing.sm,
    marginTop: 14,
  },
  input: {
    backgroundColor: colors.surface.input,
    borderColor: colors.border.default,
    borderRadius: 12,
    borderWidth: 1,
    color: colors.text.primary,
    fontSize: 16,
    paddingHorizontal: 14,
    paddingVertical: spacing.md,
  },
  signInBtn: {
    alignItems: 'center',
    backgroundColor: colors.accent,
    borderRadius: radius.lg,
    justifyContent: 'center',
    marginTop: 24,
    minHeight: 50,
  },
  signInText: {
    color: colors.text.onAccent,
    fontSize: 16,
    fontWeight: '700',
  },
  signUpBtn: {
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 18,
    // These two are plain text links about 20pt tall; 44pt keeps them tappable one-handed without
    // changing how they look.
    minHeight: MIN_TOUCH_TARGET,
  },
  signUpText: {
    color: colors.accent,
    fontSize: fontSize.subheading,
    fontWeight: '700',
  },
});

/**
 * True only for a definite "these credentials were not accepted" — 401, or 404 for an account v1 has never
 * heard of. Anything else (offline, 5xx, a parse failure) must not trigger the legacy retry: re-sending the
 * password to a second endpoint because the network blipped is the wrong trade.
 */
function isCredentialRejection(error: unknown): boolean {
  return error instanceof V1ApiError && (error.status === 401 || error.status === 404);
}
