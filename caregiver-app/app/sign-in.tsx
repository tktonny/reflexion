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
import { clearV1Session } from '../src/lib/v1AuthSession';
import { clearCaregiverCache } from '../src/lib/queryKeys';
import { colors, fontFamily, fontSize, MIN_TOUCH_TARGET, radius, scaleSize, spacing } from '../src/theme';

type SignInResponse = {
  nurseId: string;
  name?: string;
  email?: string;
};

type SignInResult = { userId: string; name: string; email: string };

/**
 * Only a rejected credential may fall through to the legacy surface. A 500, a timeout or an offline
 * device must not be retried there: the second attempt would fail for its own reason and the user would be
 * told their password was wrong.
 */
function isCredentialRejection(error: unknown): boolean {
  const status = (error as { status?: number } | null)?.status;
  return status === 401 || status === 404;
}

export default function SignInScreen() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const signInMutation = useMutation({
    mutationFn: async (): Promise<SignInResult> => {
      // v1 is primary now: every screen reads v1, so a session without a v1 token is not a usable session.
      //
      // The legacy fallback stays for one case only — an account that exists in NursePatientConfig but has
      // no v1 user yet, which the sign-in bridge repairs on the way through. It is deliberately BOUNDED:
      // only a credential rejection (401/404) falls through. A 500 or a network failure must surface as
      // itself rather than being retried against a second surface and reported as a wrong password.
      try {
        const session = await v1Login(email, password);
        return { userId: session.actor.userId, name: session.actor.name || '', email: session.actor.email || email.trim().toLowerCase() };
      } catch (v1Error) {
        if (!isCredentialRejection(v1Error)) throw v1Error;
        // Legacy sign-in bridges the account into v1; the retry then succeeds and we hold a v1 token.
        const legacy = await apiSend<SignInResponse>('/api/auth/sign-in', {
          method: 'POST',
          body: JSON.stringify({ email, password }),
        });
        const bridged = await v1Login(email, password).catch(() => null);
        return {
          userId: bridged?.actor.userId || legacy.nurseId,
          name: bridged?.actor.name || legacy.name || '',
          email: bridged?.actor.email || legacy.email || email.trim().toLowerCase(),
        };
      }
    },
    onSuccess: async (body) => {
      await setStoredAuthSession({
        userId: body.userId,
        name: body.name,
        email: body.email,
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
          <Text maxFontSizeMultiplier={1.3} style={styles.title}>Sign in</Text>
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
    padding: scaleSize(24),
  },
  card: {
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: 18,
    borderWidth: 1,
    padding: scaleSize(24),
  },
  title: {
    color: colors.text.primary,
    fontFamily: fontFamily.display,
    fontSize: scaleSize(34),
    fontWeight: '500',
  },
  subtitle: {
    color: colors.text.secondary,
    fontSize: scaleSize(16),
    marginBottom: scaleSize(24),
    marginTop: spacing.sm,
  },
  errorBox: {
    // Form-rejection red. Not a status colour (those live in src/lib/v1Status.ts) and not in the theme,
    // so it stays literal here.
    backgroundColor: colors.error.surface,
    borderColor: colors.error.border,
    borderRadius: 12,
    borderWidth: 1,
    marginBottom: scaleSize(18),
    padding: spacing.md,
  },
  errorText: {
    color: colors.error.text,
    fontSize: fontSize.bodyLarge,
    lineHeight: scaleSize(20),
  },
  label: {
    color: colors.text.secondary,
    fontSize: fontSize.bodyLarge,
    fontWeight: '700',
    marginBottom: spacing.sm,
    marginTop: scaleSize(14),
  },
  input: {
    backgroundColor: colors.surface.input,
    borderColor: colors.border.default,
    borderRadius: 12,
    borderWidth: 1,
    color: colors.text.primary,
    fontSize: scaleSize(16),
    paddingHorizontal: scaleSize(14),
    paddingVertical: spacing.md,
  },
  signInBtn: {
    alignItems: 'center',
    backgroundColor: colors.accent,
    borderRadius: radius.lg,
    justifyContent: 'center',
    marginTop: scaleSize(24),
    minHeight: scaleSize(50),
  },
  signInText: {
    color: colors.text.onAccent,
    fontSize: scaleSize(16),
    fontWeight: '700',
  },
  signUpBtn: {
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: scaleSize(18),
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
