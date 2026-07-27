import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  claimDevicePairingV1,
  createLovedOneV1,
  loadCaregiverHome,
  registerCaregiverV1,
  updateCaregiverProfileV1,
} from '../src/lib/v1Caregiver';
import { setV1Session } from '../src/lib/v1AuthSession';
import { getStoredAuthSession, setStoredAuthSession } from '../src/lib/authSession';
import { caregiverConfigKey, refreshCaregiverConfig } from '../src/lib/queryKeys';
import {
  getPushNotificationDeviceRegistration,
  registerPushNotificationDevice,
} from '../src/lib/pushNotifications';
import { AccountStep } from '../src/screens/onboarding/AccountStep';
import { ElderlyStep } from '../src/screens/onboarding/ElderlyStep';
import { ExistingProfilesState } from '../src/screens/onboarding/ExistingProfilesState';
import { MirrorStep } from '../src/screens/onboarding/MirrorStep';
import { NotificationStep } from '../src/screens/onboarding/NotificationStep';
import { blankPatient, getStepSubtitle, validateStep } from '../src/screens/onboarding/helpers';
import type { AccountForm, NotificationForm, PatientForm } from '../src/screens/onboarding/types';
import { colors, fontSize, radius, scaleSize, spacing } from '../src/theme';

type LatestConfigResponse = {
  patients?: unknown[];
};

type CreateConfigResponse = {
  nurseId?: string;
  email?: string;
  name?: string;
  patientCount?: number;
};

type AddPatientsResponse = {
  patientCount?: number;
};

type CreateConfigVariables = {
  pushDeviceRegistration: Awaited<ReturnType<typeof getPushNotificationDeviceRegistration>> | null;
};

export default function OnboardingScreen() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { mode, returnTo } = useLocalSearchParams<{ mode?: string; returnTo?: string }>();
  const isAddPatientMode = mode === 'add-patient';
  const addPatientReturnPath = returnTo === 'settings' ? '/(tabs)/settings' : '/(tabs)';
  const storedSession = getStoredAuthSession();
  const [step, setStep] = useState(isAddPatientMode ? 2 : 1);
  const [selectedPatientIndex, setSelectedPatientIndex] = useState(0);
  const [existingPatientCount, setExistingPatientCount] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [notice, setNotice] = useState<{ type: 'success' | 'error'; message: string } | null>(null);
  const [account, setAccount] = useState<AccountForm>({
    name: '',
    email: '',
    password: '',
    phoneNumber: '',
    relationshipToElderly: 'parent',
  });
  const [patients, setPatients] = useState<PatientForm[]>([blankPatient(0)]);
  const [notifications, setNotifications] = useState<NotificationForm>({
    pushNotificationsEnabled: true,
    alertSensitivity: 'only_important_changes',
    preferredDailySummaryTime: '09:00',
  });

  const selectedPatient = patients[selectedPatientIndex];
  const displayedPatientNumber = existingPatientCount + selectedPatientIndex + 1;
  const displayedTotalPatientCount = existingPatientCount + patients.length;
  const canGoBack = isAddPatientMode || step > 1;
  const totalSteps = isAddPatientMode ? 2 : 4;
  const displayStep = isAddPatientMode ? step - 1 : step;
  const isFinalStep = step === 4 || (isAddPatientMode && step === 3);
  const nextLabel = isFinalStep ? 'Finish setup' : 'Continue';
  const stepTitle = useMemo(() => {
    if (isAddPatientMode && step === 2) return `Add elderly profile ${displayedPatientNumber}`;
    if (step === 1) return 'Account creation';
    if (step === 2) return `Elderly profile ${selectedPatientIndex + 1}`;
    if (step === 3) return 'Mirror linking';
    return 'Notification setup';
  }, [displayedPatientNumber, isAddPatientMode, selectedPatientIndex, step]);

  const existingConfigQuery = useQuery({
    enabled: isAddPatientMode && Boolean(storedSession?.userId),
    queryKey: caregiverConfigKey(storedSession?.userId),
    queryFn: loadCaregiverHome,
  });
  const { refetch: refetchExistingConfig } = existingConfigQuery;

  useFocusEffect(
    useCallback(() => {
      if (isAddPatientMode && storedSession?.userId) {
        void refetchExistingConfig();
      }
    }, [isAddPatientMode, refetchExistingConfig, storedSession?.userId]),
  );

  useEffect(() => {
    if (!existingConfigQuery.data) return;
    setExistingPatientCount(existingConfigQuery.data.patients.length);
  }, [existingConfigQuery.data]);

  // The raw error stays in the log only. What the caregiver sees is <ExistingProfilesState> below: the
  // server string itself must never become on-screen copy.
  useEffect(() => {
    if (existingConfigQuery.error) {
      console.warn('[Onboarding] load existing patient count failed', existingConfigQuery.error);
    }
  }, [existingConfigQuery.error]);

  // One loved one, from the form onto v1's three resources — patient, care plan, consent. The pairing code
  // typed here is claimed after the patient exists, since a claim needs a patientId to bind the mirror to.
  async function saveLovedOne(patient: PatientForm) {
    const created = await createLovedOneV1({
      displayName: patient.name.trim(),
      preferredLanguage: patient.preferredLanguage || 'english',
      timezone: patient.timezone || 'Asia/Singapore',
      relationshipType: account.relationshipToElderly || 'caregiver',
      profile: {
        age: Number(patient.age) || null,
        gender: patient.gender || null,
        photoUrl: patient.photoUrl || null,
        phoneNumber: patient.phoneNumber || null,
      },
      wakeTime: patient.usualWakeTime,
      topics: patient.keyTopics,
      otherTopic: patient.keyTopicsOtherText,
      speechOrHearingNotes: patient.speechOrHearingConditions,
    });

    const pairingCode = patient.mirrorPairingCode.replace(/\D/g, '');
    if (pairingCode.length === 6) {
      // A mirror that will not pair must not undo a loved one who was created successfully — they can pair
      // later from Mirror management, which is exactly what that screen is for.
      await claimDevicePairingV1({
        patientId: created.patientId,
        pairingCode,
        mirrorName: patient.mirrorName.trim() || `Mirror for ${patient.name.trim()}`,
      }).catch((error) => {
        console.warn('[Onboarding] mirror pairing failed; the loved one was still created', error);
      });
    }
    return created;
  }

  const createConfigMutation = useMutation({
    mutationFn: async (_variables: CreateConfigVariables) => {
      const registration = await registerCaregiverV1({
        name: account.name.trim(),
        email: account.email.trim().toLowerCase(),
        password: account.password,
        phoneNumber: account.phoneNumber.trim(),
        relationshipToElderly: account.relationshipToElderly || null,
      });
      // Hold the session before the next call: every write below is authorised by this token.
      await setV1Session({
        accessToken: registration.accessToken,
        refreshToken: registration.refreshToken,
        actor: registration.actor,
      });

      await updateCaregiverProfileV1({ notificationPreferences: notifications });
      // Sequential rather than parallel: each loved one is three writes, and a burst of them from a phone
      // on a Singapore connection to a China-region backend is how idempotency keys start racing.
      for (const patient of patients) {
        await saveLovedOne(patient);
      }
      return {
        nurseId: registration.actor.userId,
        name: registration.actor.name || account.name.trim(),
        email: registration.actor.email || account.email.trim().toLowerCase(),
        patientCount: patients.length,
      } satisfies CreateConfigResponse;
    },
  });

  const addPatientsMutation = useMutation({
    mutationFn: async () => {
      for (const patient of patients) {
        await saveLovedOne(patient);
      }
      return { patientCount: patients.length } satisfies AddPatientsResponse;
    },
  });

  function updatePatient(index: number, updates: Partial<PatientForm>) {
    setPatients((current) =>
      current.map((patient, patientIndex) =>
        patientIndex === index ? { ...patient, ...updates } : patient,
      ),
    );
  }

  function addPatient() {
    setPatients((current) => [...current, blankPatient(current.length)]);
    setSelectedPatientIndex(patients.length);
  }

  function removePatient(index: number) {
    if (patients.length === 1) {
      Alert.alert('One profile required', 'Add at least one elderly profile before continuing.');
      return;
    }

    setPatients((current) => current.filter((_, patientIndex) => patientIndex !== index));
    setSelectedPatientIndex((current) => {
      if (current === index) {
        return Math.max(0, index - 1);
      }

      if (current > index) {
        return current - 1;
      }

      return current;
    });
  }

  async function goNext() {
    setNotice(null);
    const validationMessage = validateStep(step, account, patients);
    if (validationMessage) {
      setNotice({ type: 'error', message: validationMessage });
      return;
    }

    if (isAddPatientMode && step === 3) {
      await appendPatients();
      return;
    }

    if (step < 4) {
      setStep((current) => current + 1);
      return;
    }

    await submit();
  }

  async function submit() {
    if (isSubmitting) return;

    setIsSubmitting(true);
    setNotice(null);
    try {
      const pushDeviceRegistration = notifications.pushNotificationsEnabled
        ? await getPushNotificationDeviceRegistration()
        : null;
      if (pushDeviceRegistration && !pushDeviceRegistration.ok) {
        console.warn('[Onboarding] push registration preparation failed', pushDeviceRegistration.reason);
      }

      const body = await createConfigMutation.mutateAsync({ pushDeviceRegistration });

      if (body?.nurseId && body?.email) {
        await setStoredAuthSession({
          userId: body.nurseId,
          name: body.name || account.name.trim(),
          email: body.email,
        });

        if (notifications.pushNotificationsEnabled && !pushDeviceRegistration?.device) {
          const registration = await registerPushNotificationDevice({ nurseId: body.nurseId });
          if (!registration.ok) {
            console.warn('[Onboarding] push registration failed', registration.reason);
          }
        }
      }

      await refreshCaregiverConfig(queryClient);
      setNotice({
        type: 'success',
        message: `Created caregiver profile with ${body.patientCount || patients.length} elderly profile${body.patientCount === 1 ? '' : 's'}.`,
      });
      setTimeout(() => router.replace('/(tabs)'), 900);
    } catch (err) {
      // The server's own words used to land here verbatim. Whatever it says, the caregiver gets calm copy
      // plus the one thing they can act on; the detail goes to the log for us.
      console.warn('[Onboarding] create caregiver profile failed', err);
      setNotice({
        type: 'error',
        message:
          'We could not save your details just now. This is usually a connection problem. Please try again — or sign in instead if you already have an account.',
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  async function appendPatients() {
    if (isSubmitting) return;

    setIsSubmitting(true);
    setNotice(null);
    try {
      const body = await addPatientsMutation.mutateAsync();

      await refreshCaregiverConfig(queryClient);
      setNotice({
        type: 'success',
        message: `Added ${body.patientCount || patients.length} loved one${body.patientCount === 1 ? '' : 's'}.`,
      });
      setTimeout(() => router.replace(addPatientReturnPath), 900);
    } catch (err) {
      console.warn('[Onboarding] add loved one failed', err);
      setNotice({
        type: 'error',
        message:
          'We could not save this profile just now. This is usually a connection problem, not something about your loved one. Nothing you typed has been lost — please try again.',
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <SafeAreaView style={styles.safe}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={styles.flex}
      >
        <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
          <View style={styles.header}>
            <Text style={styles.eyebrow}>Step {displayStep} of {totalSteps}</Text>
            <Text accessibilityRole="header" maxFontSizeMultiplier={1.3} style={styles.title}>{stepTitle}</Text>
            <Text style={styles.subtitle}>{getStepSubtitle(step, displayedTotalPatientCount)}</Text>
          </View>

          {/* The eyebrow above already says "Step 2 of 4"; unlabelled bars would just add four announcements. */}
          <View
            accessibilityElementsHidden
            importantForAccessibility="no-hide-descendants"
            style={styles.progressTrack}
          >
            {Array.from({ length: totalSteps }, (_, index) => index + 1).map((item) => (
              <View
                key={item}
                style={[styles.progressStep, item <= displayStep && styles.progressStepActive]}
              />
            ))}
          </View>

          {notice ? (
            // Announced on Android without stealing focus — the notice appears after Continue is pressed,
            // often far above the button the caregiver is still looking at.
            <View
              accessibilityLiveRegion="polite"
              accessibilityRole={notice.type === 'error' ? 'alert' : 'text'}
              style={[styles.notice, notice.type === 'success' ? styles.noticeSuccess : styles.noticeError]}
            >
              <Text style={[styles.noticeText, notice.type === 'success' ? styles.noticeSuccessText : styles.noticeErrorText]}>
                {notice.message}
              </Text>
            </View>
          ) : null}

          {isAddPatientMode ? (
            <ExistingProfilesState
              hasSession={Boolean(storedSession?.userId)}
              isLoading={existingConfigQuery.isLoading}
              hasError={Boolean(existingConfigQuery.error)}
              onRetry={() => void refetchExistingConfig()}
            />
          ) : null}

          {step === 1 ? (
            <AccountStep
              account={account}
              onSignIn={() => router.push('/sign-in')}
              setAccount={setAccount}
            />
          ) : null}

          {step === 2 ? (
            <ElderlyStep
              addPatient={addPatient}
              patient={selectedPatient}
              patientIndex={selectedPatientIndex}
              patientNumberOffset={existingPatientCount}
              patients={patients}
              removePatient={removePatient}
              selectedPatientIndex={selectedPatientIndex}
              setSelectedPatientIndex={setSelectedPatientIndex}
              updatePatient={updatePatient}
            />
          ) : null}

          {step === 3 ? (
            <MirrorStep patients={patients} updatePatient={updatePatient} />
          ) : null}

          {step === 4 ? (
            <NotificationStep notifications={notifications} setNotifications={setNotifications} />
          ) : null}
        </ScrollView>

        <View style={styles.navBar}>
          <TouchableOpacity
            accessibilityRole="button"
            accessibilityState={{ disabled: !canGoBack || isSubmitting }}
            disabled={!canGoBack || isSubmitting}
            onPress={() => {
              if (isAddPatientMode && step === 2) {
                router.replace(addPatientReturnPath);
                return;
              }

              setStep((current) => Math.max(isAddPatientMode ? 2 : 1, current - 1));
            }}
            style={[styles.backBtn, (!canGoBack || isSubmitting) && styles.disabledBtn]}
          >
            <Text style={styles.backBtnText}>{canGoBack ? 'Back' : 'Cancel'}</Text>
          </TouchableOpacity>
          <TouchableOpacity
            // While saving, the spinner replaces the only text in here, so without a label the primary
            // button of the whole funnel announces nothing at all.
            accessibilityLabel={isSubmitting ? 'Saving, please wait' : nextLabel}
            accessibilityRole="button"
            accessibilityState={{ busy: isSubmitting, disabled: isSubmitting }}
            disabled={isSubmitting}
            style={styles.nextBtn}
            onPress={goNext}
          >
            {isSubmitting ? (
              <ActivityIndicator color={colors.text.onAccent} />
            ) : (
              <Text style={styles.nextBtnText}>{nextLabel}</Text>
            )}
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  safe: { flex: 1, backgroundColor: colors.surface.page },
  content: { padding: spacing.xl, paddingBottom: scaleSize(24) },
  header: { marginBottom: spacing.lg },
  eyebrow: {
    color: colors.accent,
    // "Step 2 of 4" is orientation, not decoration — 12pt uppercase was the smallest text on the screen.
    fontSize: fontSize.caption,
    fontWeight: '800',
    textTransform: 'uppercase',
  },
  title: { color: colors.text.primary, fontSize: scaleSize(27), fontWeight: '800', marginTop: spacing.xs },
  subtitle: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: scaleSize(20), marginTop: scaleSize(6) },
  progressTrack: {
    flexDirection: 'row',
    gap: spacing.sm,
    marginBottom: scaleSize(18),
  },
  progressStep: {
    backgroundColor: colors.border.default,
    borderRadius: radius.pill,
    flex: 1,
    height: 6,
  },
  progressStepActive: { backgroundColor: colors.accent },
  notice: {
    borderRadius: radius.sm,
    borderWidth: 1,
    marginBottom: scaleSize(14),
    paddingHorizontal: scaleSize(14),
    paddingVertical: scaleSize(10),
  },
  // The success/error notice tints are this screen's own pair and have no theme token; the error text
  // reuses the brand accent.
  noticeSuccess: {
    backgroundColor: '#E6F9F0',
    borderColor: '#BFE8D2',
  },
  noticeError: {
    backgroundColor: '#F9E6EC',
    borderColor: '#E7C2CE',
  },
  noticeText: {
    fontSize: fontSize.body,
    fontWeight: '700',
    lineHeight: scaleSize(18),
  },
  noticeSuccessText: {
    color: '#1A7A4A',
  },
  noticeErrorText: {
    color: colors.accent,
  },
  navBar: {
    alignItems: 'center',
    backgroundColor: colors.surface.card,
    borderTopColor: colors.border.default,
    borderTopWidth: 1,
    flexDirection: 'row',
    gap: spacing.md,
    padding: spacing.lg,
  },
  backBtn: {
    alignItems: 'center',
    borderColor: colors.border.strong,
    borderRadius: radius.sm,
    borderWidth: 1,
    flex: 1,
    minHeight: 48,
    justifyContent: 'center',
  },
  disabledBtn: {
    opacity: 0.45,
  },
  backBtnText: {
    color: colors.text.secondary,
    fontSize: fontSize.subheading,
    fontWeight: '800',
  },
  nextBtn: {
    alignItems: 'center',
    backgroundColor: colors.accent,
    borderRadius: radius.sm,
    flex: 2,
    minHeight: 48,
    justifyContent: 'center',
  },
  nextBtnText: {
    color: colors.text.onAccent,
    fontSize: fontSize.subheading,
    fontWeight: '800',
  },
});
