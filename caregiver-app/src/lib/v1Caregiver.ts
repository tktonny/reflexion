import { getV1Url } from './apiUrl';
import { generateIdempotencyKey, v1FetchWithHeaders, v1Get, v1Post } from './v1Client';
import type { V1PatientStatus } from './v1Status';
import type { SetupCategory, SetupStatus } from '../architecture/models';

// The caregiver data layer, on v1 only.
//
// Everything here replaces a call to the legacy `nurse-patient-config/*` surface, which is tokenless —
// identity is a `nurseId` in a query string — and which the whole app depended on for accounts, the
// loved-one directory, settings, onboarding and mirror management. v1 covers all of it now, so these
// functions are the single place that knows how a screen's view is assembled from v1 resources.
//
// One shape note: the legacy API returned one fat document per caregiver, so screens read a field and
// never thought about where it came from. v1 splits the same data by ownership — the caregiver on `/me`,
// the loved one on `/patients` (with display-only fields under `profile`), how Aria talks in the care plan,
// and the mirror in `/device-assignments`. `loadCaregiverHome` does that assembly once rather than letting
// every screen re-invent the joins.

// ── Caregiver profile (replaces the caregiver half of /nurse-patient-config/latest and its settings PATCH)

export type V1AlertSensitivity = 'notify_me_about_everything' | 'only_important_changes' | 'only_urgent_alerts';
export type V1SummaryTime = '09:00' | '19:00';
export type V1NotificationTrigger =
  | 'conversation-session-summary'
  | 'no-interaction-yet-today'
  | 'repeated-missed-interactions'
  | 'recent-interaction-shorter-than-usual'
  | 'device-may-be-offline'
  | 'reminder-not-completed-or-unclear'
  | 'new-chat-reply'
  | 'weekly-summary';
export type V1SummaryFrequency = 'immediately-after-each-session' | 'daily-summary' | 'weekly-summary' | 'off';

export type V1NotificationPreferences = {
  pushNotificationsEnabled: boolean;
  alertSensitivity: V1AlertSensitivity;
  preferredDailySummaryTime: V1SummaryTime;
  summaryFrequency: V1SummaryFrequency;
  triggers: Record<V1NotificationTrigger, boolean>;
};

export type V1CaregiverProfile = {
  userId: string;
  tenantId: string;
  name: string;
  email: string;
  roles: string[];
  phoneNumber: string;
  relationshipToElderly: string | null;
  appLanguage: 'en' | 'zh';
  notificationPreferences: V1NotificationPreferences;
  /** Privacy, not notifications: whether a session's summary text is kept at all. */
  storeSessionSummaries: boolean;
};

export function getCaregiverProfileV1(): Promise<V1CaregiverProfile> {
  return v1Get<V1CaregiverProfile>('/me');
}

/** Partial by design: omitted keys are left alone, so a screen cannot blank what it did not load. */
export async function updateCaregiverProfileV1(input: {
  name?: string;
  phoneNumber?: string;
  relationshipToElderly?: string | null;
  appLanguage?: 'en' | 'zh';
  notificationPreferences?: Partial<V1NotificationPreferences>;
  storeSessionSummaries?: boolean;
}): Promise<V1CaregiverProfile> {
  return v1Patch<V1CaregiverProfile>('/me', input);
}

export function requestEmailChangeV1(email: string): Promise<{ state: 'accepted' }> {
  return v1Post('/me/email-change-requests', { email: email.trim().toLowerCase() });
}

export function confirmEmailChangeV1(code: string): Promise<V1CaregiverProfile> {
  return v1Post('/me/email-changes', { code });
}

export function requestPhoneChangeV1(phoneNumber: string): Promise<{ state: 'accepted'; phoneNumber: string }> {
  return v1Post('/me/phone-change-requests', { phoneNumber: phoneNumber.trim() });
}

export function confirmPhoneChangeV1(phoneNumber: string, code: string): Promise<V1CaregiverProfile> {
  return v1Post('/me/phone-changes', { phoneNumber: phoneNumber.trim(), code: code.trim() });
}

export function changePasswordV1(currentPassword: string, newPassword: string): Promise<{ state: 'completed' }> {
  return v1Post('/me/password-changes', { currentPassword, newPassword });
}

// ── Loved ones (replaces the patients array of /nurse-patient-config/latest, create and add-patients)

export type V1PatientProfile = {
  age: number | null;
  gender: 'male' | 'female' | 'other' | null;
  photoUrl: string | null;
  phoneNumber: string | null;
  speechSpeed: 'slow' | 'normal' | 'fast' | null;
};

export type V1PatientRecord = {
  patientId: string;
  displayName: string;
  preferredLanguage: string;
  timezone: string;
  ageBand: string | null;
  profile: V1PatientProfile;
  status: string;
  version: number;
};

export async function listPatientRecordsV1(): Promise<V1PatientRecord[]> {
  const patients = await v1Get<V1PatientRecord[]>('/patients?limit=100');
  return Array.isArray(patients) ? patients : [];
}

export function createPatientV1(input: {
  displayName: string;
  preferredLanguage: string;
  timezone: string;
  profile?: Partial<V1PatientProfile>;
  relationshipType?: string;
}): Promise<V1PatientRecord> {
  return v1Post<V1PatientRecord>('/patients', input, { idempotencyKey: generateIdempotencyKey() });
}

/**
 * Versioned: v1 requires If-Match on a patient PATCH so two devices editing the same loved one cannot
 * silently overwrite each other. Callers pass the `version` they rendered from.
 */
export function updatePatientV1(
  patientId: string,
  version: number,
  input: { displayName?: string; preferredLanguage?: string; timezone?: string; profile?: Partial<V1PatientProfile> },
): Promise<V1PatientRecord> {
  return v1Patch<V1PatientRecord>(`/patients/${encodeURIComponent(patientId)}`, input, { ifMatch: String(version) });
}

// ── How Aria talks (the care plan; these fields used to sit in the legacy patient document doing nothing)

export type V1CarePlan = {
  patientId: string;
  version: number;
  dailyRoutine: { wakeTime?: string } & Record<string, unknown>;
  communicationPreferences: {
    topics?: string[];
    otherTopic?: string;
    speechSpeed?: 'slow' | 'normal' | 'fast';
    speechOrHearingNotes?: string;
  } & Record<string, unknown>;
  safetyNotes: string | null;
};

/** Null when the loved one has no plan yet, which is normal — not an error. */
export async function getCarePlanV1(patientId: string): Promise<V1CarePlan | null> {
  try {
    return await v1Get<V1CarePlan>(`/patients/${encodeURIComponent(patientId)}/care-plan`);
  } catch (error) {
    if (isNotFound(error)) return null;
    throw error;
  }
}

/**
 * PUT, versioned. `version` is the plan's current version, or 0 when there is none — which is what v1
 * expects for a first write.
 */
export function putCarePlanV1(patientId: string, version: number, input: {
  dailyRoutine: Record<string, unknown>;
  communicationPreferences: Record<string, unknown>;
  safetyNotes?: string | null;
}): Promise<V1CarePlan> {
  return v1Put<V1CarePlan>(`/patients/${encodeURIComponent(patientId)}/care-plan`, input, {
    ifMatch: String(version),
    idempotencyKey: generateIdempotencyKey(),
  });
}

export type V1Routine = {
  routineId: string;
  patientId: string;
  name: string;
  category: 'medication' | 'meals' | 'hydration' | 'medical-appointments' | 'exercise' | 'family-events' | 'custom-other';
  schedule: { timezone: string; times: string[]; recurrence: string };
  notificationPolicy: 'do-not-notify' | 'after-one-missed-or-unclear-response' | 'daily-summary';
  notes: string | null;
  status: 'active' | 'paused' | 'ended';
  version: number;
};

export function listRoutinesV1(patientId: string): Promise<V1Routine[]> {
  return v1Get(`/patients/${encodeURIComponent(patientId)}/routines`);
}

export function createRoutineV1(patientId: string, input: {
  name: string;
  category: V1Routine['category'];
  schedule: V1Routine['schedule'];
  notificationPolicy: V1Routine['notificationPolicy'];
  notes?: string;
}): Promise<V1Routine> {
  return v1Post(`/patients/${encodeURIComponent(patientId)}/routines`, input, { idempotencyKey: generateIdempotencyKey() });
}

export function updateRoutineV1(routine: V1Routine, input: Partial<Pick<V1Routine, 'name' | 'category' | 'schedule' | 'notificationPolicy' | 'notes' | 'status'>>): Promise<V1Routine> {
  return v1Patch(`/routines/${encodeURIComponent(routine.routineId)}`, input, { ifMatch: String(routine.version) });
}

export function endRoutineV1(routineId: string): Promise<{ routineId: string; state: 'ended' }> {
  return v1Delete(`/routines/${encodeURIComponent(routineId)}`, { idempotencyKey: generateIdempotencyKey() });
}

// ── Consent (a HARD gate: without it POST /sessions refuses a daily check-in, so no check-in can run)

export type V1ConsentState = {
  patientId: string;
  consents: { consentId: string; purpose: string; documentVersion: string; status: string; signedAt: string | null; withdrawnAt: string | null }[];
  requiredPurposes: string[];
  /** Non-empty means daily check-ins cannot run for this loved one until consent is given. */
  missingPurposes: string[];
};

export function getConsentStateV1(patientId: string): Promise<V1ConsentState> {
  return v1Get<V1ConsentState>(`/patients/${encodeURIComponent(patientId)}/consents`);
}

/** The purpose POST /sessions checks before it will start a daily check-in. */
export const CHECKIN_CONSENT_PURPOSE = 'home_cognitive_monitoring';

/** The document version the onboarding consent screen presents. Bump when that wording changes. */
export const CHECKIN_CONSENT_DOCUMENT_VERSION = 'checkin-consent-2026-07';

export function withdrawCheckInConsentV1(patientId: string, purpose = CHECKIN_CONSENT_PURPOSE): Promise<unknown> {
  return v1Post(`/patients/${encodeURIComponent(patientId)}/consents`, {
    purpose,
    documentVersion: CHECKIN_CONSENT_DOCUMENT_VERSION,
    status: 'withdrawn',
  }, { idempotencyKey: generateIdempotencyKey() });
}

export const RESEARCH_CONSENT_PURPOSE = 'optional_research_participation';
export const RESEARCH_CONSENT_DOCUMENT_VERSION = 'research-consent-2026-07';

export function grantResearchParticipationV1(patientId: string): Promise<unknown> {
  return v1Post(`/patients/${encodeURIComponent(patientId)}/consents`, { purpose: RESEARCH_CONSENT_PURPOSE, documentVersion: RESEARCH_CONSENT_DOCUMENT_VERSION, status: 'granted' }, { idempotencyKey: generateIdempotencyKey() });
}

export function withdrawResearchParticipationV1(patientId: string): Promise<unknown> {
  return v1Post(`/patients/${encodeURIComponent(patientId)}/consents`, { purpose: RESEARCH_CONSENT_PURPOSE, documentVersion: RESEARCH_CONSENT_DOCUMENT_VERSION, status: 'withdrawn' }, { idempotencyKey: generateIdempotencyKey() });
}

export type V1PrivacyState = {
  patientId: string;
  consent: { status: 'pending' | 'accepted' | 'declined' | 'withdrawn'; requiredPurpose: string; history: { consentId: string; purpose: string; documentVersion: string; status: string; signedAt: string | null; withdrawnAt: string | null }[] };
  research: { status: 'separate'; message: string };
  retention: { structuredData: string; sessionMedia: string; operationalLogs: string; configuredByServer: boolean };
  deletionCategories: { category: 'sessions' | 'messages' | 'routine-responses' | 'device-events'; label: string }[];
  deletionRequests: { requestId: string; categories: string[]; state: string; createdAt: string; updatedAt: string; remainingObjectKeys: string[]; error: string | null }[];
};

export function getPrivacyStateV1(patientId: string): Promise<V1PrivacyState> {
  return v1Get(`/patients/${encodeURIComponent(patientId)}/privacy`);
}

export function requestDataDeletionV1(patientId: string, categories: V1PrivacyState['deletionCategories'][number]['category'][]): Promise<V1PrivacyState['deletionRequests'][number]> {
  return v1Post(`/patients/${encodeURIComponent(patientId)}/data-deletion-requests`, { confirm: true, categories }, { idempotencyKey: generateIdempotencyKey() });
}

export type V1CareCircleMember = { memberId: string; kind: 'member' | 'invitation'; userId?: string; name?: string; email?: string | null; phoneNumber?: string | null; invitee?: string; role: 'full-access' | 'standard-access' | 'view-only' | 'custom-access'; permissions: string[]; state: string; version: number; createdAt?: string };
export type V1CareCircle = { patientId: string; members: V1CareCircleMember[]; invitations: V1CareCircleMember[] };

export function getCareCircleV1(patientId: string): Promise<V1CareCircle> {
  return v1Get(`/patients/${encodeURIComponent(patientId)}/care-circle`);
}

export function inviteCaregiverV1(patientId: string, input: { emailOrPhone: string; role: V1CareCircleMember['role']; permissions?: string[] }): Promise<V1CareCircleMember> {
  return v1Post(`/patients/${encodeURIComponent(patientId)}/care-circle/invitations`, input, { idempotencyKey: generateIdempotencyKey() });
}

export function updateCareCircleMemberV1(patientId: string, member: V1CareCircleMember, input: { role: V1CareCircleMember['role']; permissions?: string[] }): Promise<V1CareCircleMember> {
  return v1Write('PATCH', `/patients/${encodeURIComponent(patientId)}/care-circle/${encodeURIComponent(member.memberId)}`, input, { ifMatch: String(member.version) });
}

export function revokeCareCircleMemberV1(patientId: string, memberId: string): Promise<{ memberId: string; state: 'revoked' }> {
  return v1Delete(`/patients/${encodeURIComponent(patientId)}/care-circle/${encodeURIComponent(memberId)}`, { idempotencyKey: generateIdempotencyKey() });
}

// ── Mirrors (replaces /nurse-patient-config/mirrors and mirrors/connect)

export type V1DeviceAssignment = {
  patientId: string;
  patientName: string;
  /** Denormalised from the patient record so the pairing screen can seed its field without a second call. */
  timezone: string;
  assignmentId: string | null;
  deviceId: string | null;
  mirrorName: string | null;
  assignedAt: string | null;
  device: { serial: string | null; softwareVersion: string | null; status: string | null; technicalState: 'ok' | 'possible_issue' | 'unknown'; lastHeartbeatAt: string | null } | null;
};

export async function listDeviceAssignmentsV1(): Promise<V1DeviceAssignment[]> {
  const body = await v1Get<{ assignments: V1DeviceAssignment[] }>('/device-assignments');
  return Array.isArray(body?.assignments) ? body.assignments : [];
}

/** Claims the 6-digit code the mirror is displaying. Replaces legacy mirrors/connect. */
export function claimDevicePairingV1(input: { patientId: string; pairingCode: string; mirrorName?: string }): Promise<unknown> {
  return v1Post('/device-pairing-claims', input, { idempotencyKey: generateIdempotencyKey() });
}

export function revokeDeviceV1(deviceId: string, reason: string): Promise<unknown> {
  return v1Post(`/devices/${encodeURIComponent(deviceId)}/revocations`, { reason }, {
    idempotencyKey: generateIdempotencyKey(),
  });
}

// ── Family messages (caregiver → paired Mirror, one-way)

export type V1FamilyMessage = {
  messageId: string;
  patientId: string;
  body: string;
  type: 'text';
  state: 'scheduled' | 'queued' | 'delivered' | 'opened';
  scheduledFor: string;
  createdAt: string;
  deliveredAt: string | null;
  openedAt: string | null;
};

export function sendFamilyMessageV1(input: { patientId: string; body: string; scheduledFor?: string }): Promise<V1FamilyMessage> {
  return v1Post<V1FamilyMessage>(`/patients/${encodeURIComponent(input.patientId)}/family-messages`, {
    body: input.body,
    ...(input.scheduledFor ? { scheduledFor: input.scheduledFor } : {}),
  }, { idempotencyKey: generateIdempotencyKey() });
}

export async function listFamilyMessagesV1(patientId: string): Promise<V1FamilyMessage[]> {
  const body = await v1Get<{ messages: V1FamilyMessage[] }>(`/patients/${encodeURIComponent(patientId)}/family-messages`);
  return Array.isArray(body?.messages) ? body.messages : [];
}

// ── Session history (replaces /conversation-session-counts, -by-day, /patient-trend, /patient-summary)

export type V1SessionDay = { date: string; day: number; count: number; completedCount: number; hasCompletedSession: boolean };

export async function listSessionDaysV1(patientId: string, month: string): Promise<V1SessionDay[]> {
  const body = await v1Get<{ days: V1SessionDay[] }>(
    `/patients/${encodeURIComponent(patientId)}/session-days?month=${encodeURIComponent(month)}`,
  );
  return Array.isArray(body?.days) ? body.days : [];
}

export type V1SessionLog = { sentence: string; role: string; words: number; duration: number; wordsPerSecond: number };
export type V1SessionDetail = {
  id: string;
  patientId: string;
  patientName: string;
  type: 'daily_checkin' | 'companion' | string | null;
  state: string | null;
  duration: number;
  words: number;
  exchanges: number;
  avgLatency: number;
  createdAt: string | null;
  updatedAt: string | null;
  logs: V1SessionLog[];
};

export function getSessionDayV1(patientId: string, date: string): Promise<{ patientId: string; date: string; patientName: string; sessions: V1SessionDetail[] }> {
  return v1Get(`/patients/${encodeURIComponent(patientId)}/session-days/${encodeURIComponent(date)}`);
}

export type V1SessionFeed = {
  patientId: string;
  patientName: string;
  sessions: V1SessionDetail[];
  nextBefore: string | null;
};

/** Recent, transcript-backed sessions for the caregiver Sessions screen. */
export function listSessionsV1(patientId: string, options?: { limit?: number; before?: string }): Promise<V1SessionFeed> {
  const params = new URLSearchParams();
  if (options?.limit !== undefined) params.set('limit', String(options.limit));
  if (options?.before) params.set('before', options.before);
  const query = params.toString();
  return v1Get(`/patients/${encodeURIComponent(patientId)}/sessions${query ? `?${query}` : ''}`);
}

/** One session and its materialized transcript, authorized by the server for this caregiver. */
export function getSessionV1(patientId: string, sessionId: string): Promise<V1SessionDetail> {
  return v1Get(`/patients/${encodeURIComponent(patientId)}/sessions/${encodeURIComponent(sessionId)}`);
}

export type V1SessionProcessingStatus = {
  sessionId: string;
  operationId: string | null;
  state: 'accepted' | 'queued' | 'processing' | 'completed' | 'failed';
  stage: string;
  retryable: boolean;
  result: Record<string, unknown> | null;
  updatedAt: string;
};

export function getSessionProcessingStatusV1(sessionId: string): Promise<V1SessionProcessingStatus> {
  return v1Get(`/sessions/${encodeURIComponent(sessionId)}/processing-status`);
}

export type V1TrendDay = { date: string; duration: number; sessionCount: number; status: 'green' | 'amber' | 'red' | null; missed: boolean };

export async function getSessionTrendV1(patientId: string, days: 7 | 30): Promise<V1TrendDay[]> {
  const body = await v1Get<{ trend: V1TrendDay[] }>(
    `/patients/${encodeURIComponent(patientId)}/session-trend?days=${days}`,
  );
  return Array.isArray(body?.trend) ? body.trend : [];
}

/** `summary` is null with `reason: 'no_transcript'` on a quiet day — a normal outcome, not a failure. */
export function generateSessionSummaryV1(patientId: string, date?: string): Promise<{ patientId: string; date: string; summary: string | null; reason: string | null }> {
  return v1Post(`/patients/${encodeURIComponent(patientId)}/session-summaries`, date ? { date } : {}, {
    idempotencyKey: generateIdempotencyKey(),
  });
}

// ── The settings view: the home assembly plus each loved one's care plan

export type CaregiverSettingsPatient = CaregiverHomePatient & {
  /** null when this loved one has no plan yet; putCarePlanV1 takes 0 as the version for a first write. */
  carePlan: V1CarePlan | null;
};

export type CaregiverSettings = { caregiver: V1CaregiverProfile; patients: CaregiverSettingsPatient[] };

/**
 * Settings edits two resources per loved one — the patient record and the care plan — and both writes are
 * versioned, so the screen has to render from the versions it will send back. Loading them together is what
 * makes that possible; fetching the plan lazily when a row is opened would race an edit made on another
 * device between the list load and the save.
 *
 * A plan that fails to load is null rather than fatal: the loved one is still editable, and a first
 * putCarePlanV1 writes version 1.
 */
export async function loadCaregiverSettings(): Promise<CaregiverSettings> {
  const home = await loadCaregiverHome();
  const plans = await Promise.all(home.patients.map((patient) =>
    getCarePlanV1(patient.patientId).catch(() => null),
  ));
  return {
    caregiver: home.caregiver,
    patients: home.patients.map((patient, index) => ({ ...patient, carePlan: plans[index] })),
  };
}

// ── Onboarding (replaces /nurse-patient-config/create and add-patients)

export type V1Registration = {
  state: 'verification_pending' | 'authenticated';
  email: string;
  emailVerified?: boolean;
  accessToken?: string;
  refreshToken?: string;
  accessTokenExpiresAt?: string;
  refreshTokenExpiresAt?: string;
  actor?: V1VerifiedAccount['actor'];
};

export type V1VerifiedAccount = {
  state: 'verified';
  actor: { userId: string; tenantId: string; name?: string; email?: string; roles?: string[] };
  accessToken: string;
  refreshToken: string;
};

/** Creates the caregiver; the server decides whether this enters verification or an authenticated setup. */
export function registerCaregiverV1(input: {
  name: string;
  email: string;
  password: string;
  phoneNumber?: string;
  relationshipToElderly?: string | null;
}): Promise<V1Registration> {
  return v1Post<V1Registration>('/auth/registrations', input);
}

export function resendAccountVerificationV1(email: string): Promise<{ state: 'accepted' }> {
  return v1Post('/auth/account-verification-requests', { email: email.trim().toLowerCase() });
}

export function verifyAccountV1(email: string, code: string): Promise<V1VerifiedAccount> {
  return v1Post<V1VerifiedAccount>('/auth/account-verifications', { email: email.trim().toLowerCase(), code });
}

export function requestPasswordResetV1(email: string): Promise<{ state: 'accepted' }> {
  return v1Post('/auth/password-reset-requests', { email: email.trim().toLowerCase() });
}

export function verifyPasswordResetCodeV1(email: string, code: string): Promise<{ resetToken: string }> {
  return v1Post('/auth/password-reset-verifications', { email: email.trim().toLowerCase(), code });
}

export function resetPasswordV1(token: string, newPassword: string): Promise<{ state: 'completed' }> {
  return v1Post('/auth/password-resets', { token, newPassword });
}

// ── Persisted setup progress (the review screen is a completion stage, not a ninth category)

export type V1SetupProgress = {
  setupProgressId: string;
  userId: string;
  categories: Record<SetupCategory, SetupStatus>;
  completeCount: number;
  total: number;
  state: 'in-progress' | 'complete';
  version: number;
  completedAt: string | null;
};

export function getSetupProgressV1(): Promise<V1SetupProgress> {
  return v1Get<V1SetupProgress>('/setup-progress');
}

export function updateSetupProgressV1(category: SetupCategory, status: SetupStatus, version: number): Promise<V1SetupProgress> {
  return v1Patch<V1SetupProgress>('/setup-progress', { category, status }, {
    ifMatch: String(version),
    idempotencyKey: generateIdempotencyKey(),
  });
}

export type NewLovedOne = {
  displayName: string;
  preferredLanguage: string;
  timezone: string;
  profile?: Partial<V1PatientProfile>;
  relationshipType?: string;
  wakeTime?: string;
  topics?: string[];
  otherTopic?: string;
  speechOrHearingNotes?: string;
};

/**
 * Adding a loved one is three writes, and the third is the one that was never made.
 *
 * A daily check-in is refused with 403 CONSENT_REQUIRED unless a granted home_cognitive_monitoring consent
 * exists, and the monitoring pipeline drops any session without a consentRef. Adding a loved one creates the
 * patient and care plan only; the older adult must make the product-consent choice on the Mirror (or with the
 * care team). The caregiver can view the state and withdraw an existing grant, but cannot accept it for them.
 *
 * The care plan carries everything that changes how Aria talks, which is what the mirror reads from its
 * device configuration; those fields previously sat unread in the legacy patient document.
 */
export async function createLovedOneV1(input: NewLovedOne): Promise<V1PatientRecord> {
  const patient = await createPatientV1({
    displayName: input.displayName,
    preferredLanguage: input.preferredLanguage,
    timezone: input.timezone,
    profile: input.profile,
    relationshipType: input.relationshipType,
  });

  await putCarePlanV1(patient.patientId, 0, {
    dailyRoutine: { wakeTime: input.wakeTime || '' },
    communicationPreferences: {
      topics: input.topics || [],
      otherTopic: input.otherTopic || '',
      speechOrHearingNotes: input.speechOrHearingNotes || '',
      speechSpeed: input.profile?.speechSpeed || 'normal',
    },
  });

  return patient;
}

// ── Feedback (replaces the tokenless legacy POST /feedback, which took a nurseId in the body)

export function submitFeedbackV1(message: string, category?: string): Promise<{ feedbackId: string; createdAt: string }> {
  return v1Post('/feedback', category ? { message, category } : { message });
}

// ── The dashboard view, assembled once

export type CaregiverHomePatient = {
  patientId: string;
  displayName: string;
  preferredLanguage: string;
  timezone: string;
  version: number;
  profile: V1PatientProfile;
  mirrorName: string | null;
  deviceId: string | null;
  /** True when this loved one cannot have check-ins yet because consent is missing. */
  needsConsent: boolean;
};

export type CaregiverHome = { caregiver: V1CaregiverProfile; patients: CaregiverHomePatient[] };

/**
 * Everything the dashboard and settings screens used to get from one legacy document.
 *
 * Consent state is fetched per loved one because it decides whether check-ins can run at all, and a screen
 * that cannot see it can only show a vague "no data yet" for a loved one who is actually just waiting to be
 * consented for. It is requested in parallel and a failure degrades to `needsConsent: false` rather than
 * failing the whole screen — a consent prompt is worth less than the list itself.
 */
export async function loadCaregiverHome(): Promise<CaregiverHome> {
  const [caregiver, patients, assignments] = await Promise.all([
    getCaregiverProfileV1(),
    listPatientRecordsV1(),
    listDeviceAssignmentsV1().catch(() => [] as V1DeviceAssignment[]),
  ]);

  const assignmentByPatient = new Map(assignments.map((assignment) => [assignment.patientId, assignment]));
  const consentStates = await Promise.all(patients.map((patient) =>
    getConsentStateV1(patient.patientId).catch(() => null),
  ));

  return {
    caregiver,
    patients: patients.map((patient, index) => {
      const assignment = assignmentByPatient.get(patient.patientId);
      return {
        patientId: patient.patientId,
        displayName: patient.displayName,
        preferredLanguage: patient.preferredLanguage,
        timezone: patient.timezone,
        version: patient.version,
        profile: patient.profile,
        mirrorName: assignment?.mirrorName ?? null,
        deviceId: assignment?.deviceId ?? null,
        needsConsent: (consentStates[index]?.missingPurposes.length ?? 0) > 0,
      };
    }),
  };
}

export type { V1PatientStatus };

// ── Verbs v1Client did not expose yet

type WriteOptions = { ifMatch?: string; idempotencyKey?: string };

async function v1Patch<T>(path: string, body: unknown, options?: WriteOptions): Promise<T> {
  return v1Write<T>('PATCH', path, body, options);
}

async function v1Put<T>(path: string, body: unknown, options?: WriteOptions): Promise<T> {
  return v1Write<T>('PUT', path, body, options);
}

async function v1Delete<T>(path: string, options?: { idempotencyKey?: string }): Promise<T> {
  const headers: Record<string, string> = {};
  if (options?.idempotencyKey) headers['Idempotency-Key'] = options.idempotencyKey;
  const envelope = await v1FetchWithHeaders<T>(path, { method: 'DELETE', headers });
  return envelope.data;
}

/**
 * PATCH/PUT with the two headers v1 enforces on writes: If-Match on versioned resources, and an
 * Idempotency-Key where the route requires one. Kept here rather than in v1Client because it is these
 * caregiver resources that need them.
 */
async function v1Write<T>(method: 'PATCH' | 'PUT', path: string, body: unknown, options?: WriteOptions): Promise<T> {
  const headers: Record<string, string> = {};
  if (options?.ifMatch) headers['If-Match'] = options.ifMatch;
  if (options?.idempotencyKey) headers['Idempotency-Key'] = options.idempotencyKey;
  const envelope = await v1FetchWithHeaders<T>(path, { method, body, headers });
  return envelope.data;
}

function isNotFound(error: unknown): boolean {
  return typeof error === 'object' && error !== null && (error as { status?: number }).status === 404;
}

/** Re-exported so screens can build a URL for a resource this module has no verb for yet. */
export { getV1Url };
