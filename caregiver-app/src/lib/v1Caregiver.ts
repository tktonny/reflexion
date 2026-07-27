import { getV1Url } from './apiUrl';
import { generateIdempotencyKey, v1FetchWithHeaders, v1Get, v1Post } from './v1Client';
import type { V1PatientStatus } from './v1Status';

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

export type V1NotificationPreferences = {
  pushNotificationsEnabled: boolean;
  alertSensitivity: V1AlertSensitivity;
  preferredDailySummaryTime: V1SummaryTime;
};

export type V1CaregiverProfile = {
  userId: string;
  tenantId: string;
  name: string;
  email: string;
  roles: string[];
  phoneNumber: string;
  relationshipToElderly: string | null;
  notificationPreferences: V1NotificationPreferences;
};

export function getCaregiverProfileV1(): Promise<V1CaregiverProfile> {
  return v1Get<V1CaregiverProfile>('/me');
}

/** Partial by design: omitted keys are left alone, so a screen cannot blank what it did not load. */
export async function updateCaregiverProfileV1(input: {
  name?: string;
  phoneNumber?: string;
  relationshipToElderly?: string | null;
  notificationPreferences?: Partial<V1NotificationPreferences>;
}): Promise<V1CaregiverProfile> {
  return v1Patch<V1CaregiverProfile>('/me', input);
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

/** The document version the onboarding consent screen presents. Bump when that wording changes. */
export const CHECKIN_CONSENT_DOCUMENT_VERSION = 'checkin-consent-2026-07';

export function grantCheckInConsentV1(patientId: string, purpose: string): Promise<unknown> {
  return v1Post(`/patients/${encodeURIComponent(patientId)}/consents`, {
    purpose,
    documentVersion: CHECKIN_CONSENT_DOCUMENT_VERSION,
    status: 'granted',
  }, { idempotencyKey: generateIdempotencyKey() });
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
  device: { serial: string | null; softwareVersion: string | null; status: string | null; lastHeartbeatAt: string | null } | null;
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
