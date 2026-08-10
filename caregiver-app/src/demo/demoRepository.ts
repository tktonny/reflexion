import type { SetupCategory, SetupStatus } from '../architecture/models';
import type {
  CaregiverHomePatient,
  V1CarePlan,
  V1CaregiverProfile,
  V1CareCircle,
  V1CareCircleMember,
  V1ConsentState,
  V1DeviceAssignment,
  V1FamilyMessage,
  V1NotificationPreferences,
  V1PatientProfile,
  V1PatientRecord,
  V1PrivacyState,
  V1ReminderOccurrence,
  V1Routine,
  V1SessionDetail,
  V1SessionLog,
  V1SessionProcessingStatus,
  V1SessionDay,
  V1SessionFeed,
  V1TrendDay,
} from '../lib/v1Caregiver';
import type { V1Notification, V1SupportThread } from '../lib/v1Client';
import type { V1PatientStatus } from '../lib/v1Status';

// Keep the fixture aligned with the public V1 contract without importing the production caregiver
// module at runtime (v1Caregiver itself imports v1Client, which delegates to this repository in demo mode).
const CHECKIN_CONSENT_PURPOSE = 'home_cognitive_monitoring';
const CHECKIN_CONSENT_DOCUMENT_VERSION = 'checkin-consent-2026-07';
const RESEARCH_CONSENT_PURPOSE = 'optional_research_participation';
const RESEARCH_CONSENT_DOCUMENT_VERSION = 'research-consent-2026-07';

export type DemoScenario = {
  lovedOne: 'margaret' | 'second';
  device: 'online' | 'offline';
  setup: 'complete' | 'incomplete';
  session: 'none' | 'completed' | 'partial' | 'processing';
  routineResponse: 'presented' | 'reported-complete' | 'deferred' | 'declined' | 'no-response';
  messageType: 'text' | 'photo' | 'voice';
  messageDelivery: 'queued' | 'delivered' | 'failed';
  messageInteraction: 'none' | 'viewed' | 'played' | 'replayed';
  consent: 'pending' | 'accepted' | 'declined' | 'withdrawn';
  control: 'active' | 'paused';
  away: 'on' | 'off';
  research: 'invited' | 'not-invited';
};

const SCENARIO_KEY = 'reflexion.demo.scenario';
const nowIso = () => new Date().toISOString();
const dateOnly = (date = new Date()) => date.toISOString().slice(0, 10);
const selectedPatientId = () => scenario.lovedOne === 'second' ? 'demo-james' : 'demo-margaret';

export const DEFAULT_DEMO_SCENARIO: DemoScenario = {
  lovedOne: 'margaret',
  device: 'online',
  setup: 'complete',
  session: 'completed',
  routineResponse: 'reported-complete',
  messageType: 'text',
  messageDelivery: 'delivered',
  messageInteraction: 'viewed',
  consent: 'accepted',
  control: 'active',
  away: 'off',
  research: 'invited',
};

function storage(): Storage | null {
  if (typeof globalThis === 'undefined') return null;
  return (globalThis as typeof globalThis & { localStorage?: Storage }).localStorage ?? null;
}

function readScenario(): DemoScenario {
  try {
    const raw = storage()?.getItem(SCENARIO_KEY);
    if (!raw) return { ...DEFAULT_DEMO_SCENARIO };
    return { ...DEFAULT_DEMO_SCENARIO, ...(JSON.parse(raw) as Partial<DemoScenario>) };
  } catch {
    return { ...DEFAULT_DEMO_SCENARIO };
  }
}

let scenario: DemoScenario = readScenario();
const listeners = new Set<() => void>();

function persistScenario() {
  try {
    storage()?.setItem(SCENARIO_KEY, JSON.stringify(scenario));
  } catch {
    // The demo remains in memory if browser storage is unavailable.
  }
}

function notify() {
  listeners.forEach((listener) => listener());
}

export function getDemoScenario(): DemoScenario {
  return { ...scenario };
}

export function setDemoScenario(patch: Partial<DemoScenario>): void {
  scenario = { ...scenario, ...patch };
  persistScenario();
  notify();
}

export function resetDemoData(): void {
  scenario = { ...DEFAULT_DEMO_SCENARIO };
  persistScenario();
  runtime = createRuntime();
  notify();
}

export function subscribeDemoRepository(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

const profileDefaults: V1NotificationPreferences = {
  pushNotificationsEnabled: true,
  alertSensitivity: 'notify_me_about_everything',
  preferredDailySummaryTime: '19:00',
  summaryFrequency: 'daily-summary',
  triggers: {
    'conversation-session-summary': true,
    'no-interaction-yet-today': true,
    'repeated-missed-interactions': true,
    'recent-interaction-shorter-than-usual': true,
    'device-may-be-offline': true,
    'reminder-not-completed-or-unclear': true,
    'weekly-summary': true,
  },
};

const patientProfile = (overrides: Partial<V1PatientProfile> = {}): V1PatientProfile => ({
  age: 78,
  gender: 'female',
  photoUrl: null,
  phoneCountryCode: '+65',
  phoneNumber: '+6591234567',
  relationship: 'Parent',
  emergencyContact: null,
  livingArrangement: 'Lives at home',
  speechSpeed: 'normal',
  ...overrides,
});

type DemoRuntime = {
  profile: V1CaregiverProfile;
  patients: V1PatientRecord[];
  plans: Record<string, V1CarePlan>;
  routines: Record<string, V1Routine[]>;
  messages: Record<string, V1FamilyMessage[]>;
  setupCategories: Record<SetupCategory, SetupStatus>;
  notifications: V1Notification[];
  supportThreads: V1SupportThread[];
  careCircle: Record<string, V1CareCircle>;
  deletionRequests: Record<string, V1PrivacyState['deletionRequests'][number][]>;
};

function createPatients(): V1PatientRecord[] {
  return [
    {
      patientId: 'demo-margaret',
      displayName: 'Margaret',
      preferredLanguage: 'English',
      timezone: 'Asia/Singapore',
      ageBand: '75-84',
      profile: patientProfile(),
      status: 'active',
      version: 1,
    },
    {
      patientId: 'demo-james',
      displayName: 'James',
      preferredLanguage: 'English',
      timezone: 'Asia/Singapore',
      ageBand: '65-74',
      profile: patientProfile({ age: 70, gender: 'male', phoneNumber: '+6598765432', relationship: 'Parent' }),
      status: 'active',
      version: 1,
    },
  ];
}

function createPlan(patientId: string): V1CarePlan {
  return {
    patientId,
    version: 1,
    dailyRoutine: { wakeTime: '08:00' },
    communicationPreferences: {
      topics: ['family', 'music', 'local news'],
      otherTopic: '',
      speechSpeed: 'normal',
      speechOrHearingNotes: '',
      productControl: 'active',
      appLanguage: 'en',
      captions: true,
      volume: 'comfortable',
    },
    safetyNotes: null,
  };
}

function createRoutine(patientId: string, index = 0): V1Routine {
  return {
    routineId: `demo-routine-${patientId}-${index + 1}`,
    patientId,
    name: index === 0 ? 'Morning medication' : 'Afternoon tea',
    category: index === 0 ? 'medication' : 'meals',
    schedule: {
      timezone: 'Asia/Singapore',
      times: index === 0 ? ['09:00', '21:00'] : ['15:00'],
      recurrence: 'weekly',
      daysOfWeek: index === 0 ? [1, 2, 3, 4, 5] : [1, 3, 5],
      startsOn: dateOnly(),
      endsOn: null,
    },
    spokenReminder: index === 0 ? 'It is time for your morning medicine.' : 'Would you like a cup of tea?',
    notificationPolicy: index === 0 ? 'after-one-missed-or-unclear-response' : 'daily-summary',
    notificationPolicies: ['after-one-missed-or-unclear-response', 'daily-summary'],
    notes: 'A familiar, gentle reminder.',
    status: 'active',
    version: 1,
  };
}

function createMessage(patientId: string, index: number, overrides: Partial<V1FamilyMessage> = {}): V1FamilyMessage {
  const createdAt = new Date(Date.now() - index * 86_400_000).toISOString();
  return {
    messageId: `demo-message-${patientId}-${index + 1}`,
    patientId,
    body: index === 0 ? 'Thinking of you — have a lovely day.' : 'I will call after lunch.',
    type: 'text',
    state: index === 0 ? 'opened' : 'delivered',
    scheduledFor: createdAt,
    createdAt,
    deliveredAt: createdAt,
    openedAt: index === 0 ? createdAt : null,
    ...overrides,
  };
}

function createRuntime(): DemoRuntime {
  const patients = createPatients();
  const setupCategories: Record<SetupCategory, SetupStatus> = {
    household: scenario.setup === 'complete' ? 'complete' : 'complete',
    'pair-device': scenario.setup === 'complete' ? 'complete' : 'in-progress',
    'language-accessibility': scenario.setup === 'complete' ? 'complete' : 'in-progress',
    routines: scenario.setup === 'complete' ? 'complete' : 'not-started',
    notifications: scenario.setup === 'complete' ? 'complete' : 'not-started',
    'consent-control': scenario.setup === 'complete' ? 'complete' : 'in-progress',
    'research-participation': scenario.setup === 'complete' ? 'not-applicable' : 'not-started',
  };
  return {
    profile: {
      userId: 'demo-caregiver',
      tenantId: 'demo-tenant',
      name: 'Chloe',
      email: 'demo@reflexion.local',
      roles: ['caregiver'],
      phoneNumber: '+6590000000',
      relationshipToElderly: 'Daughter',
      appLanguage: 'en',
      notificationPreferences: { ...profileDefaults, triggers: { ...profileDefaults.triggers } },
      storeSessionSummaries: true,
    },
    patients,
    plans: Object.fromEntries(patients.map((patient) => [patient.patientId, createPlan(patient.patientId)])),
    routines: Object.fromEntries(patients.map((patient) => [patient.patientId, [createRoutine(patient.patientId), createRoutine(patient.patientId, 1)]])),
    messages: Object.fromEntries(patients.map((patient) => [patient.patientId, [createMessage(patient.patientId, 0), createMessage(patient.patientId, 1)]])),
    setupCategories,
    notifications: [
      {
        notificationId: 'demo-notification-session', patientId: 'demo-margaret', type: 'completion', state: 'unread',
        title: 'Session summary is ready', body: 'A completed conversation session is ready to review.', source: { type: 'session', id: 'demo-session-complete' }, localDate: dateOnly(), createdAt: nowIso(), readAt: null,
      },
      {
        notificationId: 'demo-notification-routine', patientId: 'demo-margaret', type: 'late_completion', state: 'read',
        title: 'Routine response recorded', body: 'Morning medication was reported complete.', source: { type: 'routine', id: 'demo-occurrence-1' }, localDate: dateOnly(), createdAt: new Date(Date.now() - 3_600_000).toISOString(), readAt: nowIso(),
      },
      {
        notificationId: 'demo-notification-device', patientId: 'demo-margaret', type: 'technical_issue', state: 'unread',
        title: 'Mirror connection update', body: 'The Mirror may be offline right now.', source: { type: 'device', id: 'demo-device-margaret' }, localDate: dateOnly(), createdAt: new Date(Date.now() - 7_200_000).toISOString(), readAt: null,
      },
    ],
    supportThreads: [{ threadId: 'demo-support-1', subject: 'Demo support thread', status: 'open', lastMessageAt: nowIso(), lastMessagePreview: 'This is local demo data.', caregiverUnread: false, createdAt: nowIso() }],
    careCircle: Object.fromEntries(patients.map((patient) => [patient.patientId, {
      patientId: patient.patientId,
      members: [{ memberId: `demo-member-${patient.patientId}`, kind: 'member', userId: 'demo-caregiver', name: 'Chloe', email: 'demo@reflexion.local', phoneNumber: null, role: 'full-access', permissions: ['view-loved-ones', 'receive-notifications', 'manage-routines', 'manage-devices'], state: 'active', version: 1, createdAt: nowIso() }],
      invitations: [],
    }])),
    deletionRequests: Object.fromEntries(patients.map((patient) => [patient.patientId, []])),
  };
}

let runtime: DemoRuntime = createRuntime();

function bodyOf(init: RequestInit): Record<string, any> {
  if (!init.body) return {};
  try {
    return typeof init.body === 'string' ? JSON.parse(init.body) as Record<string, any> : {};
  } catch {
    return {};
  }
}

function envelope(data: unknown, status = 200): Response {
  return new Response(JSON.stringify({ data, meta: { requestId: `demo-${Date.now()}` } }), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

function notFound(path: string): Response {
  return envelope({ error: { code: 'NOT_FOUND', message: `Demo fixture has no route for ${path}.` } }, 404);
}

function patient(patientId: string): V1PatientRecord | undefined {
  return runtime.patients.find((item) => item.patientId === patientId);
}

function carePlanFor(patientId: string): V1CarePlan | null {
  const current = runtime.plans[patientId] || null;
  if (!current) return null;
  if (patientId !== selectedPatientId()) return current;
  return {
    ...current,
    communicationPreferences: {
      ...current.communicationPreferences,
      productControl: scenario.control,
    },
  };
}

function assignmentFor(patientId: string): V1DeviceAssignment {
  const linked = scenario.setup === 'complete' && patientId === selectedPatientId();
  const offline = linked && scenario.device === 'offline';
  const person = patient(patientId);
  return {
    patientId,
    patientName: person?.displayName || 'Loved one',
    timezone: person?.timezone || 'Asia/Singapore',
    assignmentId: linked ? `demo-assignment-${patientId}` : null,
    deviceId: linked ? `demo-device-${patientId}` : null,
    mirrorName: linked ? 'Reflexion Mirror' : null,
    assignedAt: linked ? new Date(Date.now() - 14 * 86_400_000).toISOString() : null,
    device: linked ? {
      serial: `DEMO-${patientId.slice(-4).toUpperCase()}`,
      softwareVersion: 'demo.2026.08',
      status: offline ? 'offline' : 'online',
      technicalState: offline ? 'possible_issue' : 'ok',
      lastHeartbeatAt: offline ? new Date(Date.now() - 4 * 3_600_000).toISOString() : nowIso(),
    } : null,
  };
}

function consentStatusFor(patientId: string): DemoScenario['consent'] {
  return patientId === selectedPatientId() ? scenario.consent : 'accepted';
}

function consentStateFor(patientId: string): V1ConsentState {
  const status = consentStatusFor(patientId);
  const consents: V1ConsentState['consents'] = [];
  if (status !== 'pending') {
    const signedAt = new Date(Date.now() - 2 * 86_400_000).toISOString();
    consents.push({
      consentId: `demo-consent-${patientId}`,
      purpose: CHECKIN_CONSENT_PURPOSE,
      documentVersion: CHECKIN_CONSENT_DOCUMENT_VERSION,
      status: status === 'accepted' ? 'granted' : status,
      signedAt,
      withdrawnAt: status === 'withdrawn' ? nowIso() : null,
    });
  }
  if (scenario.research === 'invited') {
    consents.push({
      consentId: `demo-research-${patientId}`,
      purpose: RESEARCH_CONSENT_PURPOSE,
      documentVersion: RESEARCH_CONSENT_DOCUMENT_VERSION,
      status: 'invitation_pending',
      signedAt: null,
      withdrawnAt: null,
    });
  }
  return {
    patientId,
    consents,
    requiredPurposes: [CHECKIN_CONSENT_PURPOSE],
    missingPurposes: status === 'accepted' ? [] : [CHECKIN_CONSENT_PURPOSE],
  };
}

function statusFor(patientId: string): V1PatientStatus {
  const assignment = assignmentFor(patientId);
  const selected = patientId === selectedPatientId();
  const session = selected ? scenario.session : 'none';
  const completedToday = session === 'completed' || session === 'partial';
  const technicalState: V1PatientStatus['technicalState'] = assignment.device?.status === 'offline'
    ? 'unreachable'
    : assignment.device?.technicalState || 'unknown';
  return {
    patientId,
    baselineState: session === 'none' ? 'establishing' : 'complete',
    baselineProgress: { completedSessions: session === 'none' ? 0 : 5, requiredSessions: 7, windowDays: 14 },
    status: session === 'processing' ? 'establishing' : completedToday ? 'doing_well' : 'establishing',
    primaryReason: completedToday ? 'CHECKIN_COMPLETED_TODAY' : 'CHECKIN_MISSED_TODAY',
    secondaryReasons: technicalState === 'unreachable' ? ['DEVICE_UNREACHABLE'] : [],
    completedToday,
    technicalState,
    lastInteractionAt: completedToday ? new Date(Date.now() - 90 * 60_000).toISOString() : null,
    conversationsToday: completedToday ? 2 : 0,
    checkinsToday: completedToday ? 1 : 0,
    chatsToday: completedToday ? 1 : 0,
    updatedAt: nowIso(),
  };
}

function occurrencesFor(patientId: string): V1ReminderOccurrence[] {
  if (scenario.setup === 'incomplete') return [];
  const routine = runtime.routines[patientId]?.[0];
  if (!routine) return [];
  const responseStatus = scenario.routineResponse;
  return [
    {
      occurrenceId: 'demo-occurrence-1', patientId, scheduledAt: new Date(Date.now() - 2 * 3_600_000).toISOString(), type: 'routine', category: routine.category,
      displayText: routine.name, status: responseStatus, respondedAt: responseStatus === 'presented' || responseStatus === 'no-response' ? null : new Date(Date.now() - 90 * 60_000).toISOString(),
    },
    {
      occurrenceId: 'demo-occurrence-2', patientId, scheduledAt: new Date(Date.now() + 5 * 3_600_000).toISOString(), type: 'routine', category: 'meals',
      displayText: 'Afternoon tea', status: 'scheduled', respondedAt: null,
    },
  ];
}

function messageFixtures(patientId: string): V1FamilyMessage[] {
  const delivery = scenario.messageDelivery === 'delivered' ? 'delivered' : scenario.messageDelivery;
  const interaction = scenario.messageInteraction;
  const type = scenario.messageType;
  const base = runtime.messages[patientId]?.[0] || createMessage(patientId, 0);
  const first = {
    ...base,
    type: type as 'text',
    state: delivery === 'failed' ? 'scheduled' : delivery,
    deliveredAt: delivery === 'delivered' ? nowIso() : null,
    openedAt: interaction === 'none' ? null : nowIso(),
  } as V1FamilyMessage;
  return [first, ...runtime.messages[patientId].slice(1)];
}

function sessionsFor(patientId: string): V1SessionDetail[] {
  if (patientId !== selectedPatientId() || scenario.session === 'none') return [];
  const name = patient(patientId)?.displayName || 'Loved one';
  const createdAt = new Date(Date.now() - 90 * 60_000).toISOString();
  const logs: V1SessionLog[] = [
    { role: 'mirror', sentence: `Good morning, ${name}. How are you feeling about the day ahead?`, words: 10, duration: 8, wordsPerSecond: 1.25 },
    { role: 'loved-one', sentence: 'I am looking forward to seeing the family later.', words: 10, duration: 7, wordsPerSecond: 1.42 },
  ];
  return [{
    id: 'demo-session-complete', patientId, patientName: name, type: 'daily_checkin', state: scenario.session,
    duration: scenario.session === 'partial' ? 48 : 132, words: scenario.session === 'partial' ? 24 : 82, exchanges: scenario.session === 'partial' ? 1 : 4,
    avgLatency: 1.2, createdAt, updatedAt: nowIso(), logs,
  }];
}

function trendFor(patientId: string, days: number): V1TrendDay[] {
  const hasData = patientId === selectedPatientId() && scenario.session !== 'none';
  return Array.from({ length: days }, (_, index) => {
    const date = new Date();
    date.setDate(date.getDate() - (days - index - 1));
    const active = hasData && (index === days - 1 || index % 4 === 1);
    return { date: dateOnly(date), duration: active ? (index % 3 === 0 ? 90 : 132) : 0, sessionCount: active ? 1 : 0, status: active ? 'green' : null, missed: !active };
  });
}

function setupProgress(): { setupProgressId: string; userId: string; categories: Record<SetupCategory, SetupStatus>; completeCount: number; total: number; state: 'in-progress' | 'complete'; version: number; completedAt: string | null } {
  const categories = { ...runtime.setupCategories };
  const completeCount = Object.values(categories).filter((value) => value === 'complete' || value === 'not-applicable').length;
  return { setupProgressId: 'demo-setup', userId: 'demo-caregiver', categories, completeCount, total: 7, state: completeCount === 7 ? 'complete' : 'in-progress', version: 1, completedAt: completeCount === 7 ? nowIso() : null };
}

function privacyFor(patientId: string): V1PrivacyState {
  const consent = consentStateFor(patientId);
  const record = consent.consents.find((item) => item.purpose === CHECKIN_CONSENT_PURPOSE);
  const status = !record ? 'pending' : record.status === 'granted' ? 'accepted' : record.status as V1PrivacyState['consent']['status'];
  return {
    patientId,
    consent: { status, requiredPurpose: CHECKIN_CONSENT_PURPOSE, history: consent.consents.filter((item) => item.purpose === CHECKIN_CONSENT_PURPOSE) },
    research: { status: 'separate', message: 'Optional research is separate from ordinary Reflexion use.' },
    retention: { structuredData: 'Kept while your account is active', sessionMedia: 'Limited retention configured by Reflexion', operationalLogs: 'Used for security and reliability', configuredByServer: true },
    deletionCategories: [
      { category: 'sessions', label: 'Conversation sessions' },
      { category: 'messages', label: 'Family messages' },
      { category: 'routine-responses', label: 'Routine responses' },
      { category: 'device-events', label: 'Device events' },
    ],
    deletionRequests: runtime.deletionRequests[patientId] || [],
  };
}

function requestPath(path: string): string[] {
  return path.split('?')[0].replace(/^\/+/, '').split('/').filter(Boolean).map((item) => decodeURIComponent(item));
}

function query(path: string): URLSearchParams {
  return new URLSearchParams(path.includes('?') ? path.slice(path.indexOf('?') + 1) : '');
}

function methodOf(init: RequestInit): string {
  return String(init.method || 'GET').toUpperCase();
}

function updateScenarioFromConsent(status: string) {
  if (status === 'granted') setDemoScenario({ consent: 'accepted' });
  else if (status === 'declined') setDemoScenario({ consent: 'declined' });
  else if (status === 'withdrawn') setDemoScenario({ consent: 'withdrawn' });
}

/**
 * Central local repository for the development-only fixture mode. It returns the same envelopes as the
 * production V1 API, so screens and their navigation do not know which repository supplied the data.
 */
export async function requestDemo(path: string, init: RequestInit = {}): Promise<Response> {
  const method = methodOf(init);
  const parts = requestPath(path);
  const payload = bodyOf(init);
  const patientId = parts[0] === 'patients' ? parts[1] : undefined;

  if (parts[0] === 'me') {
    if (method === 'GET') return envelope(runtime.profile);
    if (method === 'PATCH') {
      runtime.profile = {
        ...runtime.profile,
        ...payload,
        notificationPreferences: { ...runtime.profile.notificationPreferences, ...(payload.notificationPreferences || {}), triggers: { ...runtime.profile.notificationPreferences.triggers, ...(payload.notificationPreferences?.triggers || {}) } },
      };
      return envelope(runtime.profile);
    }
    if (method === 'POST' && parts[1] === 'email-change-requests') return envelope({ state: 'accepted' });
    if (method === 'POST' && parts[1] === 'phone-change-requests') return envelope({ state: 'accepted', phoneNumber: payload.phoneNumber || runtime.profile.phoneNumber });
    if (method === 'POST' && (parts[1] === 'email-changes' || parts[1] === 'phone-changes' || parts[1] === 'password-changes')) return envelope({ ...runtime.profile, state: 'completed' });
  }

  if (parts[0] === 'patients' && parts.length === 1 && method === 'GET') return envelope(runtime.patients);
  if (parts[0] === 'patients' && parts.length === 1 && method === 'POST') {
    const created: V1PatientRecord = {
      patientId: `demo-patient-${Date.now()}`,
      displayName: String(payload.displayName || 'New loved one'),
      preferredLanguage: String(payload.preferredLanguage || 'English'),
      timezone: String(payload.timezone || 'Asia/Singapore'),
      ageBand: null,
      profile: patientProfile(payload.profile || {}),
      status: 'active',
      version: 1,
    };
    runtime.patients.push(created);
    runtime.plans[created.patientId] = createPlan(created.patientId);
    runtime.routines[created.patientId] = [];
    runtime.messages[created.patientId] = [];
    runtime.deletionRequests[created.patientId] = [];
    runtime.careCircle[created.patientId] = { patientId: created.patientId, members: [], invitations: [] };
    return envelope(created);
  }

  if (patientId && parts[2] === 'status' && method === 'GET') return envelope(statusFor(patientId));
  if (parts[0] === 'patient-statuses' && method === 'GET') {
    const ids = query(path).get('ids')?.split(',').filter(Boolean) || [];
    return envelope(ids.map((id) => ({ patientId: id, outcome: 'ok', status: statusFor(id) })));
  }

  if (parts[0] === 'patients' && patientId && parts.length === 2 && method === 'PATCH') {
    const existing = patient(patientId);
    if (!existing) return notFound(path);
    Object.assign(existing, payload, { profile: { ...existing.profile, ...(payload.profile || {}) }, version: existing.version + 1 });
    return envelope(existing);
  }

  if (patientId && parts[2] === 'care-plan') {
    if (method === 'GET') return envelope(carePlanFor(patientId));
    if (method === 'PUT') {
      const current = runtime.plans[patientId] || createPlan(patientId);
      runtime.plans[patientId] = { ...current, ...payload, patientId, version: current.version + 1, communicationPreferences: { ...current.communicationPreferences, ...(payload.communicationPreferences || {}) }, dailyRoutine: { ...current.dailyRoutine, ...(payload.dailyRoutine || {}) } };
      if (payload.communicationPreferences?.productControl === 'paused' || payload.communicationPreferences?.productControl === 'active') setDemoScenario({ control: payload.communicationPreferences.productControl });
      return envelope(carePlanFor(patientId));
    }
  }

  if (parts[0] === 'patients' && patientId && parts[2] === 'routines') {
    if (method === 'GET') return envelope(runtime.routines[patientId] || []);
    if (method === 'POST') {
      const created: V1Routine = { ...createRoutine(patientId, runtime.routines[patientId]?.length || 0), ...payload, routineId: `demo-routine-${Date.now()}`, patientId, version: 1 };
      runtime.routines[patientId] = [...(runtime.routines[patientId] || []), created];
      return envelope(created);
    }
  }
  if (parts[0] === 'routines' && parts[1] && (method === 'PATCH' || method === 'DELETE')) {
    for (const [owner, routines] of Object.entries(runtime.routines)) {
      const index = routines.findIndex((routine) => routine.routineId === parts[1]);
      if (index >= 0) {
        if (method === 'DELETE') {
          routines[index] = { ...routines[index], status: 'ended', version: routines[index].version + 1 };
          return envelope({ routineId: parts[1], state: 'ended' });
        }
        routines[index] = { ...routines[index], ...payload, version: routines[index].version + 1 };
        return envelope(routines[index]);
      }
      void owner;
    }
  }

  if (patientId && parts[2] === 'reminder-occurrences' && method === 'GET') return envelope(occurrencesFor(patientId));
  if (parts[0] === 'reminder-occurrences' && parts[1] && parts[2] === 'responses' && method === 'POST') {
    const current = occurrencesFor(selectedPatientId()).find((occurrence) => occurrence.occurrenceId === parts[1]);
    return envelope({ ...(current || occurrencesFor(selectedPatientId())[0]), ...payload });
  }

  if (patientId && parts[2] === 'consents') {
    if (method === 'GET') return envelope(consentStateFor(patientId));
    if (method === 'POST') {
      if (payload.purpose === CHECKIN_CONSENT_PURPOSE) updateScenarioFromConsent(String(payload.status));
      if (payload.purpose === RESEARCH_CONSENT_PURPOSE) setDemoScenario({ research: 'invited' });
      return envelope(consentStateFor(patientId));
    }
  }

  if (patientId && parts[2] === 'privacy' && method === 'GET') return envelope(privacyFor(patientId));
  if (patientId && parts[2] === 'data-deletion-requests' && method === 'POST') {
    const request = { requestId: `demo-deletion-${Date.now()}`, categories: payload.categories || [], state: 'accepted', createdAt: nowIso(), updatedAt: nowIso(), remainingObjectKeys: [], error: null };
    runtime.deletionRequests[patientId] = [...(runtime.deletionRequests[patientId] || []), request];
    return envelope(request);
  }

  if (patientId && parts[2] === 'care-circle') {
    const current = runtime.careCircle[patientId] || { patientId, members: [], invitations: [] };
    if (method === 'GET' && parts.length === 3) return envelope(current);
    if (method === 'POST' && parts[3] === 'invitations') {
      const invitation: V1CareCircleMember = { memberId: `demo-invitation-${Date.now()}`, kind: 'invitation', invitee: payload.emailOrPhone, email: payload.emailOrPhone, role: payload.role || 'standard-access', permissions: payload.permissions || [], state: 'pending', version: 1, createdAt: nowIso() };
      current.invitations.push(invitation);
      return envelope(invitation);
    }
    if (method === 'PATCH' && parts[3]) {
      const member = [...current.members, ...current.invitations].find((item) => item.memberId === parts[3]);
      if (member) Object.assign(member, payload, { version: member.version + 1 });
      return envelope(member || { memberId: parts[3], ...payload });
    }
    if (method === 'DELETE' && parts[3]) {
      current.members = current.members.filter((member) => member.memberId !== parts[3]);
      current.invitations = current.invitations.filter((member) => member.memberId !== parts[3]);
      return envelope({ memberId: parts[3], state: 'revoked' });
    }
  }

  if (parts[0] === 'device-assignments' && method === 'GET') return envelope({ assignments: runtime.patients.map((item) => assignmentFor(item.patientId)) });
  if (parts[0] === 'device-pairing-claims' && method === 'POST') {
    setDemoScenario({ setup: 'complete', device: 'online' });
    return envelope(assignmentFor(String(payload.patientId || selectedPatientId())));
  }
  if (parts[0] === 'devices' && parts[2] === 'revocations' && method === 'POST') {
    if (parts[1] === `demo-device-${selectedPatientId()}`) setDemoScenario({ setup: 'incomplete' });
    return envelope({ deviceId: parts[1], state: 'revoked' });
  }

  if (patientId && parts[2] === 'family-messages') {
    if (method === 'GET') return envelope({ messages: messageFixtures(patientId) });
    if (method === 'POST') {
      const sent = createMessage(patientId, 0, { messageId: `demo-message-${Date.now()}`, body: String(payload.body || 'A demo message'), state: scenario.messageDelivery === 'failed' ? 'scheduled' : scenario.messageDelivery, type: scenario.messageType as 'text', deliveredAt: scenario.messageDelivery === 'delivered' ? nowIso() : null, openedAt: scenario.messageInteraction === 'none' ? null : nowIso() });
      runtime.messages[patientId] = [sent, ...(runtime.messages[patientId] || [])];
      return envelope(sent);
    }
  }

  if (patientId && parts[2] === 'session-days' && parts.length === 3 && method === 'GET') {
    const month = query(path).get('month') || dateOnly().slice(0, 7);
    const days = trendFor(patientId, 30).filter((item) => item.date.startsWith(month)).map((item) => ({ date: item.date, day: Number(item.date.slice(-2)), count: item.sessionCount, completedCount: item.sessionCount, hasCompletedSession: !item.missed }));
    return envelope({ days });
  }
  if (patientId && parts[2] === 'session-days' && parts[3] && method === 'GET') {
    const sessions = sessionsFor(patientId);
    return envelope({ patientId, date: parts[3], patientName: patient(patientId)?.displayName || 'Loved one', sessions: sessions.filter((session) => session.createdAt?.slice(0, 10) === parts[3] || parts[3] === dateOnly()).length ? sessions.filter((session) => session.createdAt?.slice(0, 10) === parts[3] || parts[3] === dateOnly()) : (scenario.session === 'none' ? [] : sessions) });
  }
  if (patientId && parts[2] === 'sessions' && parts.length === 3 && method === 'GET') {
    const sessions = sessionsFor(patientId);
    const feed: V1SessionFeed = { patientId, patientName: patient(patientId)?.displayName || 'Loved one', sessions, nextBefore: null };
    return envelope(feed);
  }
  if (patientId && parts[2] === 'sessions' && parts[3] && method === 'GET') return envelope(sessionsFor(patientId).find((session) => session.id === parts[3]) || sessionsFor(patientId)[0] || { id: parts[3], patientId, patientName: patient(patientId)?.displayName || 'Loved one', type: 'daily_checkin', state: scenario.session, duration: 0, words: 0, exchanges: 0, avgLatency: 0, createdAt: null, updatedAt: null, logs: [] });
  if (parts[0] === 'sessions' && parts[1] && parts[2] === 'processing-status' && method === 'GET') {
    const state: V1SessionProcessingStatus['state'] = scenario.session === 'processing' ? 'processing' : scenario.session === 'none' ? 'accepted' : scenario.session === 'partial' ? 'failed' : 'completed';
    return envelope({ sessionId: parts[1], operationId: `demo-operation-${parts[1]}`, state, stage: state === 'completed' ? 'ready' : 'transcript', retryable: state === 'failed', result: state === 'completed' ? { summary: 'A completed demo session is ready to review.' } : null, updatedAt: nowIso() });
  }
  if (patientId && parts[2] === 'away-periods' && method === 'POST') {
    setDemoScenario({ away: 'on' });
    return envelope({ awayPeriodId: `demo-away-${Date.now()}`, patientId, startsOn: payload.startsOn || dateOnly(), endsOn: payload.endsOn || dateOnly(), timezone: payload.timezone || 'Asia/Singapore', state: 'active' });
  }
  if (patientId && parts[2] === 'session-trend' && method === 'GET') return envelope({ trend: trendFor(patientId, Number(query(path).get('days') || 7)) });
  if (patientId && parts[2] === 'session-summaries' && method === 'POST') return envelope({ patientId, date: payload.date || dateOnly(), summary: scenario.session === 'none' ? null : 'The Mirror recorded a conversation session and the loved one shared a few updates.', reason: scenario.session === 'none' ? 'no_transcript' : null });

  if (parts[0] === 'notifications' && parts.length === 1 && method === 'GET') return envelope(runtime.notifications, 200);
  if (parts[0] === 'notifications' && parts[1] && parts[2] === 'read' && method === 'POST') {
    const notification = runtime.notifications.find((item) => item.notificationId === parts[1]);
    if (notification) Object.assign(notification, { state: 'read', readAt: nowIso() });
    return envelope(notification || runtime.notifications[0]);
  }
  if (parts[0] === 'notification-devices' && parts[1] === 'test' && method === 'POST') return envelope({ outcome: 'accepted', devices: 1, delivered: 1, detail: null });
  if (parts[0] === 'notification-devices' && method === 'POST') return envelope({ deviceId: 'demo-notification-device', state: 'registered' });

  if (parts[0] === 'support' && parts[1] === 'threads') {
    if (method === 'GET' && parts.length === 2) return envelope(runtime.supportThreads);
    if (method === 'POST' && parts.length === 2) {
      const thread: V1SupportThread = { threadId: `demo-support-${Date.now()}`, subject: payload.subject || 'Demo support request', status: 'open', lastMessageAt: nowIso(), lastMessagePreview: payload.body || '', caregiverUnread: false, createdAt: nowIso() };
      runtime.supportThreads.unshift(thread);
      return envelope(thread);
    }
    if (method === 'POST' && parts[3] === 'messages') return envelope({ messageId: `demo-support-message-${Date.now()}` });
  }

  if (parts[0] === 'setup-progress') {
    if (method === 'GET') return envelope(setupProgress());
    if (method === 'PATCH') {
      const category = payload.category as SetupCategory;
      if (category && category in runtime.setupCategories) runtime.setupCategories[category] = payload.status as SetupStatus;
      return envelope(setupProgress());
    }
  }

  if (parts[0] === 'auth') {
    if (method === 'POST' && parts[1] === 'registrations') return envelope({ state: 'authenticated', email: payload.email || 'demo@reflexion.local', emailVerified: true, accessToken: 'demo-access-token', refreshToken: 'demo-refresh-token', actor: { userId: 'demo-caregiver', tenantId: 'demo-tenant', name: payload.name || 'Chloe', email: payload.email || 'demo@reflexion.local', roles: ['caregiver'] } });
    if (method === 'POST' && (parts[1] === 'account-verification-requests' || parts[1] === 'account-verifications' || parts[1] === 'password-reset-requests' || parts[1] === 'password-reset-verifications' || parts[1] === 'password-resets')) return envelope({ state: parts[1] === 'password-reset-verifications' ? undefined : 'accepted', resetToken: parts[1] === 'password-reset-verifications' ? 'demo-reset-token' : undefined });
    if (method === 'POST' && parts[1] === 'session-refreshes') return envelope({ accessToken: 'demo-access-token', refreshToken: 'demo-refresh-token', accessTokenExpiresAt: '2099-01-01T00:00:00.000Z', refreshTokenExpiresAt: '2099-01-01T00:00:00.000Z' });
    if (method === 'DELETE' && parts[1] === 'sessions') return envelope({ state: 'revoked' });
  }
  if (parts[0] === 'feedback' && method === 'POST') return envelope({ feedbackId: `demo-feedback-${Date.now()}`, createdAt: nowIso() });
  if (parts[0] === 'patient-statuses') return envelope([]);

  return notFound(path);
}

export type DemoHomePatient = CaregiverHomePatient;
