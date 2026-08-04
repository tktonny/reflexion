/** Product vocabulary from Final Reflexion Caregiver-App Architecture. */

export type InteractionState =
  | 'interaction-recorded-today'
  | 'no-interaction-yet-today'
  | 'recent-interaction-shorter-than-usual'
  | 'device-may-be-offline'
  | 'needs-your-attention';

export type DeviceType = 'mirror' | 'bear' | 'app' | 'other-supported-device';
export type DeviceState = 'unpaired' | 'pairing' | 'connected' | 'needs-setup' | 'may-be-offline';

export type SetupCategory =
  | 'household'
  | 'pair-device'
  | 'language-accessibility'
  | 'routines'
  | 'notifications'
  | 'consent-control'
  | 'care-circle'
  | 'research-participation';
export type SetupStatus = 'not-started' | 'in-progress' | 'complete' | 'skipped';

export type RoutineCategory =
  | 'medication'
  | 'meals'
  | 'hydration'
  | 'medical-appointments'
  | 'exercise'
  | 'family-events'
  | 'custom-other';
export type RoutineResponseState =
  | 'scheduled'
  | 'presented'
  | 'reported-complete'
  | 'deferred'
  | 'declined'
  | 'no-response'
  | 'device-unavailable';
export type RoutineNotification = 'do-not-notify' | 'after-one-missed-or-unclear-response' | 'daily-summary';

export type NotificationTrigger =
  | 'conversation-session-summary'
  | 'no-interaction-yet-today'
  | 'repeated-missed-interactions'
  | 'recent-interaction-shorter-than-usual'
  | 'device-may-be-offline'
  | 'reminder-not-completed-or-unclear'
  | 'new-chat-reply'
  | 'weekly-summary';
export type SessionSummaryFrequency = 'immediately-after-each-session' | 'daily-summary' | 'weekly-summary' | 'off';

export type ConsentStatus = 'pending' | 'accepted' | 'declined' | 'withdrawn';
export type ConsentControl = 'conversations-active' | 'conversations-paused' | 'conversation-stopped' | 'sharing-paused' | 'participation-withdrawn';
export type CareCircleRole = 'full-access' | 'standard-access' | 'view-only' | 'custom-access';
export type CareCirclePermission = 'view-loved-ones' | 'receive-notifications' | 'manage-routines' | 'manage-devices' | 'invite-or-remove-caregivers';

export type MessageType = 'text' | 'photo' | 'voice';
export type MessageStatus = 'draft' | 'scheduled' | 'queued' | 'delivered-to-device' | 'opened-or-played' | 'dismissed' | 'expired' | 'failed';
export type MessageSchedule = 'now' | 'specific-date-and-time';

export type ScreenState =
  | 'loading'
  | 'empty'
  | 'error'
  | 'caregiver-phone-offline'
  | 'partial-data'
  | 'device-offline'
  | 'no-baseline'
  | 'setup-incomplete'
  | 'away-mode'
  | 'permission-restricted'
  | 'summary-processing'
  | 'summary-failed';

export type ProvenanceLabel = 'observed' | 'what-mum-shared' | 'compared-with-usual' | 'suggested-next-step' | 'limitations';

export const SETUP_CATEGORIES: ReadonlyArray<{ id: SetupCategory; title: string; description: string }> = [
  { id: 'household', title: 'Household', description: 'Add the people you care for.' },
  { id: 'pair-device', title: 'Pair device', description: 'Connect a Mirror, Bear, App or other supported device.' },
  { id: 'language-accessibility', title: 'Language & accessibility', description: 'Choose familiar, comfortable settings.' },
  { id: 'routines', title: 'Routines', description: 'Set up gentle prompts that fit the day.' },
  { id: 'notifications', title: 'Notifications', description: 'Decide which updates you would like to receive.' },
  { id: 'consent-control', title: 'Older-Adult Consent & Control', description: 'Review choices together, in plain language.' },
  { id: 'care-circle', title: 'Care Circle', description: 'Invite the people who help you care.' },
  { id: 'research-participation', title: 'Research participation', description: 'Choose separately whether to support optional research.' },
] as const;

export const ROUTINE_CATEGORIES: ReadonlyArray<{ id: RoutineCategory; title: string }> = [
  { id: 'medication', title: 'Medication' },
  { id: 'meals', title: 'Meals' },
  { id: 'hydration', title: 'Hydration' },
  { id: 'medical-appointments', title: 'Medical appointments' },
  { id: 'exercise', title: 'Exercise' },
  { id: 'family-events', title: 'Family events' },
  { id: 'custom-other', title: 'Custom / Other' },
] as const;

export const NOTIFICATION_TRIGGERS: ReadonlyArray<{ id: NotificationTrigger; title: string }> = [
  { id: 'conversation-session-summary', title: 'Conversation session summary' },
  { id: 'no-interaction-yet-today', title: 'No interaction yet today' },
  { id: 'repeated-missed-interactions', title: 'Repeated missed interactions' },
  { id: 'recent-interaction-shorter-than-usual', title: 'Recent interaction shorter than usual' },
  { id: 'device-may-be-offline', title: 'Device may be offline' },
  { id: 'reminder-not-completed-or-unclear', title: 'Reminder not completed or unclear' },
  { id: 'new-chat-reply', title: 'New chat reply' },
  { id: 'weekly-summary', title: 'Weekly summary' },
] as const;

export const BASELINE_RULE = { minimumValidSessions: 2, rollingDays: 14 } as const;
