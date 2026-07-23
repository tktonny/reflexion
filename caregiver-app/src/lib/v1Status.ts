// Presentation layer for the authoritative v1 caregiver status read model
// (reflexion-implementation-baseline.md §4). The app RENDERS what the backend sends — it never
// computes status. Wording is warm and reassurance-first; no clinical or diagnostic terms.

export type V1Status = 'establishing' | 'doing_well' | 'worth_checking' | 'needs_attention';
export type V1TechnicalState = 'ok' | 'possible_issue' | 'unreachable' | 'unknown';

export type V1PatientStatus = {
  patientId: string;
  baselineState: 'establishing' | 'complete';
  baselineProgress: { completedSessions: number; requiredSessions: number; windowDays: number };
  status: V1Status;
  primaryReason: string;
  secondaryReasons: string[];
  completedToday: boolean;
  technicalState: V1TechnicalState;
  lastInteractionAt: string | null;
  updatedAt: string;
};

// Option-1 muted status palette (doc §2.9 / task spec). Small dot/pill, never loud blocks.
export const STATUS_META: Record<V1Status, { color: string; dot: string; emoji: string; label: string }> = {
  establishing: { color: '#8E877C', dot: '#8E877C', emoji: '⚪', label: 'Learning routine' },
  doing_well: { color: '#596C56', dot: '#596C56', emoji: '🟢', label: 'Doing well' },
  worth_checking: { color: '#9A7A45', dot: '#9A7A45', emoji: '🟡', label: 'Worth checking' },
  needs_attention: { color: '#9B5F4E', dot: '#9B5F4E', emoji: '🔴', label: 'Needs attention' },
};

// Neutral fallback while a status is loading / unavailable. Never red, never legacy.
export const NEUTRAL_STATUS_COLOR = '#B4ADA2';

export function firstName(name?: string | null): string {
  return (name || '').trim().split(/\s+/)[0] || '';
}

export function getStatusLabel(status: V1Status, name?: string): string {
  if (status === 'establishing') {
    const first = firstName(name);
    return first ? `Learning ${first}'s routine` : 'Learning their routine';
  }
  return STATUS_META[status].label;
}

// Plain-English mapping for every reason code in §4. Warm, non-clinical phrasing.
export function getReasonText(code: string | null | undefined, name?: string): string {
  const first = firstName(name);
  const whos = first ? `${first}'s` : 'their';
  const who = first || 'them';

  const map: Record<string, string> = {
    LEARNING_PERSONAL_ROUTINE: `Still getting to know ${whos} daily routine.`,
    DAILY_PATTERN_ON_TRACK: `Following ${whos} usual daily pattern.`,
    CHECKIN_COMPLETED_TODAY: `Had ${whos} check-in today.`,
    CHECKIN_MISSED_TODAY: 'No check-in yet today.',
    CHECKIN_MISSED_REPEATEDLY: 'A few check-ins missed recently.',
    CHECKIN_MISSED_3_DAYS: 'No check-in for about three days.',
    CHECKIN_OUTSIDE_USUAL_WINDOW: 'Checked in at a different time than usual.',
    WEEKLY_ENGAGEMENT_DOWN: 'A little quieter this week than usual.',
    SPOKE_LESS_THAN_USUAL: 'Spoke a little less than usual.',
    FEWER_RESPONSES: 'Gave fewer responses than usual.',
    SLOWER_TO_RESPOND: 'Took a little longer to respond than usual.',
    DEVICE_UNREACHABLE: 'The mirror may be offline right now.',
    AWAY_PERIOD_ACTIVE: 'Marked as away at the moment.',
    CAREGIVER_FLAG_WORTH_CHECKING: `You asked to keep an eye on ${who}.`,
    CAREGIVER_FLAG_NEEDS_ATTENTION: `You flagged ${who} as needing attention.`,
  };

  return map[code ?? ''] ?? 'Everything looks steady.';
}

// Device/technical framing — always presented as a connection issue, never as personal decline.
export function getTechnicalNote(state: V1TechnicalState): string | null {
  if (state === 'unreachable') {
    return 'The mirror may be offline. This looks like a device connection issue, not a change in how they are doing.';
  }
  if (state === 'possible_issue') {
    return 'There may be a minor issue with the mirror connection.';
  }
  return null;
}

export function getBaselineProgressText(progress: V1PatientStatus['baselineProgress']): string {
  const completed = Number(progress?.completedSessions || 0);
  const required = Number(progress?.requiredSessions || 7);
  return `${completed} of ${required} sessions recorded`;
}

// "Today, 8:15am" / "Yesterday, 7:52am" / "2 days ago" / "No check-in yet".
export function formatLastInteraction(iso: string | null | undefined): string {
  if (!iso) {
    return 'No check-in yet';
  }
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return 'No check-in yet';
  }

  const time = new Intl.DateTimeFormat('en-SG', { hour: 'numeric', minute: '2-digit' })
    .format(date)
    .replace(/\s/g, '')
    .toLowerCase();

  const startOfDay = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const dayDiff = Math.round((startOfDay(new Date()) - startOfDay(date)) / 86_400_000);

  if (dayDiff <= 0) {
    return `Today, ${time}`;
  }
  if (dayDiff === 1) {
    return `Yesterday, ${time}`;
  }
  return `${dayDiff} days ago`;
}
