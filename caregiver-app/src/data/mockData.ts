export type Status = 'green' | 'yellow' | 'red';

export interface ElderlyProfile {
  id: string;
  name: string;
  nickname: string;
  age: number;
  gender: 'male' | 'female';
  language: 'English' | 'Mandarin';
  wakeTime: string;
  photo: string | null;
  topics: string[];
  conditions: string;
  mirrorId: string | null;
  caregiverId: string;
}

export interface Session {
  id: string;
  elderlyId: string;
  date: string;
  time: string;
  duration: number;
  wordCount: number;
  responseLatency: number;
  exchanges: number;
  status: Status;
  summary: string;
  topics: string[];
  transcript: TranscriptLine[];
  audioUrl: string | null;
}

export interface TranscriptLine {
  speaker: 'Aria' | 'User';
  text: string;
  timestamp: number;
}

export interface Alert {
  id: string;
  elderlyId: string;
  elderlyName: string;
  type: 'missed_routine' | 'reduced_engagement' | 'no_interaction' | 'speech_change' | 'positive';
  title: string;
  message: string;
  action: 'call' | 'note' | 'none';
  timestamp: string;
  read: boolean;
}

export interface TrendDay {
  date: string;
  duration: number;
  status: Status;
  missed: boolean;
}

export interface Caregiver {
  id: string;
  name: string;
  email: string;
  phone: string;
  relationship: string;
  notificationPreference: 'all' | 'important' | 'urgent';
  dailySummaryTime: 'morning' | 'evening';
}

// ── Mock Caregiver ──────────────────────────────────────────────────────────
export const MOCK_CAREGIVER: Caregiver = {
  id: 'cg-001',
  name: 'Sarah Lim',
  email: 'sarah.lim@email.com',
  phone: '+65 9123 4567',
  relationship: 'child',
  notificationPreference: 'important',
  dailySummaryTime: 'morning',
};

// ── Mock Elderly Profiles ───────────────────────────────────────────────────
export const MOCK_ELDERLY: ElderlyProfile[] = [
  {
    id: 'el-001',
    name: 'Margaret Lim',
    nickname: 'Grandma Lim',
    age: 78,
    gender: 'female',
    language: 'English',
    wakeTime: '07:30',
    photo: null,
    topics: ['Family', 'Food', 'Travel'],
    conditions: '',
    mirrorId: 'mirror-001',
    caregiverId: 'cg-001',
  },
  {
    id: 'el-002',
    name: 'Robert Tan',
    nickname: 'Grandpa Tan',
    age: 82,
    gender: 'male',
    language: 'Mandarin',
    wakeTime: '08:00',
    photo: null,
    topics: ['Work', 'Food', 'Family'],
    conditions: '',
    mirrorId: 'mirror-002',
    caregiverId: 'cg-001',
  },
  {
    id: 'el-003',
    name: 'Agnes Chan',
    nickname: 'Grandma Chan',
    age: 75,
    gender: 'female',
    language: 'English',
    wakeTime: '07:00',
    photo: null,
    topics: ['Family', 'Travel'],
    conditions: 'Mild hearing loss',
    mirrorId: null,
    caregiverId: 'cg-001',
  },
];

// ── Mock Sessions ───────────────────────────────────────────────────────────
const MOCK_TRANSCRIPT: TranscriptLine[] = [
  { speaker: 'Aria', text: "Good morning Margaret, it's lovely to see you. How are you feeling today?", timestamp: 0 },
  { speaker: 'User', text: "I'm feeling quite well today, thank you. Slept better last night.", timestamp: 4 },
  { speaker: 'Aria', text: "That's wonderful to hear! What did you have for dinner yesterday?", timestamp: 9 },
  { speaker: 'User', text: "Oh I had fishball noodles. My daughter made it for me. It was very good.", timestamp: 13 },
  { speaker: 'Aria', text: "That sounds delicious! What are you planning to do today?", timestamp: 22 },
  { speaker: 'User', text: "I think I'll go to the wet market later this morning. I need to buy some vegetables.", timestamp: 27 },
  { speaker: 'Aria', text: "How lovely! Is anyone visiting you this week?", timestamp: 38 },
  { speaker: 'User', text: "Yes, my grandson is coming on Saturday. I'm looking forward to that.", timestamp: 42 },
  { speaker: 'Aria', text: "That sounds wonderful, I'm sure he loves spending time with you. Tell me about a holiday you loved.", timestamp: 53 },
  { speaker: 'User', text: "Oh, we went to Cameron Highlands many years ago. The weather was so cool and we picked strawberries. My children were young then.", timestamp: 60 },
  { speaker: 'Aria', text: "That sounds like such a beautiful memory. Thank you so much for chatting with me today, Margaret. Enjoy your morning!", timestamp: 85 },
  { speaker: 'User', text: "Thank you Aria. See you tomorrow!", timestamp: 92 },
];

export const MOCK_SESSIONS: Session[] = [
  {
    id: 'sess-001',
    elderlyId: 'el-001',
    date: '2026-06-09',
    time: '08:15',
    duration: 272,
    wordCount: 142,
    responseLatency: 2.1,
    exchanges: 6,
    status: 'green',
    summary: "Grandma completed her morning conversation today. She sounded cheerful and mentioned she is planning to go to the wet market later. She recalled having fishball noodles for dinner yesterday. No changes from her usual pattern.",
    topics: ['Market', 'Food', 'Family', 'Weather'],
    transcript: MOCK_TRANSCRIPT,
    audioUrl: null,
  },
  {
    id: 'sess-002',
    elderlyId: 'el-002',
    date: '2026-06-08',
    time: '09:10',
    duration: 185,
    wordCount: 89,
    responseLatency: 3.4,
    exchanges: 4,
    status: 'yellow',
    summary: "Grandpa Tan completed his morning chat but was noticeably quieter than usual. He mentioned feeling a bit tired. The conversation was shorter than his typical sessions this week.",
    topics: ['Family', 'Weather'],
    transcript: [],
    audioUrl: null,
  },
  {
    id: 'sess-003',
    elderlyId: 'el-003',
    date: '2026-06-07',
    time: '07:45',
    duration: 0,
    wordCount: 0,
    responseLatency: 0,
    exchanges: 0,
    status: 'red',
    summary: "Grandma Chan did not interact with Aria today.",
    topics: [],
    transcript: [],
    audioUrl: null,
  },
];

// ── Status helpers ──────────────────────────────────────────────────────────
export function getElderlyStatus(elderlyId: string): Status {
  const map: Record<string, Status> = {
    'el-001': 'green',
    'el-002': 'yellow',
    'el-003': 'red',
  };
  return map[elderlyId] ?? 'yellow';
}

export function getStatusLabel(status: Status): string {
  if (status === 'green') return 'Doing well';
  if (status === 'yellow') return 'Worth checking';
  return 'Needs attention';
}

export function getLastSeen(elderlyId: string): string {
  const map: Record<string, string> = {
    'el-001': 'Last spoke today, 8:15am',
    'el-002': 'Last spoke yesterday, 7:52am',
    'el-003': 'No interaction for 2 days',
  };
  return map[elderlyId] ?? 'Unknown';
}

export function getInitials(nickname: string): string {
  return nickname.split(' ').map(w => w[0]).join('').toUpperCase();
}

export function getLatestSession(elderlyId: string): Session | null {
  return MOCK_SESSIONS.find(s => s.elderlyId === elderlyId) ?? null;
}

// ── Mock Alerts ─────────────────────────────────────────────────────────────
export const MOCK_ALERTS: Alert[] = [
  {
    id: 'alert-001',
    elderlyId: 'el-003',
    elderlyName: 'Grandma Chan',
    type: 'no_interaction',
    title: 'No chat for 2 days',
    message: "Grandma Chan has not interacted with Aria for 2 days. This is unusual for her. It may be worth checking in.",
    action: 'call',
    timestamp: '2026-06-09T10:30:00',
    read: false,
  },
  {
    id: 'alert-002',
    elderlyId: 'el-002',
    elderlyName: 'Grandpa Tan',
    type: 'reduced_engagement',
    title: 'Quieter than usual',
    message: "Grandpa Tan has had shorter conversations this week compared to his usual pattern.",
    action: 'call',
    timestamp: '2026-06-08T09:15:00',
    read: false,
  },
  {
    id: 'alert-003',
    elderlyId: 'el-001',
    elderlyName: 'Grandma Lim',
    type: 'positive',
    title: 'Great morning chat',
    message: "Grandma Lim had a cheerful and lively chat today. She mentioned her upcoming market trip and her grandson visiting!",
    action: 'none',
    timestamp: '2026-06-09T08:20:00',
    read: true,
  },
];

// ── Mock Trend Data ─────────────────────────────────────────────────────────
function generateTrend(baseSeconds: number, variance: number, days: number, missChance: number): TrendDay[] {
  const result: TrendDay[] = [];
  const today = new Date('2026-06-09');
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(today.getDate() - i);
    const missed = Math.random() < missChance;
    const duration = missed ? 0 : Math.max(60, baseSeconds + (Math.random() - 0.5) * variance * 2);
    let status: Status = 'green';
    if (missed) status = 'red';
    else if (duration < baseSeconds * 0.7) status = 'yellow';
    result.push({
      date: d.toISOString().split('T')[0],
      duration: missed ? 0 : Math.round(duration),
      status,
      missed,
    });
  }
  return result;
}

export const MOCK_TRENDS: Record<string, TrendDay[]> = {
  'el-001': generateTrend(270, 60, 90, 0.05),
  'el-002': generateTrend(200, 80, 90, 0.2),
  'el-003': generateTrend(180, 100, 90, 0.35),
};

export const FAQ_ITEMS = [
  {
    q: 'How does Reflexion work?',
    a: 'Reflexion uses a smart mirror with a built-in voice AI companion called Aria. Each morning, Aria has a gentle 5-minute conversation with your loved one. The app shows you a daily summary and alerts you to any changes in their routine.',
  },
  {
    q: 'Does my parent need to do anything to start?',
    a: 'No. Aria greets them automatically each morning at their usual wake time. They just need to talk back. There are no buttons to press and no apps to open.',
  },
  {
    q: 'What does the status colour mean?',
    a: 'Doing well means they had their usual conversation and everything looks normal. Worth checking means the session was shorter or they seem quieter than usual. Needs attention means they have not had a conversation for one or more days.',
  },
  {
    q: 'Will the conversation be stored?',
    a: "Yes, a summary and transcript of each session is stored securely. You can opt in to storing audio recordings in Settings. All data is stored in Singapore and encrypted.",
  },
  {
    q: 'Can I adjust how fast Aria speaks?',
    a: "Yes. Go to Settings and adjust the speech rate per person. You can choose Slow, Normal, or Fast.",
  },
  {
    q: 'How do I link the mirror to the app?',
    a: 'During onboarding, scan the QR code on the back of the Reflexion device. You can also do this later in Settings.',
  },
  {
    q: 'What languages does Aria speak?',
    a: 'Aria currently supports English and Mandarin. Malay and Tamil support are coming in Phase 2.',
  },
  {
    q: 'Can multiple caregivers monitor the same person?',
    a: 'Not yet — this is planned for a future update. Currently, one caregiver account is linked per loved one.',
  },
];
