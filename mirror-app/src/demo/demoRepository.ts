import AsyncStorage from '@react-native-async-storage/async-storage'
import type { MirrorHomeStatus } from '../components/mirror/MirrorExperience'
import { DEMO_STORAGE_NAMESPACE } from './demoConfig'
import {
  createDemoCheckInSnapshot,
  type DemoCheckInScenario,
  type DemoCheckInSnapshot,
} from './demoCheckinController'

export type DemoRoutineResponse = 'pending' | 'complete' | 'deferred' | 'declined'
export type DemoConsentState = 'Accepted' | 'Declined' | 'Withdrawn'
export type DemoResearchState = 'Not invited' | 'Invitation pending' | 'Consented' | 'Declined' | 'Withdrawn' | 'Study closed'
export type DemoMessageKind = 'text' | 'photo' | 'voice'

export type DemoMirrorMessage = {
  messageId: string
  kind: DemoMessageKind
  senderName: string
  text?: string
  caption?: string
  preview: string
}

export type DemoVoiceReply = {
  state: 'idle' | 'sent'
  sentAt: string | null
}

export type DemoMirrorState = {
  homeStatus: MirrorHomeStatus
  routineResponse: DemoRoutineResponse
  consent: DemoConsentState
  research: DemoResearchState
  checkIn: DemoCheckInSnapshot
  activeMessageId: string
  messageInteractions: Record<string, 'Viewed' | 'Played' | 'Replayed'>
  voiceReply: DemoVoiceReply
}

export const DEMO_PATIENT = {
  name: 'Margaret',
  caregiver: 'Mei',
  greeting: 'Good morning',
}

export const DEMO_MESSAGES: DemoMirrorMessage[] = [
  {
    messageId: 'demo-text-message',
    kind: 'text',
    senderName: 'Mei',
    text: 'Thinking of you, Margaret. I hope you have a lovely day.',
    preview: 'Thinking of you and sending a big hug.',
  },
  {
    messageId: 'demo-photo-message',
    kind: 'photo',
    senderName: 'Mei',
    caption: 'A little memory from our garden day.',
    preview: 'A photo from Mei.',
  },
  {
    messageId: 'demo-voice-message',
    kind: 'voice',
    senderName: 'Mei',
    text: 'A short voice message from Mei.',
    preview: 'A voice message from Mei.',
  },
]

export const DEMO_MIRROR_DEFAULTS: DemoMirrorState = {
  homeStatus: 'ready',
  routineResponse: 'pending',
  consent: 'Accepted',
  research: 'Invitation pending',
  checkIn: createDemoCheckInSnapshot('standard'),
  activeMessageId: DEMO_MESSAGES[0].messageId,
  messageInteractions: {},
  voiceReply: { state: 'idle', sentAt: null },
}

// v2 deliberately starts the web demo with a Ready check-in instead of inheriting older controller state.
const DEMO_MIRROR_STATE_KEY = `${DEMO_STORAGE_NAMESPACE}mirror-state-v2`
const DEMO_VOICE_KEY = `${DEMO_STORAGE_NAMESPACE}voice-uri`

function normalizeState(value: unknown): DemoMirrorState {
  if (!value || typeof value !== 'object') return { ...DEMO_MIRROR_DEFAULTS, messageInteractions: {} }
  const candidate = value as Partial<DemoMirrorState>
  const homeStatus: MirrorHomeStatus = ['ready', 'partial', 'complete', 'paused', 'away', 'offline'].includes(String(candidate.homeStatus))
    ? candidate.homeStatus as MirrorHomeStatus
    : DEMO_MIRROR_DEFAULTS.homeStatus
  const routineResponse: DemoRoutineResponse = ['pending', 'complete', 'deferred', 'declined'].includes(String(candidate.routineResponse))
    ? candidate.routineResponse as DemoRoutineResponse
    : DEMO_MIRROR_DEFAULTS.routineResponse
  const consent: DemoConsentState = ['Accepted', 'Declined', 'Withdrawn'].includes(String(candidate.consent))
    ? candidate.consent as DemoConsentState
    : DEMO_MIRROR_DEFAULTS.consent
  const research: DemoResearchState = ['Not invited', 'Invitation pending', 'Consented', 'Declined', 'Withdrawn', 'Study closed'].includes(String(candidate.research))
    ? candidate.research as DemoResearchState
    : DEMO_MIRROR_DEFAULTS.research
  const checkIn = normalizeCheckIn(candidate.checkIn)
  const messageInteractions = candidate.messageInteractions && typeof candidate.messageInteractions === 'object'
    ? candidate.messageInteractions as DemoMirrorState['messageInteractions']
    : {}
  const voiceReply = candidate.voiceReply && typeof candidate.voiceReply === 'object'
    ? {
        state: (candidate.voiceReply as Partial<DemoVoiceReply>).state === 'sent' ? 'sent' as const : 'idle' as const,
        sentAt: typeof (candidate.voiceReply as Partial<DemoVoiceReply>).sentAt === 'string' ? (candidate.voiceReply as Partial<DemoVoiceReply>).sentAt || null : null,
      }
    : DEMO_MIRROR_DEFAULTS.voiceReply
  return {
    homeStatus,
    routineResponse,
    consent,
    research,
    checkIn,
    activeMessageId: DEMO_MESSAGES.some((message) => message.messageId === candidate.activeMessageId)
      ? String(candidate.activeMessageId)
      : DEMO_MIRROR_DEFAULTS.activeMessageId,
    messageInteractions,
    voiceReply,
  }
}

function normalizeCheckIn(value: unknown): DemoCheckInSnapshot {
  if (!value || typeof value !== 'object') return createDemoCheckInSnapshot('standard')
  const candidate = value as Partial<DemoCheckInSnapshot>
  const scenario: DemoCheckInScenario = ['standard', 'full', 'multi-topic', 'loved-one-question', 'natural-follow-up', 'family-voice-reply', 'partial', 'complete', 'clarification'].includes(String(candidate.scenario))
    ? candidate.scenario as DemoCheckInScenario
    : 'standard'
  const fixture = candidate.fixture && typeof candidate.fixture === 'object'
    ? {
        routineDue: Boolean((candidate.fixture as Partial<DemoCheckInSnapshot['fixture']>).routineDue),
        reminiscenceDue: Boolean((candidate.fixture as Partial<DemoCheckInSnapshot['fixture']>).reminiscenceDue),
        familyMessagePending: Boolean((candidate.fixture as Partial<DemoCheckInSnapshot['fixture']>).familyMessagePending),
      }
    : undefined
  const fresh = createDemoCheckInSnapshot(scenario, fixture)
  const state = ['Ready', 'In progress', 'Partially complete', 'Complete'].includes(String(candidate.state))
    ? candidate.state as DemoCheckInSnapshot['state']
    : fresh.state
  const validTopics = candidate.topics && typeof candidate.topics === 'object' ? candidate.topics as Partial<DemoCheckInSnapshot['topics']> : {}
  const topics = { ...fresh.topics }
  for (const topic of Object.keys(topics) as Array<keyof typeof topics>) {
    const status = validTopics[topic]
    if (status && ['Not covered', 'Covered', 'Skipped', 'Not applicable', 'Pending'].includes(status)) topics[topic] = status
  }
  return {
    state,
    scenario,
    fixture: fresh.fixture,
    topics,
    clarificationFor: Object.keys(topics).includes(String(candidate.clarificationFor)) ? candidate.clarificationFor || null : null,
    responseCount: Number.isFinite(candidate.responseCount) ? Math.max(0, Number(candidate.responseCount)) : 0,
    lastUserTurn: candidate.lastUserTurn && typeof candidate.lastUserTurn === 'object'
      ? {
          text: typeof candidate.lastUserTurn.text === 'string' ? candidate.lastUserTurn.text : '',
          topicsDetected: Array.isArray(candidate.lastUserTurn.topicsDetected)
            ? candidate.lastUserTurn.topicsDetected.filter((topic): topic is keyof typeof topics => Object.keys(topics).includes(String(topic)))
            : [],
          directQuestion: Boolean(candidate.lastUserTurn.directQuestion),
          followUpUsed: Boolean(candidate.lastUserTurn.followUpUsed),
          nextCoreTopic: Object.keys(topics).includes(String(candidate.lastUserTurn.nextCoreTopic)) ? candidate.lastUserTurn.nextCoreTopic || null : null,
        }
      : null,
  }
}

export async function getDemoMirrorState(): Promise<DemoMirrorState> {
  try {
    const raw = await AsyncStorage.getItem(DEMO_MIRROR_STATE_KEY)
    return raw ? normalizeState(JSON.parse(raw)) : { ...DEMO_MIRROR_DEFAULTS, messageInteractions: {} }
  } catch {
    return { ...DEMO_MIRROR_DEFAULTS, messageInteractions: {} }
  }
}

export async function updateDemoMirrorState(patch: Partial<DemoMirrorState>): Promise<DemoMirrorState> {
  const next = normalizeState({ ...(await getDemoMirrorState()), ...patch })
  await AsyncStorage.setItem(DEMO_MIRROR_STATE_KEY, JSON.stringify(next))
  return next
}

export async function resetDemoMirrorState(): Promise<void> {
  await AsyncStorage.removeItem(DEMO_MIRROR_STATE_KEY)
}

export async function getDemoVoicePreference(): Promise<string | null> {
  try { return await AsyncStorage.getItem(DEMO_VOICE_KEY) } catch { return null }
}

export async function setDemoVoicePreference(voiceId: string): Promise<void> {
  try { await AsyncStorage.setItem(DEMO_VOICE_KEY, voiceId) } catch { /* local preference is best effort */ }
}

export async function markDemoVoiceReplySent(): Promise<void> {
  await updateDemoMirrorState({ voiceReply: { state: 'sent', sentAt: new Date().toISOString() } })
}

export function getDemoMessage(messageId?: string): DemoMirrorMessage {
  return DEMO_MESSAGES.find((message) => message.messageId === messageId) || DEMO_MESSAGES[0]
}

export function routineLabel(response: DemoRoutineResponse): string {
  switch (response) {
    case 'complete': return 'Reported complete'
    case 'deferred': return 'Remind me later'
    case 'declined': return 'Declined in demo'
    default: return 'Ready to respond'
  }
}
