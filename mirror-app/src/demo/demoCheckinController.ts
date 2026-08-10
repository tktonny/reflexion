export type DemoCheckInTopic =
  | 'general'
  | 'recentRecall'
  | 'planning'
  | 'routine'
  | 'reminiscence'
  | 'familyMessage'
  | 'warmClose'

export type DemoTopicStatus = 'Not covered' | 'Covered' | 'Skipped' | 'Not applicable' | 'Pending'
export type DemoCheckInState = 'Ready' | 'In progress' | 'Partially complete' | 'Complete'
export type DemoCheckInScenario = 'standard' | 'full' | 'multi-topic' | 'loved-one-question' | 'natural-follow-up' | 'family-voice-reply' | 'partial' | 'complete' | 'clarification'

export type DemoCheckInFixture = {
  routineDue: boolean
  reminiscenceDue: boolean
  familyMessagePending: boolean
}

export type DemoCheckInTopics = Record<DemoCheckInTopic, DemoTopicStatus>

export type DemoLastUserTurn = {
  text: string
  topicsDetected: DemoCheckInTopic[]
  directQuestion: boolean
  followUpUsed: boolean
  nextCoreTopic: DemoCheckInTopic | null
}

export type DemoCheckInSnapshot = {
  state: DemoCheckInState
  scenario: DemoCheckInScenario
  fixture: DemoCheckInFixture
  topics: DemoCheckInTopics
  clarificationFor: DemoCheckInTopic | null
  responseCount: number
  lastUserTurn: DemoLastUserTurn | null
}

export type DemoCheckInResult = {
  kind: 'free_talk' | 'prompt' | 'question' | 'reprompt' | 'complete' | 'continue'
  snapshot: DemoCheckInSnapshot
  assistantText: string
  topic?: DemoCheckInTopic
}

export const DEMO_CHECKIN_DEFAULT_FIXTURE: DemoCheckInFixture = {
  routineDue: false,
  reminiscenceDue: false,
  familyMessagePending: false,
}

export const DEMO_CHECKIN_PROMPTS: Record<DemoCheckInTopic, string> = {
  general: 'Good morning, Margaret. How are you feeling today?',
  recentRecall: 'What did you get up to yesterday?',
  planning: 'What would you like to do today?',
  routine: 'It is time for your morning tablets. Would you like to take them now?',
  reminiscence: 'Is there a happy memory you would like to share with me today?',
  familyMessage: 'Mei has left you a message. Would you like to hear it?',
  warmClose: 'Thank you for sharing with me today. Is there anything else you would like to say?',
}

const TOPIC_ORDER: DemoCheckInTopic[] = [
  'general',
  'recentRecall',
  'planning',
  'routine',
  'reminiscence',
  'familyMessage',
  'warmClose',
]

export function createDemoCheckInSnapshot(
  scenario: DemoCheckInScenario = 'standard',
  fixture: DemoCheckInFixture = fixtureForScenario(scenario),
): DemoCheckInSnapshot {
  const topics = createDemoCheckInTopics(fixture)
  if (scenario === 'complete') {
    for (const topic of TOPIC_ORDER) {
      if (topics[topic] !== 'Not applicable') topics[topic] = 'Covered'
    }
  }
  return {
    state: scenario === 'complete' ? 'Complete' : 'Ready',
    scenario,
    fixture,
    topics,
    clarificationFor: null,
    responseCount: 0,
    lastUserTurn: null,
  }
}

export function fixtureForScenario(scenario: DemoCheckInScenario): DemoCheckInFixture {
  if (scenario === 'full' || scenario === 'complete') {
    return { routineDue: true, reminiscenceDue: true, familyMessagePending: true }
  }
  if (scenario === 'family-voice-reply') return { ...DEMO_CHECKIN_DEFAULT_FIXTURE, familyMessagePending: true }
  return { ...DEMO_CHECKIN_DEFAULT_FIXTURE }
}

export function createDemoCheckInTopics(fixture: DemoCheckInFixture): DemoCheckInTopics {
  return {
    general: 'Not covered',
    recentRecall: 'Not covered',
    planning: 'Not covered',
    routine: fixture.routineDue ? 'Pending' : 'Not applicable',
    reminiscence: fixture.reminiscenceDue ? 'Pending' : 'Not applicable',
    familyMessage: fixture.familyMessagePending ? 'Pending' : 'Not applicable',
    warmClose: 'Not covered',
  }
}

export function nextDemoCheckInTopic(snapshot: DemoCheckInSnapshot): DemoCheckInTopic | null {
  return TOPIC_ORDER.find((topic) => snapshot.topics[topic] === 'Not covered' || snapshot.topics[topic] === 'Pending') || null
}

export function startDemoCheckIn(snapshot: DemoCheckInSnapshot): DemoCheckInResult {
  if (snapshot.state === 'Complete') {
    return {
      kind: 'free_talk',
      snapshot,
      assistantText: 'What would you like to talk about?',
    }
  }
  const topic = nextDemoCheckInTopic(snapshot)
  if (!topic) {
    const complete = markTopic(snapshot, 'warmClose', 'Covered', 'Complete')
    return { kind: 'complete', snapshot: complete, assistantText: `Thank you, Margaret.` }
  }
  const next = { ...snapshot, state: 'In progress' as DemoCheckInState }
  return { kind: 'prompt', snapshot: next, assistantText: DEMO_CHECKIN_PROMPTS[topic], topic }
}

export function applyDemoCheckInResponse(snapshot: DemoCheckInSnapshot, rawText: string): DemoCheckInResult {
  if (snapshot.state === 'Complete') {
    return { kind: 'free_talk', snapshot, assistantText: 'Tell me more about that.' }
  }

  const currentTopic = nextDemoCheckInTopic(snapshot)
  if (!currentTopic) {
    if (snapshot.topics.warmClose === 'Covered') {
      const complete = { ...snapshot, state: 'Complete' as DemoCheckInState }
      return { kind: 'complete', snapshot: complete, assistantText: `Thank you, Margaret.` }
    }
    const complete = markTopic(snapshot, 'warmClose', 'Covered', 'Complete')
    return { kind: 'complete', snapshot: complete, assistantText: `Thank you, Margaret.` }
  }

  const text = rawText.trim()
  if (isLovedOneQuestion(text)) {
    const next = withLastUserTurn({ ...snapshot, state: 'In progress', responseCount: snapshot.responseCount + 1 }, {
      text, topicsDetected: [], directQuestion: true, followUpUsed: true, nextCoreTopic: currentTopic,
    })
    return {
      kind: 'question',
      snapshot: next,
      assistantText: `I do not have a live weather report in this local demo, but we can take things at your pace. ${DEMO_CHECKIN_PROMPTS[currentTopic]}`,
      topic: currentTopic,
    }
  }

  if (isUnclearResponse(text)) {
    if (snapshot.clarificationFor !== currentTopic) {
      return {
        kind: 'reprompt',
        snapshot: withLastUserTurn({ ...snapshot, state: 'In progress', clarificationFor: currentTopic }, {
          text, topicsDetected: [], directQuestion: false, followUpUsed: false, nextCoreTopic: currentTopic,
        }),
        assistantText: `I did not quite catch that. ${DEMO_CHECKIN_PROMPTS[currentTopic]}`,
        topic: currentTopic,
      }
    }
    const skipped = markTopic({ ...snapshot, clarificationFor: null, responseCount: snapshot.responseCount + 1 }, currentTopic, 'Skipped', 'In progress')
    return continueAfterUpdate(skipped)
  }

  if (snapshot.scenario === 'natural-follow-up' && snapshot.responseCount === 0) {
    const next = withLastUserTurn({ ...snapshot, state: 'In progress', responseCount: snapshot.responseCount + 1, clarificationFor: null }, {
      text, topicsDetected: [currentTopic], directQuestion: false, followUpUsed: true, nextCoreTopic: currentTopic,
    })
    return {
      kind: 'continue',
      snapshot: next,
      assistantText: `Thank you for telling me. Could you share a little more about that?`,
      topic: currentTopic,
    }
  }

  const covered = topicsCoveredByResponse(snapshot, currentTopic, text)
  let next: DemoCheckInSnapshot = { ...snapshot, state: 'In progress', clarificationFor: null, responseCount: snapshot.responseCount + 1 }
  for (const topic of covered) {
    if (next.topics[topic] === 'Not covered' || next.topics[topic] === 'Pending') {
      next = markTopic(next, topic, 'Covered', 'In progress')
    }
  }
  next = withLastUserTurn(next, {
    text,
    topicsDetected: covered,
    directQuestion: /\?/.test(text),
    followUpUsed: false,
    nextCoreTopic: nextDemoCheckInTopic(next),
  })
  return continueAfterUpdate(next)
}

export function stopDemoCheckIn(snapshot: DemoCheckInSnapshot): DemoCheckInSnapshot {
  if (snapshot.state === 'Complete') return snapshot
  return { ...snapshot, state: 'Partially complete', clarificationFor: null }
}

export function topicLabel(topic: DemoCheckInTopic): string {
  switch (topic) {
    case 'general': return 'General check-in'
    case 'recentRecall': return 'Recent recall'
    case 'planning': return 'Planning'
    case 'routine': return 'Routine branch'
    case 'reminiscence': return 'Reminiscence'
    case 'familyMessage': return 'Family message'
    default: return 'Warm close'
  }
}

function continueAfterUpdate(snapshot: DemoCheckInSnapshot): DemoCheckInResult {
  const topic = nextDemoCheckInTopic(snapshot)
  if (!topic) {
    if (snapshot.topics.warmClose === 'Covered') {
      const complete = { ...snapshot, state: 'Complete' as DemoCheckInState }
      return { kind: 'complete', snapshot: complete, assistantText: `Thank you, Margaret.` }
    }
    const close = { ...snapshot, state: 'In progress' as DemoCheckInState }
    return { kind: 'continue', snapshot: close, assistantText: DEMO_CHECKIN_PROMPTS.warmClose, topic: 'warmClose' }
  }
  return { kind: 'continue', snapshot, assistantText: DEMO_CHECKIN_PROMPTS[topic], topic }
}

function markTopic(
  snapshot: DemoCheckInSnapshot,
  topic: DemoCheckInTopic,
  status: DemoTopicStatus,
  state: DemoCheckInState,
): DemoCheckInSnapshot {
  return { ...snapshot, state, topics: { ...snapshot.topics, [topic]: status } }
}

function withLastUserTurn(snapshot: DemoCheckInSnapshot, lastUserTurn: DemoLastUserTurn): DemoCheckInSnapshot {
  return { ...snapshot, lastUserTurn }
}

function topicsCoveredByResponse(snapshot: DemoCheckInSnapshot, currentTopic: DemoCheckInTopic, text: string): DemoCheckInTopic[] {
  if (snapshot.scenario === 'multi-topic' && snapshot.responseCount === 0) {
    return ['general', 'recentRecall', 'planning']
  }
  const normalized = text.toLowerCase()
  const covered: DemoCheckInTopic[] = [currentTopic]
  if (currentTopic === 'general' && /(yesterday|last night|slept|visited|saw|came)/.test(normalized)) covered.push('recentRecall')
  if (/(today|plan|going to|breakfast|market|later|look forward)/.test(normalized)) covered.push('planning')
  if (/(yesterday|last night|slept|visited|saw|came)/.test(normalized)) covered.push('recentRecall')
  return covered
}

function isLovedOneQuestion(text: string): boolean {
  return /\b(weather|temperature|rain|raining|forecast|outside)\b/i.test(text)
}

function isUnclearResponse(text: string): boolean {
  return !text || /^(hmm+|um+|uh+|i don't know|not sure|what|\?)$/i.test(text.trim())
}
