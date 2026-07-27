// Fixture responses for every v1 endpoint the screens read, served by request interception so nothing
// touches production. Content is chosen to STRESS the layout rather than to look good: a long name, a long
// reason sentence, a long mirror name, and enough alerts to fill a list.

const TENANT = 'ten_shot'
const USER = 'usr_shot'
export const PATIENT_A = 'pat_shot_long'
export const PATIENT_B = 'pat_shot_short'

const envelope = (data, meta) => ({ data, meta: meta || { requestId: 'shot' } })

export const V1_SESSION = {
  accessToken: 'shot.access.token',
  refreshToken: 'shot.refresh.token',
  accessTokenExpiresAt: new Date(Date.now() + 3_600_000).toISOString(),
  refreshTokenExpiresAt: new Date(Date.now() + 86_400_000).toISOString(),
  actor: { userId: USER, tenantId: TENANT, name: 'Wei Ling Tan', email: 'weiling@example.com', roles: ['caregiver'] },
}

export const AUTH_SESSION = { userId: USER, name: 'Wei Ling Tan', email: 'weiling@example.com' }

const patients = [
  {
    patientId: PATIENT_A, displayName: 'Grandma Siew Lan Tan', preferredLanguage: 'mandarin',
    timezone: 'Asia/Singapore', ageBand: '85_plus', status: 'active', version: 3,
    profile: { age: 87, gender: 'female', photoUrl: null, phoneNumber: '+65 9123 4567', speechSpeed: 'slow' },
  },
  {
    patientId: PATIENT_B, displayName: 'Nana', preferredLanguage: 'english',
    timezone: 'Asia/Singapore', ageBand: '75_84', status: 'active', version: 1,
    profile: { age: 79, gender: 'female', photoUrl: null, phoneNumber: '+65 8000 1111', speechSpeed: 'normal' },
  },
]

const status = (patientId, overrides) => ({
  patientId, baselineState: 'establishing',
  baselineProgress: { completedSessions: 8, requiredSessions: 12, windowDays: 14 },
  status: 'doing_well', primaryReason: 'ROUTINE_STEADY', secondaryReasons: [],
  completedToday: true, technicalState: 'ok',
  lastInteractionAt: new Date(Date.now() - 3 * 3_600_000).toISOString(),
  conversationsToday: 2, checkinsToday: 1, chatsToday: 1,
  updatedAt: new Date().toISOString(), ...overrides,
})

const notifications = [
  { notificationId: 'notif_1', patientId: PATIENT_A, type: 'completion', state: 'unread',
    title: 'Grandma Siew Lan Tan checked in today', body: "Today's check-in is done.",
    source: 'Checked in', localDate: '2026-07-27', createdAt: new Date(Date.now() - 2 * 3_600_000).toISOString(), readAt: null },
  { notificationId: 'notif_2', patientId: PATIENT_A, type: 'missed_7pm', state: 'unread',
    title: 'No check-in yet today', body: 'Grandma Siew Lan Tan has not had a check-in yet today.',
    source: 'No check-in yet', localDate: '2026-07-26', createdAt: new Date(Date.now() - 20 * 3_600_000).toISOString(), readAt: null },
  { notificationId: 'notif_3', patientId: PATIENT_B, type: 'technical_issue', state: 'read',
    title: 'The mirror may be offline', body: "We cannot reach Nana's mirror right now. This looks like a device or connection problem, not something about Nana.",
    source: 'Connection', localDate: '2026-07-26', createdAt: new Date(Date.now() - 30 * 3_600_000).toISOString(),
    readAt: new Date(Date.now() - 25 * 3_600_000).toISOString() },
  { notificationId: 'notif_4', patientId: PATIENT_A, type: 'streak', state: 'read',
    title: 'Three quiet days in a row', body: 'Grandma Siew Lan Tan has not checked in for three days.',
    source: 'Quiet stretch', localDate: '2026-07-24', createdAt: new Date(Date.now() - 72 * 3_600_000).toISOString(),
    readAt: new Date(Date.now() - 70 * 3_600_000).toISOString() },
]

const carePlan = (patientId) => ({
  patientId, version: 2,
  dailyRoutine: { wakeTime: '07:30' },
  communicationPreferences: { topics: ['family', 'food', 'travel'], otherTopic: 'her old sewing shop in Chinatown',
    speechSpeed: 'slow', speechOrHearingNotes: 'Hard of hearing on the left side; speak a little slower than usual.' },
  safetyNotes: null,
})

const sessionDay = (patientId, date) => ({
  patientId, date, patientName: patients.find((p) => p.patientId === patientId)?.displayName || 'Grandma',
  sessions: [{
    id: 'ses_shot_1', patientId, patientName: 'Grandma Siew Lan Tan', duration: 96, words: 142, exchanges: 6,
    avgLatency: 1.4, createdAt: new Date(Date.now() - 4 * 3_600_000).toISOString(), updatedAt: null,
    logs: [
      { sentence: '早安，今天感觉怎么样？', role: 'assistant', words: 8, duration: 3, wordsPerSecond: 2.6 },
      { sentence: '还不错啦，早上吃了粥，然后在楼下走了一圈。', role: 'patient', words: 18, duration: 7, wordsPerSecond: 2.5 },
      { sentence: '听起来很好。有人来看你吗？', role: 'assistant', words: 9, duration: 3, wordsPerSecond: 3 },
      { sentence: '女儿有打电话来，我们聊了周末要去哪里吃饭。', role: 'patient', words: 19, duration: 8, wordsPerSecond: 2.4 },
    ],
  }],
})

const trend = (days) => ({
  patientId: PATIENT_A, days,
  trend: Array.from({ length: days }, (_, index) => {
    const day = new Date(Date.now() - (days - 1 - index) * 86_400_000).toISOString().slice(0, 10)
    const missed = index % 5 === 3
    return { date: day, duration: missed ? 0 : 70 + ((index * 13) % 60), sessionCount: missed ? 0 : 1,
      status: index % 7 === 5 ? 'amber' : 'green', missed }
  }),
})

/** path (after /api/v1) -> body. Regex keys are matched in order. */
export const ROUTES = [
  [/^\/me$/, () => envelope({
    userId: USER, tenantId: TENANT, name: 'Wei Ling Tan', email: 'weiling@example.com', roles: ['caregiver'],
    phoneNumber: '+65 9777 8888', relationshipToElderly: 'grandma',
    notificationPreferences: { pushNotificationsEnabled: true, alertSensitivity: 'only_important_changes', preferredDailySummaryTime: '19:00' },
    storeSessionSummaries: true,
  })],
  [/^\/patients\?/, () => envelope(patients, { requestId: 'shot', nextCursor: null })],
  [/^\/patients$/, () => envelope(patients, { requestId: 'shot', nextCursor: null })],
  [/^\/patient-statuses/, () => envelope([
    { patientId: PATIENT_A, outcome: 'ok', status: status(PATIENT_A) },
    // Kept internally consistent on purpose: a patient who has not checked in today must not also report
    // conversations today, or a screenshot shows a contradiction that reads as an app bug.
    { patientId: PATIENT_B, outcome: 'ok', status: status(PATIENT_B, {
      status: 'worth_checking', primaryReason: 'SPOKE_LESS_THAN_USUAL', completedToday: false,
      technicalState: 'possible_issue', lastInteractionAt: new Date(Date.now() - 32 * 3_600_000).toISOString(),
      conversationsToday: 0, checkinsToday: 0, chatsToday: 0 }) },
  ])],
  [/^\/patients\/[^/]+\/status$/, (p) => envelope(status(p.split('/')[2]))],
  [/^\/patients\/[^/]+\/consents$/, (p) => envelope({
    patientId: p.split('/')[2], consents: [], requiredPurposes: ['home_cognitive_monitoring'],
    missingPurposes: p.includes(PATIENT_B) ? ['home_cognitive_monitoring'] : [],
  })],
  [/^\/patients\/[^/]+\/care-plan$/, (p) => envelope(carePlan(p.split('/')[2]))],
  [/^\/patients\/[^/]+\/monitoring\/baseline$/, (p) => envelope({
    patientId: p.split('/')[2],
    operational: { state: 'establishing', sessionCount: 8, window: { days: 14, requiredSessions: 7, observedDays: 6 }, algorithmVersion: 'v1', revision: 7 },
    longitudinal: { state: 'building' },
  })],
  [/^\/patients\/[^/]+\/session-days\/[\d-]+$/, (p) => {
    const [, , patientId, , date] = p.split('/')
    return envelope(sessionDay(patientId, date))
  }],
  [/^\/patients\/[^/]+\/session-days\?/, (p) => envelope({
    patientId: p.split('/')[2],
    days: Array.from({ length: 31 }, (_, index) => ({
      date: `2026-07-${String(index + 1).padStart(2, '0')}`, day: index + 1,
      count: index % 4 === 2 ? 0 : 1, completedCount: index % 4 === 2 ? 0 : 1,
      hasCompletedSession: index % 4 !== 2,
    })),
  })],
  [/^\/patients\/[^/]+\/session-trend\?days=30/, () => envelope(trend(30))],
  [/^\/patients\/[^/]+\/session-trend/, () => envelope(trend(7))],
  [/^\/patients\/[^/]+\/session-summaries$/, () => envelope({
    patientId: PATIENT_A, date: '2026-07-27',
    summary: 'She sounded settled today. She had porridge in the morning, walked once around the block, and her daughter called about weekend plans. Nothing in the conversation stood out as different from her usual mornings.',
    reason: null,
  }, undefined)],
  [/^\/device-assignments$/, () => envelope({
    assignments: [
      { patientId: PATIENT_A, patientName: 'Grandma Siew Lan Tan', timezone: 'Asia/Singapore',
        assignmentId: 'asg_1', deviceId: 'dev_4873b9517dae49409821a8a0d0bc9c29',
        mirrorName: "Grandma's bedroom mirror upstairs", assignedAt: new Date(Date.now() - 9 * 86_400_000).toISOString(),
        device: { serial: 'MIR-0042', softwareVersion: '1.0.3', status: 'active', lastHeartbeatAt: new Date(Date.now() - 600_000).toISOString() } },
      { patientId: PATIENT_B, patientName: 'Nana', timezone: 'Asia/Singapore',
        assignmentId: null, deviceId: null, mirrorName: null, assignedAt: null, device: null },
    ],
  })],
  [/^\/notifications/, () => envelope(notifications, { requestId: 'shot', nextCursor: null })],
  [/^\/notification-devices$/, () => envelope({ deviceId: 'ndev_1', platform: 'web', state: 'active', registeredAt: new Date().toISOString() })],
  [/^\/support\/threads$/, () => envelope([])],
  [/^\/auth\/sessions$/, () => envelope(V1_SESSION, undefined)],
]

export function respondFor(path) {
  for (const [pattern, build] of ROUTES) {
    if (pattern.test(path)) return build(path)
  }
  return null
}
