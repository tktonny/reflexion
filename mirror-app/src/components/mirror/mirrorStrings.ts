// Localized copy for the mirror UI chrome, driven by the patient's language (set in the caregiver app
// → device configuration → the mirror's `language`). Previously every string was hardcoded English.
// English + Chinese are first-class; Chinese covers mandarin/cantonese/minnan buckets, other languages
// fall back to English (their conversation still runs in-language; only the on-screen chrome falls back).

export type MirrorCopy = {
  wakePhraseReady: string
  sayHello: (phrase: string) => string
  tapToBegin: string
  toBegin: string
  micActive: string
  startingChat: string
  todaysConversation: string
  ariaListeningHeader: string
  connectingCaption: string
  hi: (name: string) => string
  listeningTitle: string
  heardTitle: string
  bargeTitle: string
  listeningCaption: string
  bargeCaption: string
  thinkingTitle: string
  thinkingCaption: string
  ariaSpeakingFallback: string
  interruptHint: string
  savingHeader: string
  goodbyeHeader: string
  savingTitle: string
  goodbyeFallback: string
  savingCaption: string
  goodbyeCaption: string
  offlineTitle: string
  micErrorTitle: string
  serviceTitle: string
  offlineBody: string
  micErrorBody: string
  serviceBodyFallback: string
  offlineAssurance: string
  retryOffline: string
  retry: string
  endHint: string
  heardLabel: string
  tapToStartWake: string
  closingEyebrow: string
  closingLetCaregiverKnow: (name: string) => string
  closingHaveLovelyDay: string
  closingSavedSyncing: string
  closingSavedSecurely: string
  closingReturningHome: string
}

const EN: MirrorCopy = {
  wakePhraseReady: 'WAKE PHRASE READY',
  sayHello: (phrase) => `Say “${phrase}”`,
  tapToBegin: 'or tap here to begin',
  toBegin: 'to begin',
  micActive: 'MICROPHONE ACTIVE',
  startingChat: 'STARTING YOUR CHAT',
  todaysConversation: 'TODAY’S CONVERSATION',
  ariaListeningHeader: 'ARIA IS LISTENING',
  connectingCaption: 'Let’s have a short chat.',
  hi: (name) => `Hi ${name},`,
  listeningTitle: 'I’m listening…',
  heardTitle: 'I can hear you.',
  bargeTitle: 'I heard you — I’ve paused.',
  listeningCaption: 'Take your time. I’ll wait until you finish.',
  bargeCaption: 'Please keep speaking.',
  thinkingTitle: 'Thank you.',
  thinkingCaption: 'I’m thinking about what you said.',
  ariaSpeakingFallback: 'Aria is speaking…',
  interruptHint: 'You can speak at any time to interrupt me',
  savingHeader: 'SAVING TODAY’S CHAT',
  goodbyeHeader: 'GOODBYE FOR NOW',
  savingTitle: 'Thank you for chatting with me.',
  goodbyeFallback: 'Have a good day.',
  savingCaption: 'Your check-in is being saved.',
  goodbyeCaption: 'I’ll finish after saying goodbye.',
  offlineTitle: 'Reflexion is offline right now.',
  micErrorTitle: 'I’m having trouble hearing you.',
  serviceTitle: 'Aria needs a moment.',
  offlineBody: 'Please check the connection. Anything already recorded will update your caregiver once we’re connected again.',
  micErrorBody: 'Please check the microphone or restart Reflexion.',
  serviceBodyFallback: 'Please try again in a moment.',
  offlineAssurance: 'Nothing already recorded will be lost.',
  retryOffline: 'Check connection',
  retry: 'Try again',
  endHint: 'Say “goodbye” to finish',
  heardLabel: 'I HEARD YOU',
  tapToStartWake: 'Tap the mirror to start',
  closingEyebrow: 'CONVERSATION COMPLETE',
  closingLetCaregiverKnow: (name) => `I’ll let ${name} know you checked in today.`,
  closingHaveLovelyDay: 'Have a lovely day.',
  closingSavedSyncing: 'Saved safely — will sync when connection returns',
  closingSavedSecurely: 'Saved securely',
  closingReturningHome: 'Returning to your home screen…',
}

const ZH: MirrorCopy = {
  wakePhraseReady: '唤醒词已就绪',
  sayHello: (phrase) => `说 “${phrase}”`,
  tapToBegin: '或点这里开始',
  toBegin: '开始对话',
  micActive: '麦克风已开启',
  startingChat: '正在开始对话',
  todaysConversation: '今天的对话',
  ariaListeningHeader: 'Aria 在聆听',
  connectingCaption: '我们来聊一会儿吧。',
  hi: (name) => `${name}，你好，`,
  listeningTitle: '我在听…',
  heardTitle: '我听得到你。',
  bargeTitle: '我听到你了——我先停一下。',
  listeningCaption: '别着急，慢慢来，我会等你说完。',
  bargeCaption: '请继续说。',
  thinkingTitle: '谢谢你。',
  thinkingCaption: '我在想你刚才说的话。',
  ariaSpeakingFallback: 'Aria 正在说话…',
  interruptHint: '你随时都可以说话打断我',
  savingHeader: '正在保存今天的对话',
  goodbyeHeader: '先到这里',
  savingTitle: '谢谢你今天和我聊天。',
  goodbyeFallback: '祝你今天愉快。',
  savingCaption: '正在保存你的今日记录。',
  goodbyeCaption: '说完再见我就结束。',
  offlineTitle: 'Reflexion 暂时离线了。',
  micErrorTitle: '我有点听不清你说话。',
  serviceTitle: 'Aria 需要缓一下。',
  offlineBody: '请检查网络连接。已经记录的内容会在重新联网后同步给你的家人。',
  micErrorBody: '请检查麦克风，或重启 Reflexion。',
  serviceBodyFallback: '请稍后再试一次。',
  offlineAssurance: '已经记录的内容不会丢失。',
  retryOffline: '检查连接',
  retry: '重试',
  endHint: '说“再见”即可结束',
  heardLabel: '我听到了',
  tapToStartWake: '轻触镜子即可开始',
  closingEyebrow: '对话已完成',
  closingLetCaregiverKnow: (name) => `我会告诉${name}你今天问过好了。`,
  closingHaveLovelyDay: '祝你今天愉快。',
  closingSavedSyncing: '已安全保存 —— 联网后会自动同步',
  closingSavedSecurely: '已安全保存',
  closingReturningHome: '正在返回主界面…',
}

// The UI language is chosen automatically from the patient's backend preference
// (Mongo patients.preferredLanguage → device configuration → the mirror's `language`). Two UI locales
// exist — Chinese and English — and this picks Chinese for ANY Chinese variant the backend might store
// (zh, zh-CN, zh-Hans, cmn, yue, mandarin, cantonese, 中文…), English for everything else. Robust to
// casing/format so a differently-shaped backend value never silently falls back to the wrong locale.
export function isChineseLanguage(language?: string): boolean {
  const key = String(language || '').trim().toLowerCase()
  if (!key) return false
  if (key.startsWith('zh') || key.startsWith('cmn') || key.startsWith('yue') || key.startsWith('nan')) return true
  return ['chinese', 'mandarin', 'cantonese', 'minnan', 'hokkien', 'putonghua', '中文', '普通话', '國語', '国语', '粤', '闽', '華語', '华语']
    .some((token) => key.includes(token))
}

export function getMirrorCopy(language?: string): MirrorCopy {
  return isChineseLanguage(language) ? ZH : EN
}
