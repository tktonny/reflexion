import type { LanguageKey } from './voice'

export type DailyConversationStage =
  | 'warm_up'
  | 'yesterday_recall'
  | 'present_planning'
  | 'medication_reminder'
  | 'reminiscence'
  | 'close'

export type DailyConversationPlan = {
  protocolVersion: 'daily-conversation-v2'
  patientName: string
  includeReminiscence: boolean
  reminiscencePrompt: 'holiday' | 'childhood_food'
  medicationReminder?: {
    occurrenceId: string
    displayText: string
    scheduledAt: string
  }
}

export type DailyConversationTurnMetadata = {
  protocolStage: Exclude<DailyConversationStage, 'close'>
  cognitiveSignals: string[]
}

export const BASE_DAILY_PATIENT_TURNS = 5
export const DEFAULT_REMINISCENCE_WEEKDAYS = [2, 5] as const // Tuesday + Friday
export const SCREENING_TOTAL_QUESTIONS = BASE_DAILY_PATIENT_TURNS

export function createDailyConversationPlan(options: {
  patientName?: string | null
  now?: Date
  medicationReminder?: DailyConversationPlan['medicationReminder']
  reminiscenceWeekdays?: readonly number[]
} = {}): DailyConversationPlan {
  const now = options.now ?? new Date()
  const weekdays = options.reminiscenceWeekdays ?? DEFAULT_REMINISCENCE_WEEKDAYS
  return {
    protocolVersion: 'daily-conversation-v2',
    patientName: cleanName(options.patientName),
    includeReminiscence: weekdays.includes(now.getDay()),
    reminiscencePrompt: now.getDay() === weekdays.at(-1) ? 'childhood_food' : 'holiday',
    ...(options.medicationReminder ? { medicationReminder: options.medicationReminder } : {}),
  }
}

export function dailyConversationPatientTurns(plan: DailyConversationPlan): number {
  return BASE_DAILY_PATIENT_TURNS + Number(Boolean(plan.medicationReminder)) + Number(plan.includeReminiscence)
}

const BASE_QUESTIONS: Partial<Record<LanguageKey, string[]>> = {
  english: [
    'What did you have for dinner yesterday?',
    'And did you sleep well last night?',
    'What are you planning to do today?',
    'Is anyone visiting you this week?',
  ],
  mandarin: ['你昨天晚饭吃了什么？', '那你昨晚睡得好吗？', '你今天打算做些什么？', '这周会有人来看你吗？'],
  cantonese: ['你尋晚食咗啲咩？', '噉你尋晚瞓得好唔好？', '你今日打算做啲咩？', '今個星期會唔會有人嚟探你？'],
  minnan: ['你昨昏暗頓食啥物？', '你昨暗睏了好無？', '你今仔日拍算欲做啥物？', '這禮拜有人欲來看你無？'],
  malay: ['Apa yang anda makan untuk makan malam semalam?', 'Adakah anda tidur lena malam tadi?', 'Apa yang anda rancang untuk lakukan hari ini?', 'Adakah sesiapa akan melawat anda minggu ini?'],
  tamil: ['நேற்று இரவு உணவிற்கு என்ன சாப்பிட்டீர்கள்?', 'நேற்று இரவு நன்றாகத் தூங்கினீர்களா?', 'இன்று என்ன செய்யத் திட்டமிட்டுள்ளீர்கள்?', 'இந்த வாரம் யாராவது உங்களைப் பார்க்க வருகிறார்களா?'],
}

function cleanName(value: string | null | undefined): string {
  return String(value || '').replace(/\s+/g, ' ').replace(/["“”<>]/g, '').trim().slice(0, 80) || 'there'
}

function cleanMedicationName(value: string): string {
  return String(value || '').replace(/\s+/g, ' ').replace(/["“”<>]/g, '').trim().slice(0, 120) || 'your medication'
}

export function openingTextForLanguage(language: LanguageKey, patientName?: string | null): string {
  const name = cleanName(patientName)
  switch (language) {
    case 'mandarin': return `${name}，早上好，很高兴见到你。你今天感觉怎么样？`
    case 'cantonese': return `${name}，早晨，好開心見到你。你今日感覺點呀？`
    case 'minnan': return `${name}，早安，真歡喜看著你。你今仔日感覺按怎？`
    case 'malay': return `Selamat pagi ${name}, gembira berjumpa dengan anda. Bagaimana perasaan anda hari ini?`
    case 'tamil': return `காலை வணக்கம் ${name}, உங்களைப் பார்ப்பதில் மகிழ்ச்சி. இன்று எப்படி உணர்கிறீர்கள்?`
    default: return `Good morning ${name}, it's lovely to see you. How are you feeling today?`
  }
}

function medicationQuestion(language: LanguageKey, plan: DailyConversationPlan): string | null {
  if (!plan.medicationReminder) return null
  const medication = cleanMedicationName(plan.medicationReminder.displayText)
  const name = cleanName(plan.patientName)
  switch (language) {
    case 'mandarin': return `${name}，提醒一下照护者为你安排的${medication}。你已经服用了吗？`
    case 'cantonese': return `${name}，提提你照顧者安排咗嘅${medication}。你食咗未呀？`
    case 'minnan': return `${name}，提醒你照顧者安排的${medication}。你食過矣無？`
    case 'malay': return `${name}, peringatan ringkas tentang ${medication} yang telah dijadualkan oleh penjaga anda. Sudahkah anda mengambilnya?`
    case 'tamil': return `${name}, உங்கள் பராமரிப்பாளர் திட்டமிட்ட ${medication} பற்றிய ஒரு நினைவூட்டல். அதை எடுத்துக்கொண்டீர்களா?`
    default: return `Good afternoon ${name} — a quick reminder about ${medication}, which your caregiver has scheduled. Have you taken it yet?`
  }
}

function reminiscenceQuestion(language: LanguageKey, prompt: DailyConversationPlan['reminiscencePrompt']): string {
  const holiday = prompt === 'holiday'
  switch (language) {
    case 'mandarin': return holiday ? '在结束前，我很想听你讲讲一次你特别喜欢的假期。' : '在结束前，跟我说说你小时候最喜欢吃的食物吧。'
    case 'cantonese': return holiday ? '完之前，我好想聽你講下一次你好鍾意嘅旅行或假期。' : '完之前，同我講下你細個最鍾意食嘅嘢呀。'
    case 'minnan': return holiday ? '欲結束進前，我真想欲聽你講一擺你真佮意的旅行抑是假期。' : '欲結束進前，講一下你細漢上愛食的物件。'
    case 'malay': return holiday ? 'Sebelum kita selesai, ceritakan tentang percutian yang sangat anda sukai.' : 'Sebelum kita selesai, ceritakan makanan kegemaran anda semasa kecil.'
    case 'tamil': return holiday ? 'முடிப்பதற்கு முன், நீங்கள் மிகவும் விரும்பிய ஒரு விடுமுறையைப் பற்றிச் சொல்லுங்கள்.' : 'முடிப்பதற்கு முன், சிறுவயதில் உங்களுக்குப் பிடித்த உணவைப் பற்றிச் சொல்லுங்கள்.'
    default: return holiday ? 'Before we finish, tell me about a holiday you loved.' : 'Before we finish, what was your favourite food as a child?'
  }
}

/** Returns the next question after `completedPatientTurns`, or null when the warm close is due. */
export function screeningQuestionForTurn(
  language: LanguageKey,
  completedPatientTurns: number,
  plan = createDailyConversationPlan({ reminiscenceWeekdays: [] }),
): string | null {
  const questions = BASE_QUESTIONS[language] ?? BASE_QUESTIONS.english!
  if (completedPatientTurns >= 1 && completedPatientTurns <= questions.length) {
    return questions[completedPatientTurns - 1]
  }
  let optionalIndex = completedPatientTurns - BASE_DAILY_PATIENT_TURNS
  if (plan.medicationReminder) {
    if (optionalIndex === 0) return medicationQuestion(language, plan)
    optionalIndex -= 1
  }
  if (plan.includeReminiscence && optionalIndex === 0) return reminiscenceQuestion(language, plan.reminiscencePrompt)
  return null
}

export function dailyConversationMetadataForPatientTurn(
  patientTurn: number,
  plan: DailyConversationPlan,
): DailyConversationTurnMetadata {
  if (patientTurn <= 1) return { protocolStage: 'warm_up', cognitiveSignals: ['mood', 'speech_initiation', 'response_latency'] }
  if (patientTurn <= 3) return { protocolStage: 'yesterday_recall', cognitiveSignals: ['episodic_memory', 'temporal_orientation', 'narrative_coherence'] }
  if (patientTurn <= 5) return { protocolStage: 'present_planning', cognitiveSignals: ['executive_function', 'prospective_memory', 'social_connectedness'] }
  const medicationTurn = plan.medicationReminder ? 6 : -1
  if (patientTurn === medicationTurn) return { protocolStage: 'medication_reminder', cognitiveSignals: ['memory', 'caregiver_adjunct'] }
  return { protocolStage: 'reminiscence', cognitiveSignals: ['semantic_memory', 'language_richness', 'lexical_diversity', 'speech_fluency'] }
}

export function closingTextForLanguage(language: LanguageKey): string {
  switch (language) {
    case 'mandarin': return '听起来真不错，非常感谢你今天和我聊天。祝你今天过得愉快，再见。'
    case 'cantonese': return '聽落真係幾好，多謝你今日同我傾偈。祝你今日愉快，拜拜。'
    case 'minnan': return '聽起來真好，多謝你今仔日佮我開講。祝你今仔日歡喜，再會。'
    case 'malay': return 'Kedengarannya indah. Terima kasih banyak kerana berbual dengan saya hari ini. Semoga hari anda menyenangkan. Selamat tinggal.'
    case 'tamil': return 'அது அருமையாக இருக்கிறது. இன்று என்னுடன் பேசியதற்கு மிக்க நன்றி. உங்கள் நாள் இனிதாக அமையட்டும். பிரியாவிடை.'
    default: return 'That sounds lovely, thank you so much for chatting with me today. Enjoy your morning! Goodbye.'
  }
}

// A single gentle silence re-prompt (doc: wait 7–10s, then "Take your time", never rush, once only).
export function takeYourTimeForLanguage(language: LanguageKey): string {
  switch (language) {
    case 'mandarin': return '别着急，慢慢来，我一直在这里。'
    case 'cantonese': return '唔使急，慢慢嚟，我一直喺度。'
    case 'minnan': return '免急，沓沓仔來就好，我攏佇遮。'
    case 'malay': return 'Ambil masa anda. Saya di sini bila-bila anda sedia.'
    case 'tamil': return 'நிதானமாக இருங்கள். நீங்கள் தயாராகும்போது நான் இங்கே இருக்கிறேன்.'
    default: return "Take your time. I'm right here whenever you're ready."
  }
}

export function companionClosingTextForLanguage(language: LanguageKey): string {
  switch (language) {
    case 'mandarin': return '不客气，和你聊天很开心。下次再聊，再见。'
    case 'cantonese': return '唔使客氣，同你傾偈好開心。下次再傾，拜拜。'
    case 'minnan': return '免客氣，佮你開講真歡喜。下擺閣講，再會。'
    case 'malay': return 'Sama-sama. Seronok berbual dengan anda. Jumpa lagi.'
    case 'tamil': return 'பரவாயில்லை. உங்களுடன் பேசியது மகிழ்ச்சி. மீண்டும் சந்திப்போம்.'
    default: return "You're welcome. It was nice talking with you. Goodbye."
  }
}

function bytesToBase64(bytes: Uint8Array): string {
  const encoder = (globalThis as unknown as { btoa?: (value: string) => string }).btoa
  if (!encoder) throw new Error('The JavaScript runtime does not provide base64 audio encoding.')
  let binary = ''
  for (let index = 0; index < bytes.length; index += 1) binary += String.fromCharCode(bytes[index])
  return encoder(binary)
}

export function base64ToBytes(base64: string): Uint8Array {
  const decoder = (globalThis as unknown as { atob?: (value: string) => string }).atob
  if (!decoder) throw new Error('The JavaScript runtime does not provide base64 audio decoding.')
  const binary = decoder(base64)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index) & 0xff
  return bytes
}

function uint16(bytes: Uint8Array, offset: number): number {
  return bytes[offset] | (bytes[offset + 1] << 8)
}

function uint32(bytes: Uint8Array, offset: number): number {
  return (bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16) | (bytes[offset + 3] << 24)) >>> 0
}

function ascii(bytes: Uint8Array, offset: number, length: number): string {
  let value = ''
  for (let index = 0; index < length; index += 1) value += String.fromCharCode(bytes[offset + index])
  return value
}

/** Extract Qwen TTS WAV payload as native playback chunks (PCM16 mono @ 24 kHz). */
export function qwenWavToPcm24kChunks(wav: Uint8Array, chunkBytes = 48_000): string[] {
  if (wav.length < 44 || ascii(wav, 0, 4) !== 'RIFF' || ascii(wav, 8, 4) !== 'WAVE') {
    throw new Error('Qwen TTS returned an invalid WAV file.')
  }
  let format = 0
  let channels = 0
  let sampleRate = 0
  let bits = 0
  let dataOffset = -1
  let dataLength = 0
  let offset = 12
  while (offset + 8 <= wav.length) {
    const id = ascii(wav, offset, 4)
    const size = uint32(wav, offset + 4)
    const payload = offset + 8
    if (id === 'fmt ' && size >= 16 && payload + 16 <= wav.length) {
      format = uint16(wav, payload)
      channels = uint16(wav, payload + 2)
      sampleRate = uint32(wav, payload + 4)
      bits = uint16(wav, payload + 14)
    } else if (id === 'data') {
      dataOffset = payload
      dataLength = Math.min(size, wav.length - payload)
      break
    }
    offset = payload + size + (size & 1)
  }
  if (format !== 1 || channels !== 1 || sampleRate !== 24_000 || bits !== 16 || dataOffset < 0) {
    throw new Error(`Unsupported Qwen TTS WAV format (${format}/${channels}/${sampleRate}/${bits}).`)
  }
  const chunks: string[] = []
  const safeChunkBytes = Math.max(2, chunkBytes - (chunkBytes % 2))
  for (let cursor = 0; cursor < dataLength; cursor += safeChunkBytes) {
    const end = Math.min(cursor + safeChunkBytes, dataLength)
    chunks.push(bytesToBase64(wav.subarray(dataOffset + cursor, dataOffset + end)))
  }
  return chunks
}
