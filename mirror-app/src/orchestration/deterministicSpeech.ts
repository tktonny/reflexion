import type { LanguageKey } from './voice'

export const SCREENING_TOTAL_QUESTIONS = 7

const SCREENING_QUESTIONS: Partial<Record<LanguageKey, string[]>> = {
  english: [
    'Thank you. Let\'s start with where you are. Are you at home, at a clinic, or somewhere else?',
    'That helps. Now think about this moment. Is it morning, afternoon, or evening, and about what day is it?',
    'Thank you. Think back from when you woke up until now. What is one thing you have done today?',
    'Take your time. What happened just before or just after that?',
    'Thanks, that helps. Let\'s talk about everyday routines. How do you keep track of meals, medicines, or appointments—for example, with a pillbox, calendar, reminders, or help from someone?',
  ],
  mandarin: [
    '谢谢你。我们先从你现在的位置聊起。你是在家里、诊所，还是其他地方？',
    '好的。现在想一想此刻。现在是早上、下午还是晚上？大约星期几？',
    '谢谢。请从今天起床后慢慢想一想。你今天做过的一件事是什么？',
    '不用着急。那件事之前或之后发生了什么？',
    '谢谢，这很有帮助。我们再聊聊日常安排。你怎么记住吃饭、吃药或预约，比如药盒、日历、提醒，或者家人的帮助？',
  ],
  cantonese: [
    '多謝你。我哋先由你而家喺邊度開始。你喺屋企、診所，定係其他地方？',
    '好啊。諗一諗而家呢一刻。係朝早、下晝定夜晚？大概星期幾？',
    '多謝。由今日起身開始慢慢諗。你今日做過一件咩事？',
    '唔使急。嗰件事之前或者之後發生咗咩？',
    '多謝，呢個好有幫助。我哋再傾下日常安排。你點樣記住食飯、食藥或者約會，例如藥盒、日曆、提示，或者屋企人幫手？',
  ],
  minnan: [
    '多謝你。咱先對你這馬佇佗位開始。你佇厝裡、診所，抑是別位？',
    '好。想一下這馬這个時陣。是早起、下晡抑是暗時？大約拜幾？',
    '多謝。對今仔日起床了後慢慢想。你今仔日有做一項啥物代誌？',
    '免緊張，慢慢來。彼件代誌進前抑是了後發生啥物？',
    '多謝，這真有幫助。咱閣講一下平常的安排。你按怎記得食飯、食藥抑是約會，親像藥盒、日曆、提醒，抑是厝裡人的幫忙？',
  ],
  malay: [
    'Terima kasih. Mari mulakan dengan tempat anda berada. Adakah anda di rumah, di klinik, atau di tempat lain?',
    'Baik. Fikirkan saat ini. Adakah sekarang pagi, petang, atau malam, dan lebih kurang hari apa?',
    'Terima kasih. Fikir perlahan-lahan dari waktu anda bangun. Apakah satu perkara yang anda lakukan hari ini?',
    'Luangkan masa anda. Apa yang berlaku sejurus sebelum atau selepas itu?',
    'Terima kasih, itu membantu. Mari bercakap tentang rutin harian. Bagaimana anda mengingati waktu makan, ubat, atau janji temu—contohnya dengan kotak ubat, kalendar, peringatan, atau bantuan seseorang?',
  ],
  tamil: [
    'நன்றி. நீங்கள் இருக்கும் இடத்திலிருந்து தொடங்கலாம். நீங்கள் வீட்டிலா, மருத்துவமனையிலா, அல்லது வேறு எங்காவதா இருக்கிறீர்கள்?',
    'சரி. இந்த நேரத்தைப் பற்றி யோசியுங்கள். இப்போது காலை, மதியம், அல்லது மாலையா, மேலும் தோராயமாக என்ன நாள்?',
    'நன்றி. இன்று எழுந்ததிலிருந்து மெதுவாக நினைத்துப் பாருங்கள். இன்று நீங்கள் செய்த ஒரு விஷயம் என்ன?',
    'அவசரப்பட வேண்டாம். அதற்கு சற்று முன் அல்லது பின் என்ன நடந்தது?',
    'நன்றி, அது உதவுகிறது. அன்றாட நடைமுறைகளைப் பற்றிப் பேசலாம். உணவு, மருந்து அல்லது சந்திப்புகளை மருந்துப் பெட்டி, நாட்காட்டி, நினைவூட்டல், அல்லது ஒருவரின் உதவியுடன் எப்படி நினைவில் வைத்துக்கொள்கிறீர்கள்?',
  ],
}

/** Question following accepted patient turns 1–5; turn 6 is followed by delayed recall. */
export function screeningQuestionForTurn(
  language: LanguageKey,
  completedPatientTurns: number,
): string | null {
  const questions = SCREENING_QUESTIONS[language] ?? SCREENING_QUESTIONS.english!
  return questions[completedPatientTurns - 1] ?? null
}

function cleanDetail(detail: string): string {
  return detail.replace(/\s+/g, ' ').replace(/["“”]/g, '').trim().replace(/[.!?。！？]+$/, '').slice(0, 160)
}

export function recallQuestionForLanguage(language: LanguageKey, detail: string): string {
  const memory = cleanDetail(detail) || 'what you were doing earlier'
  switch (language) {
    case 'mandarin': return `我们快聊完了。你刚才提到“${memory}”。不用着急，可以再跟我说说这件事吗？`
    case 'cantonese': return `我哋差唔多傾完喇。你頭先提到「${memory}」。唔使急，可以再同我講下呢件事嗎？`
    case 'minnan': return `咱欲講煞矣。你頭前有講著「${memory}」。免緊張，會使閣共我講一下這件代誌無？`
    case 'malay': return `Kita hampir selesai. Tadi anda menyebut “${memory}”. Luangkan masa anda dan ceritakan tentangnya sekali lagi.`
    case 'tamil': return `நாம் கிட்டத்தட்ட முடித்துவிட்டோம். முன்பு நீங்கள் “${memory}” என்று சொன்னீர்கள். அவசரப்படாமல் அதைப் பற்றி மீண்டும் சொல்ல முடியுமா?`
    default: return `We are nearly done. Earlier you mentioned “${memory}”. Take your time—could you tell me about that once more?`
  }
}

export function closingTextForLanguage(language: LanguageKey): string {
  switch (language) {
    case 'mandarin': return '好的，我们今天先聊到这里。谢谢你今天跟我聊聊。再见。'
    case 'cantonese': return '好啦，我哋今日傾到呢度。多謝你今日同我傾偈。拜拜。'
    case 'minnan': return '好，咱今仔日就講到遮。多謝你今仔日佮我開講。再會。'
    case 'malay': return 'Baiklah, kita berhenti di sini untuk hari ini. Terima kasih kerana berbual dengan saya. Selamat tinggal.'
    case 'tamil': return 'சரி, இன்று இங்கே நிறுத்திக்கொள்வோம். இன்று என்னுடன் பேசியதற்கு நன்றி. பிரியாவிடை.'
    default: return "All right, let's wrap up here for today. Thank you for chatting with me. Goodbye."
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
