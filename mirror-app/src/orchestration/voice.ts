// Voice-profile selection + transcript language detection (client TS port of server/voice.mjs).
// Pure functions, no I/O. Shared by the turn-based (v2) and direct-WS (v3) conversation hooks.

import { QWEN } from '../config/conversationMode'

export type LanguageKey = 'english' | 'mandarin' | 'cantonese' | 'minnan' | 'malay' | 'tamil' | 'custom'
export type VoiceProfile = {
  languageKey: LanguageKey
  languageLabel: string
  voice: string
  source: string
}
export type LanguageSignal = { languageKey: LanguageKey; confidence: number; source: string }

export const LANGUAGE_HINT_ALIASES: Record<string, string[]> = {
  english: ['en', 'en-us', 'en-gb', 'english'],
  mandarin: ['zh', 'zh-cn', 'zh-hans', 'cmn', 'chinese', 'mandarin', 'mandarin chinese', 'putonghua', '普通话', '国语', '中文', '汉语', '漢語'],
  cantonese: ['yue', 'zh-hk', 'cantonese', 'cantonese chinese', 'guangdonghua', '广东话', '廣東話', '粤语', '粵語'],
  minnan: ['nan', 'minnan', 'hokkien', 'taiwanese', 'taiyu', 'min nan', 'minnan chinese', '闽南', '闽南话', '闽南语', '閩南', '閩南話', '閩南語', '台语', '台語', '臺語'],
  malay: ['ms', 'ms-my', 'malay', 'bahasa', 'bahasa melayu', 'melayu'],
  tamil: ['ta', 'ta-in', 'tamil', 'தமிழ்'],
}

const MINNAN_MARKERS = ['按怎', '啥物', '歹势', '歹勢', '有影', '毋知', '欲', '今仔日', '昨昏', '恁', '阮', '咱', '這馬', '这马', '家己', '逐家', '伊', '彼个', '啥人', '無啥', '无啥', '有夠', '真濟', '遐', '食饱未', '食飽未']
const CANTONESE_MARKERS = ['佢', '冇', '唔', '喺', '而家', '依家', '有冇', '咩', '乜', '嘅', '咗', '嚟', '係', '呢', '啦', '喇', '喎', '啱啱', '頭先', '返屋企', '佢哋', '唔係', '咁樣']
const ENGLISH_FUNCTION_WORDS = new Set(['a', 'am', 'and', 'are', 'at', 'did', 'for', 'had', 'have', 'hello', 'home', 'i', "i'm", 'im', 'in', 'is', 'it', 'me', 'my', 'name', 'now', 'right', 'the', 'this', 'today', 'was', 'went', 'where', 'you'])

export function normalizeLanguageKey(languageHint: string | null | undefined): LanguageKey | null {
  if (!languageHint) return null
  const normalized = String(languageHint).replace(/[\s_]+/g, ' ').trim().toLowerCase()
  if (!normalized) return null
  for (const key of Object.keys(LANGUAGE_HINT_ALIASES)) {
    if (LANGUAGE_HINT_ALIASES[key].includes(normalized)) return key as LanguageKey
  }
  return null
}

function countUniqueMarkers(text: string, markers: string[]): number {
  let hits = 0
  for (const m of markers) if (m && text.includes(m)) hits += 1
  return hits
}

export function detectLanguageSignal(transcript: string | null | undefined): LanguageSignal | null {
  const normalized = String(transcript || '').trim().toLowerCase()
  if (!normalized) return null
  const minnanHits = countUniqueMarkers(normalized, MINNAN_MARKERS)
  if (minnanHits >= 1) return { languageKey: 'minnan', confidence: minnanHits >= 2 ? 0.95 : 0.82, source: 'transcript_reassessment' }
  const cantoneseHits = countUniqueMarkers(normalized, CANTONESE_MARKERS)
  if (cantoneseHits >= 1) return { languageKey: 'cantonese', confidence: cantoneseHits >= 2 ? 0.95 : 0.82, source: 'transcript_reassessment' }
  const englishTokens = normalized.match(/[a-z]+(?:'[a-z]+)?/g) || []
  const englishWords = englishTokens.length
  const englishFunctionHits = new Set(englishTokens.filter((t) => ENGLISH_FUNCTION_WORDS.has(t))).size
  const containsCjk = /[一-鿿]/.test(normalized)
  if (englishWords >= 3 && !containsCjk) {
    let confidence: number
    if (englishWords >= 4 && englishFunctionHits >= 2) confidence = 0.84
    else if (englishWords >= 3 && englishFunctionHits >= 2) confidence = 0.79
    else if (englishWords >= 6) confidence = 0.8
    else if (englishFunctionHits >= 1) confidence = 0.74
    else confidence = 0.65
    return { languageKey: 'english', confidence, source: 'transcript_reassessment' }
  }
  if (containsCjk) return { languageKey: 'mandarin', confidence: 0.72, source: 'transcript_reassessment' }
  return null
}

function profile(languageKey: LanguageKey, languageLabel: string, voice: string, source: string): VoiceProfile {
  return { languageKey, languageLabel, voice, source }
}

export function defaultVoiceProfile(source = 'default'): VoiceProfile {
  return profile('mandarin', 'Mandarin Chinese', QWEN.defaultVoice, source)
}

export function voiceProfileForLanguageKey(languageKey: LanguageKey, source: string): VoiceProfile {
  switch (languageKey) {
    case 'english': return profile('english', 'English', QWEN.englishVoice, source)
    case 'mandarin': return profile('mandarin', 'Mandarin Chinese', QWEN.defaultVoice, source)
    case 'minnan': return profile('minnan', 'Minnan Chinese', QWEN.minnanVoice, source)
    case 'cantonese': return profile('cantonese', 'Cantonese', QWEN.cantoneseVoice, source)
    case 'malay': return profile('malay', 'Malay', QWEN.defaultVoice, source)
    case 'tamil': return profile('tamil', 'Tamil', QWEN.defaultVoice, source)
    default: return defaultVoiceProfile(source)
  }
}

export function voiceProfileForSession(languageHint: string | null | undefined, transcript: string | null = null): VoiceProfile {
  const detectedKey = detectLanguageSignal(transcript)?.languageKey ?? null
  if (detectedKey) return voiceProfileForLanguageKey(detectedKey, 'transcript_reassessment')
  const hintedKey = normalizeLanguageKey(languageHint)
  if (hintedKey) return voiceProfileForLanguageKey(hintedKey, 'language_hint')
  const cleanHint = String(languageHint || '').trim()
  if (cleanHint) return profile('custom', cleanHint, QWEN.defaultVoice, 'language_hint')
  return defaultVoiceProfile('language_hint')
}

export function voiceProfileFromRecentSignals(args: {
  languageHint: string | null | undefined
  recentSignals: LanguageSignal[]
  currentProfile: VoiceProfile
}): VoiceProfile | null {
  const { languageHint, recentSignals, currentProfile } = args
  if (!recentSignals.length) return null
  const latest = recentSignals[recentSignals.length - 1]
  if (latest.languageKey === currentProfile.languageKey) return null
  if (['minnan', 'cantonese'].includes(latest.languageKey) && latest.confidence >= 0.8) return voiceProfileForLanguageKey(latest.languageKey, latest.source)
  if (latest.languageKey === 'english' && latest.confidence >= 0.75) return voiceProfileForLanguageKey(latest.languageKey, latest.source)
  if (latest.confidence >= 0.9) return voiceProfileForLanguageKey(latest.languageKey, latest.source)
  if (recentSignals.length >= 2) {
    const lastTwo = recentSignals.slice(-2)
    if (lastTwo.every((s) => s.languageKey === latest.languageKey)) return voiceProfileForLanguageKey(latest.languageKey, latest.source)
  }
  const hintedKey = normalizeLanguageKey(languageHint)
  if (hintedKey === currentProfile.languageKey && latest.confidence >= 0.6) return voiceProfileForLanguageKey(latest.languageKey, latest.source)
  return null
}

export function shouldRestartResponseForLanguageSwitch(args: {
  transcriptTurnIndex: number
  currentProfile: VoiceProfile
  detectedProfile: VoiceProfile
  assistantResponseDoneCount: number
}): boolean {
  const { transcriptTurnIndex, currentProfile, detectedProfile, assistantResponseDoneCount } = args
  if (transcriptTurnIndex !== 1) return false
  if (assistantResponseDoneCount > 0) return false
  return detectedProfile.voice !== currentProfile.voice || detectedProfile.languageLabel !== currentProfile.languageLabel
}

export function languageInputValue(languageKey: LanguageKey, languageLabel: string): string {
  const map: Record<string, string> = { english: 'en', mandarin: 'zh', minnan: 'nan', cantonese: 'yue', malay: 'ms', tamil: 'ta' }
  return map[languageKey] || languageLabel
}
