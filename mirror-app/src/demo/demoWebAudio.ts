import { Platform } from 'react-native'

export type DemoSpeechRecognitionResult = {
  isFinal: boolean
  length: number
  [index: number]: { transcript: string }
}

export type DemoSpeechRecognitionEvent = {
  resultIndex: number
  results: { length: number; [index: number]: DemoSpeechRecognitionResult }
}

export type DemoSpeechRecognition = {
  continuous: boolean
  interimResults: boolean
  lang: string
  onend: (() => void) | null
  onerror: ((event: { error?: string }) => void) | null
  onresult: ((event: DemoSpeechRecognitionEvent) => void) | null
  start: () => void
  stop: () => void
}

export type DemoSpeechRecognitionConstructor = new () => DemoSpeechRecognition

export type DemoVoiceOption = {
  id: string
  name: string
  lang: string
  default: boolean
  localService: boolean
}

type DemoBrowserWindow = Window & {
  SpeechRecognition?: DemoSpeechRecognitionConstructor
  webkitSpeechRecognition?: DemoSpeechRecognitionConstructor
  webkitAudioContext?: typeof AudioContext
}

export function isWebDemoRuntime() {
  return Platform.OS === 'web' && typeof window !== 'undefined'
}

export function getDemoSpeechRecognitionConstructor(): DemoSpeechRecognitionConstructor | null {
  if (!isWebDemoRuntime()) return null
  const browserWindow = window as DemoBrowserWindow
  return browserWindow.SpeechRecognition ?? browserWindow.webkitSpeechRecognition ?? null
}

export function getDemoWebAudioDiagnostics() {
  if (!isWebDemoRuntime()) {
    return {
      secureContext: false,
      microphoneSupported: false,
      speechSynthesisSupported: false,
      speechRecognitionSupported: false,
    }
  }
  const browserWindow = window as DemoBrowserWindow
  return {
    secureContext: Boolean(window.isSecureContext),
    microphoneSupported: Boolean(navigator.mediaDevices?.getUserMedia),
    speechSynthesisSupported: Boolean(browserWindow.speechSynthesis && typeof SpeechSynthesisUtterance !== 'undefined'),
    speechRecognitionSupported: Boolean(getDemoSpeechRecognitionConstructor()),
  }
}

export function getDemoSpeechVoices(): SpeechSynthesisVoice[] {
  if (!isWebDemoRuntime() || !window.speechSynthesis) return []
  return window.speechSynthesis.getVoices()
}

export function demoVoiceId(voice: SpeechSynthesisVoice): string {
  return voice.voiceURI || `${voice.name}|${voice.lang}`
}

export function getDemoVoiceOptions(): DemoVoiceOption[] {
  return getDemoSpeechVoices().map((voice) => ({
    id: demoVoiceId(voice),
    name: voice.name,
    lang: voice.lang,
    default: voice.default,
    localService: voice.localService,
  }))
}

export function chooseDemoVoice(locale = 'en-US', preferredVoiceId?: string): SpeechSynthesisVoice | undefined {
  if (!isWebDemoRuntime() || !window.speechSynthesis) return undefined
  const voices = getDemoSpeechVoices()
  if (!voices.length) return undefined
  const language = locale.toLowerCase()
  const languagePrefix = language.split('-')[0]
  const matchingVoices = voices.filter((voice) => voice.lang.toLowerCase() === language || voice.lang.toLowerCase().startsWith(languagePrefix))
  const naturalEnglish = languagePrefix === 'en'
    ? matchingVoices.find((voice) => /female|woman|natural|enhanced|premium/i.test(voice.name)) || matchingVoices.find((voice) => voice.localService)
    : undefined
  return voices.find((voice) => preferredVoiceId && demoVoiceId(voice) === preferredVoiceId)
    || (languagePrefix === 'en' ? naturalEnglish : undefined)
    || matchingVoices[0]
    || voices.find((voice) => voice.default)
    || voices[0]
}

export function safeDemoAudioError(error: unknown, fallback: string) {
  if (error instanceof DOMException && error.name) return error.name
  if (error instanceof Error && error.message) return error.message.slice(0, 120)
  return fallback
}
