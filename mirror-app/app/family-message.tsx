import { AudioModule, createAudioPlayer, RecordingPresets, setAudioModeAsync, useAudioRecorder } from 'expo-audio'
import { CryptoDigestAlgorithm, digest } from 'expo-crypto'
import * as FileSystem from 'expo-file-system/legacy'
import { router, useLocalSearchParams } from 'expo-router'
import { useCallback, useEffect, useRef, useState } from 'react'
import { Image, Platform, Pressable, StyleSheet, Text, View } from 'react-native'

import {
  commitFamilyVoiceReply,
  getDeviceFamilyMessages,
  prepareFamilyVoiceReplyUpload,
  recordFamilyMessageInteraction,
  type MirrorFamilyMessage,
  uploadFamilyVoiceReply,
} from '../src/api/familyMessages'
import { playAndWaitForCompletion, type PlaybackCompletionPlayer } from '../src/orchestration/playbackCompletion'
import { getDeviceCredential } from '../src/storage/deviceCredentials'
import { MirrorCard, MirrorPage, OutlineButton, PrimaryButton } from '../src/components/mirror/MirrorChrome'
import { MirrorIcon } from '../src/components/mirror/MirrorIcon'
import { mirrorColors as c, mirrorFonts as f } from '../src/theme/mirrorTheme'
import { isDemoRoute } from '../src/demo/demoConfig'
import { getDemoMessage, getDemoMirrorState, markDemoVoiceReplySent, updateDemoMirrorState } from '../src/demo/demoRepository'

type MessageKind = 'text' | 'photo' | 'voice'
type VoiceReplyState = 'idle' | 'recording' | 'review' | 'sending' | 'sent' | 'failed'

export default function FamilyMessageScreen() {
  const params = useLocalSearchParams<{ demo?: string; messageId?: string; kind?: string; sender?: string; body?: string; caption?: string; mediaUri?: string }>()
  const demo = isDemoRoute(params.demo)
  const [message, setMessage] = useState<MirrorFamilyMessage | null>(null)
  const [interaction, setInteraction] = useState<'Viewed' | 'Played' | 'Replayed'>('Viewed')
  const [playing, setPlaying] = useState(false)
  const [voiceReplyState, setVoiceReplyState] = useState<VoiceReplyState>('idle')
  const [recordingUri, setRecordingUri] = useState<string | null>(null)
  const [recordingDurationMs, setRecordingDurationMs] = useState(0)
  const [voiceReplyError, setVoiceReplyError] = useState('')
  const playerRef = useRef<ReturnType<typeof createAudioPlayer> | null>(null)
  const replyPlayerRef = useRef<ReturnType<typeof createAudioPlayer> | null>(null)
  const recordingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const clientReplyIdRef = useRef<string | null>(null)
  const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY)
  const kind: MessageKind = message?.kind || (params.kind === 'photo' || params.kind === 'voice' ? params.kind : 'text')
  const sender = message?.senderName || 'your family'
  const body = String(message?.text || message?.caption || '').trim()

  useEffect(() => {
    let mounted = true
    async function load() {
      if (!params.messageId) return
      if (demo) {
        const fixture = getDemoMessage(String(params.messageId))
        const demoState = await getDemoMirrorState()
        if (mounted) setMessage({
          messageId: fixture.messageId,
          kind: fixture.kind,
          senderName: fixture.senderName,
          text: fixture.text || null,
          caption: fixture.caption || null,
          mediaUrl: null,
          deliveryState: 'delivered',
          interactionState: null,
          createdAt: new Date().toISOString(),
        })
        if (mounted && demoState.voiceReply.state === 'sent') setVoiceReplyState('sent')
        return
      }
      const credential = await getDeviceCredential()
      if (!credential) return
      const messages = await getDeviceFamilyMessages(credential.deviceId).catch(() => [])
      const next = messages.find((candidate) => candidate.messageId === String(params.messageId))
      if (mounted && next) setMessage(next)
    }
    void load()
    return () => { mounted = false }
  }, [demo, params.messageId])

  useEffect(() => {
    if (!message) return
    void sendInteraction('viewed')
    // Viewed is distinct from dismissing; dismissal never clears the server-side interaction state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [message?.messageId])

  useEffect(() => () => {
    try { playerRef.current?.remove() } catch { /* best-effort cleanup */ }
    try { replyPlayerRef.current?.remove() } catch { /* best-effort cleanup */ }
    if (recordingTimerRef.current) clearInterval(recordingTimerRef.current)
    playerRef.current = null
    replyPlayerRef.current = null
  }, [])

  async function sendInteraction(kindValue: 'viewed' | 'played' | 'replayed') {
    setInteraction(kindValue === 'viewed' ? 'Viewed' : kindValue === 'played' ? 'Played' : 'Replayed')
    if (demo) {
      if (message?.messageId) {
        const current = await getDemoMirrorStateForInteraction()
        await updateDemoMirrorState({ messageInteractions: { ...current, [message.messageId]: kindValue === 'viewed' ? 'Viewed' : kindValue === 'played' ? 'Played' : 'Replayed' } })
      }
      return
    }
    const credential = await getDeviceCredential()
    if (credential && message?.messageId && message.messageId !== 'local') {
      await recordFamilyMessageInteraction(credential.deviceId, message.messageId, kindValue).catch(() => undefined)
    }
  }

  const playVoice = useCallback(async (replay: boolean) => {
    if (demo) {
      setPlaying(true)
      setTimeout(() => setPlaying(false), 1400)
      void sendInteraction(replay ? 'replayed' : 'played')
      return
    }
    const source = message?.mediaUrl
    if (!source) return
    try {
      playerRef.current?.remove()
      playerRef.current = null
      await setAudioModeAsync({ playsInSilentMode: true, allowsRecording: false })
      const player = createAudioPlayer(source, { updateInterval: 100 })
      playerRef.current = player
      setPlaying(true)
      await playAndWaitForCompletion(player as unknown as PlaybackCompletionPlayer, 120_000)
      void sendInteraction(replay ? 'replayed' : 'played')
    } catch {
      // The caregiver-facing state remains safe; a failed player must not be reported as Played.
      setPlaying(false)
    } finally {
      try { playerRef.current?.remove() } catch { /* best-effort cleanup */ }
      playerRef.current = null
      setPlaying(false)
    }
  }, [demo, message?.mediaUrl])

  const stopRecordingTimer = useCallback(() => {
    if (recordingTimerRef.current) clearInterval(recordingTimerRef.current)
    recordingTimerRef.current = null
  }, [])

  const startVoiceReplyRecording = useCallback(async () => {
    setVoiceReplyError('')
    try {
      const permission = await AudioModule.requestRecordingPermissionsAsync()
      if (!permission.granted) throw new Error('Microphone permission was denied.')
      await setAudioModeAsync({ playsInSilentMode: true, allowsRecording: true })
      await recorder.prepareToRecordAsync()
      recorder.record()
      clientReplyIdRef.current = `mirror_voice_reply_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`
      setRecordingDurationMs(0)
      setVoiceReplyState('recording')
      stopRecordingTimer()
      recordingTimerRef.current = setInterval(() => setRecordingDurationMs(Math.max(0, Math.round(recorder.currentTime * 1000))), 250)
    } catch (error) {
      setVoiceReplyState('failed')
      setVoiceReplyError(error instanceof Error ? error.message : 'The microphone could not start. Please try again.')
    }
  }, [recorder, stopRecordingTimer])

  const stopVoiceReplyRecording = useCallback(async () => {
    if (voiceReplyState !== 'recording') return
    stopRecordingTimer()
    try {
      await recorder.stop()
      const uri = recorder.uri
      if (!uri) throw new Error('The recording was not saved. Please try again.')
      setRecordingUri(uri)
      setRecordingDurationMs(Math.max(1, Math.round(recorder.currentTime * 1000)))
      setVoiceReplyState('review')
    } catch (error) {
      setVoiceReplyState('failed')
      setVoiceReplyError(error instanceof Error ? error.message : 'The recording could not be saved. Please try again.')
    }
  }, [recorder, stopRecordingTimer, voiceReplyState])

  const playVoiceReply = useCallback(async () => {
    if (!recordingUri) return
    try {
      replyPlayerRef.current?.remove()
      await setAudioModeAsync({ playsInSilentMode: true, allowsRecording: false })
      const player = createAudioPlayer(recordingUri, { updateInterval: 100 })
      replyPlayerRef.current = player
      setPlaying(true)
      await playAndWaitForCompletion(player as unknown as PlaybackCompletionPlayer, 120_000)
    } catch (error) {
      setVoiceReplyError(error instanceof Error ? error.message : 'The recording could not be played.')
    } finally {
      try { replyPlayerRef.current?.remove() } catch { /* best effort */ }
      replyPlayerRef.current = null
      setPlaying(false)
    }
  }, [recordingUri])

  const reRecordVoiceReply = useCallback(() => {
    try { replyPlayerRef.current?.remove() } catch { /* best effort */ }
    replyPlayerRef.current = null
    setRecordingUri(null)
    setRecordingDurationMs(0)
    setVoiceReplyError('')
    clientReplyIdRef.current = null
    setVoiceReplyState('idle')
  }, [])

  const sendVoiceReply = useCallback(async () => {
    if (!recordingUri || !message?.messageId || !clientReplyIdRef.current) return
    setVoiceReplyState('sending')
    setVoiceReplyError('')
    try {
      if (demo) {
        await markDemoVoiceReplySent()
        setVoiceReplyState('sent')
        return
      }
      const credential = await getDeviceCredential()
      if (!credential) throw new Error('The mirror is not paired.')
      const bytes = await readRecordingBytes(recordingUri)
      const hash = bytesToHex(await digest(CryptoDigestAlgorithm.SHA256, bytes.buffer as ArrayBuffer))
      const contentType = Platform.OS === 'web' ? 'audio/webm' : 'audio/m4a'
      const plan = await prepareFamilyVoiceReplyUpload({
        deviceId: credential.deviceId,
        messageId: message.messageId,
        clientReplyId: clientReplyIdRef.current,
        contentType,
        hash,
        sizeBytes: bytes.byteLength,
        durationMs: Math.max(1, recordingDurationMs),
      })
      if (plan.state !== 'sent') {
        await uploadFamilyVoiceReply(plan, bytes)
        await commitFamilyVoiceReply({
          deviceId: credential.deviceId,
          messageId: message.messageId,
          replyId: plan.replyId,
          hash,
          sizeBytes: bytes.byteLength,
          clientReplyId: clientReplyIdRef.current,
        })
      }
      setVoiceReplyState('sent')
    } catch (error) {
      // Keep recordingUri and clientReplyIdRef so the same local recording can be sent again.
      setVoiceReplyState('failed')
      setVoiceReplyError(error instanceof Error ? error.message : 'The reply could not be sent. Please try again.')
    }
  }, [demo, message?.messageId, recordingDurationMs, recordingUri])

  function replay() {
    if (kind === 'voice') {
      void playVoice(true)
      return
    }
    void sendInteraction('replayed')
  }

  return (
    <MirrorPage headerStatus="Mirror ready" onHelp={() => router.push(demo ? '/status?demo=1' : '/status')}>
      <View style={styles.content}>
        <Text style={styles.eyebrow}>FAMILY MESSAGE</Text>
        <Text style={styles.title}>A message for you</Text>
        {message ? <MirrorCard accent>
          <View style={styles.senderRow}>
            <View style={styles.senderAvatar}><Text style={styles.senderInitial}>{sender.charAt(0).toUpperCase()}</Text></View>
            <View style={styles.senderCopy}><Text style={styles.from}>From {sender}</Text><Text style={styles.status}>{interaction}</Text></View>
          </View>
          {kind === 'photo' ? <Image resizeMode="cover" source={demo ? require('../assets/images/aria-avatar.png') : { uri: message.mediaUrl || undefined }} style={styles.photo} /> : null}
          {kind === 'voice' ? (
            <View style={styles.voicePlayer}>
              <MirrorIcon name={playing ? 'volume-high-outline' : 'play'} size={42} color={c.sageDeep} />
              <View style={styles.voiceWave}><View style={styles.voiceTrack} /><Text style={styles.voiceLabel}>{playing ? 'Playing message…' : 'Select Play to hear this message'}</Text></View>
            </View>
          ) : <Text style={styles.messageText}>{body || 'This family message has no text content.'}</Text>}
        </MirrorCard> : <MirrorCard>
          <MirrorIcon name="mail-open-outline" size={46} color={c.sageDeep} />
          <Text style={styles.emptyTitle}>This message is no longer available.</Text>
          <Text style={styles.emptyBody}>Ask your caregiver to send it again if you still need it.</Text>
        </MirrorCard>}
        {message && kind === 'voice' ? (
          <PrimaryButton label={playing ? 'Replay message' : 'Play message'} icon={playing ? 'refresh-outline' : 'play'} onPress={() => { if (playing) replay(); else void playVoice(false) }} />
        ) : message ? <PrimaryButton label={kind === 'photo' ? 'Replay caption' : 'Replay message'} icon="refresh-outline" onPress={replay} /> : null}
        {message && voiceReplyState === 'idle' ? <OutlineButton label="Reply by voice" icon="mic-outline" onPress={() => void startVoiceReplyRecording()} /> : null}
        {voiceReplyState === 'recording' ? <MirrorCard accent>
          <View style={styles.replyStatus}><MirrorIcon name="mic" size={42} color={c.coral} /><Text style={styles.replyTitle}>Recording your reply</Text><Text style={styles.replyTimer}>{formatDuration(recordingDurationMs)}</Text><Text style={styles.replyHint}>Speak naturally. Tap Stop recording when you are finished.</Text></View>
          <PrimaryButton label="Stop recording" icon="stop-circle-outline" onPress={() => void stopVoiceReplyRecording()} />
        </MirrorCard> : null}
        {voiceReplyState === 'review' || voiceReplyState === 'failed' ? <MirrorCard accent>
          <Text style={styles.replyTitle}>{voiceReplyState === 'failed' ? 'Your reply is ready to retry' : 'Review your reply'}</Text>
          <Text style={styles.replyHint}>{voiceReplyState === 'failed' ? voiceReplyError : `Recorded ${formatDuration(recordingDurationMs)}. Nothing has been sent yet.`}</Text>
          {voiceReplyState === 'failed' ? <Text style={styles.replyError}>{voiceReplyError}</Text> : null}
          <PrimaryButton label={playing ? 'Playing recording…' : 'Play recording'} icon="play" onPress={playing ? undefined : () => void playVoiceReply()} />
          <OutlineButton label="Re-record" icon="refresh-outline" onPress={reRecordVoiceReply} />
          <PrimaryButton label="Send reply" icon="send" onPress={() => void sendVoiceReply()} />
        </MirrorCard> : null}
        {voiceReplyState === 'sending' ? <MirrorCard accent><View style={styles.replyStatus}><MirrorIcon name="time-outline" size={38} color={c.goldDeep} /><Text style={styles.replyTitle}>Sending your reply…</Text><Text style={styles.replyHint}>Please keep this message open.</Text></View></MirrorCard> : null}
        {voiceReplyState === 'sent' ? <MirrorCard accent><View style={styles.replyStatus}><MirrorIcon name="checkmark-circle" size={46} color={c.sageDeep} /><Text style={styles.replyTitle}>Voice reply sent</Text><Text style={styles.replyHint}>Your caregiver will see it with this family message.</Text></View></MirrorCard> : null}
        {voiceReplyState !== 'recording' && voiceReplyState !== 'sending' ? <OutlineButton label="Dismiss" icon="close-outline" onPress={() => router.replace(demo ? '/demo' : '/conversation')} /> : null}
        <Text style={styles.note}>{demo ? 'Demo only — the recording stays local and is not sent to production.' : 'Your recording is uploaded only after you tap Send reply.'}</Text>
      </View>
    </MirrorPage>
  )
}

const styles = StyleSheet.create({
  content: { alignItems: 'center', maxWidth: 760, width: '100%' },
  eyebrow: { color: c.goldDeep, fontFamily: f.bodyMedium, fontSize: 13, letterSpacing: 2, marginBottom: 7 },
  title: { color: c.text, fontFamily: f.display, fontSize: 48, lineHeight: 58, textAlign: 'center' },
  senderRow: { alignItems: 'center', flexDirection: 'row', gap: 15 },
  senderAvatar: { alignItems: 'center', backgroundColor: c.sage, borderRadius: 37, height: 74, justifyContent: 'center', width: 74 },
  senderInitial: { color: c.sageDeep, fontFamily: f.display, fontSize: 34 },
  senderCopy: { flex: 1 },
  from: { color: c.text, fontFamily: f.display, fontSize: 28 },
  status: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 14, marginTop: 4 },
  photo: { borderRadius: 18, height: 270, marginTop: 22, width: '100%' },
  messageText: { color: c.text, fontFamily: f.display, fontSize: 30, lineHeight: 42, marginTop: 25 },
  voicePlayer: { alignItems: 'center', borderTopColor: c.lineWarm, borderTopWidth: StyleSheet.hairlineWidth, flexDirection: 'row', gap: 16, marginTop: 24, paddingTop: 22 },
  voiceWave: { flex: 1 },
  voiceTrack: { backgroundColor: c.sage, borderRadius: 4, height: 34, marginBottom: 7, width: '100%' },
  voiceLabel: { color: c.textSecondary, fontFamily: f.body, fontSize: 15 },
  replyStatus: { alignItems: 'center' },
  replyTitle: { color: c.text, fontFamily: f.display, fontSize: 29, marginTop: 8, textAlign: 'center' },
  replyTimer: { color: c.coral, fontFamily: f.display, fontSize: 40, marginTop: 5 },
  replyHint: { color: c.textSecondary, fontFamily: f.body, fontSize: 16, lineHeight: 24, marginTop: 8, textAlign: 'center' },
  replyError: { color: c.coral, fontFamily: f.bodyMedium, fontSize: 14, marginTop: 8, textAlign: 'center' },
  emptyTitle: { color: c.text, fontFamily: f.display, fontSize: 27, marginTop: 18, textAlign: 'center' },
  emptyBody: { color: c.textSecondary, fontFamily: f.body, fontSize: 16, lineHeight: 23, marginTop: 8, textAlign: 'center' },
  note: { color: c.textSecondary, fontFamily: f.body, fontSize: 13, lineHeight: 19, marginTop: 17, textAlign: 'center' },
})

async function getDemoMirrorStateForInteraction() {
  const { getDemoMirrorState } = await import('../src/demo/demoRepository')
  return (await getDemoMirrorState()).messageInteractions
}

async function readRecordingBytes(uri: string): Promise<Uint8Array> {
  if (Platform.OS === 'web') return new Uint8Array(await (await fetch(uri)).arrayBuffer())
  const base64 = await FileSystem.readAsStringAsync(uri, { encoding: 'base64' as FileSystem.EncodingType })
  const binary = globalThis.atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index)
  return bytes
}

function bytesToHex(value: ArrayBuffer): string {
  return Array.from(new Uint8Array(value), (byte) => byte.toString(16).padStart(2, '0')).join('')
}

function formatDuration(milliseconds: number): string {
  const seconds = Math.max(0, Math.floor(milliseconds / 1000))
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`
}
