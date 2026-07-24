import { Ionicons } from '@expo/vector-icons'
import { useEffect, useRef } from 'react'
import {
  ActivityIndicator,
  Animated,
  Easing,
  Image,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native'

import { mirrorColors as c, mirrorFonts as f } from '../../theme/mirrorTheme'
import { getMirrorCopy } from './mirrorStrings'

const WAKE_PHRASE = (process.env.EXPO_PUBLIC_WAKE_WORD_PHRASE || 'Hello Aria').trim()

export type MirrorVisualState =
  | 'ambient'
  | 'connecting'
  | 'listening'
  | 'heard'
  | 'thinking'
  | 'speaking'
  | 'closing'
  | 'saving'
  | 'offline'
  | 'microphone_error'
  | 'service_error'

export type MirrorHomeWidget = {
  icon: keyof typeof Ionicons.glyphMap
  label: string
  value: string
}

type Props = {
  state: MirrorVisualState
  time: string
  date: string
  greeting: string
  patientName: string
  assistantText?: string
  userText?: string
  statusText?: string
  progressText?: string
  /** Patient language (from the caregiver-app setting via device config) — localizes all UI chrome. */
  language?: string
  homeWidgets?: MirrorHomeWidget[]
  wakeListening?: boolean
  wakeError?: string
  bargeInActive?: boolean
  microphoneActive?: boolean
  onEnd?: () => void
  onRetry?: () => void
  /** Short tap on the ambient prompt — starts free chat (companion). */
  onBegin?: () => void
  /** Long-press/hold on the ambient prompt — starts the 6-stage daily check-in (screening). */
  onBeginCheckin?: () => void
}

const ARIA_AVATAR = require('../../../assets/images/aria-avatar.png')

export function MirrorExperience(props: Props) {
  return (
    <View style={styles.shell}>
      <View pointerEvents="none" style={styles.warmWash} />
      <View pointerEvents="none" style={styles.reflectionTop} />
      <View pointerEvents="none" style={styles.reflectionBottom} />
      <View style={styles.stage}>
        {props.state === 'ambient' ? <Ambient {...props} /> : null}
        {props.state === 'connecting' ? <Connecting {...props} /> : null}
        {props.state === 'listening' || props.state === 'heard' ? <Listening {...props} /> : null}
        {props.state === 'thinking' ? <Thinking {...props} /> : null}
        {props.state === 'speaking' ? <Speaking {...props} /> : null}
        {props.state === 'closing' || props.state === 'saving' ? <Closing {...props} /> : null}
        {props.state === 'offline' || props.state === 'microphone_error' || props.state === 'service_error'
          ? <Problem {...props} />
          : null}
      </View>
    </View>
  )
}

function Ambient(props: Props) {
  const t = getMirrorCopy(props.language)
  const widgets = props.homeWidgets?.slice(0, 2) ?? []
  return (
    <View style={styles.ambient}>
      <View style={styles.homeHeader}>
        <Brand lockup />
        {props.wakeListening ? <MicPill label={t.wakePhraseReady} /> : null}
      </View>

      <View style={styles.ambientGreeting}>
        <Text style={styles.greeting}>{props.greeting},</Text>
        <Text adjustsFontSizeToFit numberOfLines={1} style={styles.name}>{props.patientName}</Text>
        <Text style={styles.date}>{props.date}</Text>
        <Text adjustsFontSizeToFit numberOfLines={1} style={styles.time}>{props.time}</Text>
      </View>

      {widgets.length ? (
        <View style={styles.widgetRow}>
          {widgets.map((widget) => (
            <View key={`${widget.label}-${widget.value}`} style={styles.widgetCard}>
              <Ionicons name={widget.icon} size={25} color={c.goldDeep} />
              <View style={styles.widgetCopy}>
                <Text style={styles.widgetValue}>{widget.value}</Text>
                <Text numberOfLines={1} style={styles.widgetLabel}>{widget.label}</Text>
              </View>
            </View>
          ))}
        </View>
      ) : null}

      <Pressable
        accessibilityRole="button"
        accessibilityHint={t.tapToBegin}
        onPress={props.onBegin}
        onLongPress={props.onBeginCheckin}
        delayLongPress={500}
        style={styles.ambientPrompt}
      >
        <ReadyOrb active={Boolean(props.wakeListening)} />
        <Text style={styles.prompt}>{t.sayHello(WAKE_PHRASE)}</Text>
        <Text style={styles.promptSub}>{props.onBegin ? t.tapToBegin : t.toBegin}</Text>
        {props.wakeError ? <Text style={styles.wakeNote}>{props.wakeError}</Text> : null}
      </Pressable>
    </View>
  )
}

function Connecting(props: Props) {
  const t = getMirrorCopy(props.language)
  return (
    <View style={styles.conversationScene}>
      <ConversationHeader label={t.startingChat} />
      <View style={styles.centerContent}>
        <AriaPortrait active mode="thinking" size={210} />
        <Text style={styles.sceneTitle}>{t.hi(props.patientName)}</Text>
        <Text style={styles.sceneTitleCompact}>{props.greeting}.</Text>
        <Text style={styles.sceneCaption}>{props.statusText || t.connectingCaption}</Text>
        <ActivityIndicator color={c.goldDeep} size="small" style={styles.spinner} />
      </View>
    </View>
  )
}

function Listening(props: Props) {
  const t = getMirrorCopy(props.language)
  const heard = props.state === 'heard'
  const hasTranscript = heard && Boolean(props.userText?.trim())
  return (
    <View style={styles.conversationScene}>
      <ConversationHeader label={props.progressText || t.todaysConversation} />
      <View style={styles.centerContent}>
        {props.assistantText ? (
          <Text numberOfLines={4} style={styles.question}>{props.assistantText}</Text>
        ) : null}
        <AriaPortrait active mode="listening" size={heard ? 174 : 194} />
        <MicPill label={t.micActive} />
        <Text style={styles.sceneTitle}>
          {props.bargeInActive ? t.bargeTitle : heard ? t.heardTitle : t.listeningTitle}
        </Text>
        <Text style={styles.sceneCaption}>
          {props.bargeInActive ? t.bargeCaption : t.listeningCaption}
        </Text>
        {hasTranscript ? <HeardCard text={props.userText || ''} language={props.language} /> : null}
        <Waveform mode="listening" />
      </View>
      <EndHint onPress={props.onEnd} language={props.language} />
    </View>
  )
}

function Thinking(props: Props) {
  const t = getMirrorCopy(props.language)
  return (
    <View style={styles.conversationScene}>
      <ConversationHeader label={props.progressText || t.todaysConversation} />
      <View style={styles.centerContent}>
        <AriaPortrait active mode="thinking" size={188} />
        <Text style={styles.sceneTitle}>{t.thinkingTitle}</Text>
        <Text style={styles.sceneCaption}>{t.thinkingCaption}</Text>
        {props.userText ? <HeardCard text={props.userText} language={props.language} /> : null}
        <ThinkingDots />
      </View>
      <EndHint onPress={props.onEnd} language={props.language} />
    </View>
  )
}

function Speaking(props: Props) {
  const t = getMirrorCopy(props.language)
  return (
    <View style={styles.conversationScene}>
      <ConversationHeader label={props.progressText || t.todaysConversation} />
      <View style={styles.centerContent}>
        <Text style={styles.ariaLabel}>Aria</Text>
        <AriaPortrait active mode="speaking" size={206} />
        <Text numberOfLines={7} style={styles.ariaSpeech}>{props.assistantText || t.ariaSpeakingFallback}</Text>
        <Waveform mode="speaking" />
        <View style={styles.interruptHint}>
          <Ionicons name="mic-outline" size={22} color={c.sageDeep} />
          <Text style={styles.interruptText}>{t.interruptHint}</Text>
        </View>
        {props.microphoneActive ? <MicPill label={t.micActive} quiet /> : null}
      </View>
      <EndHint onPress={props.onEnd} language={props.language} />
    </View>
  )
}

function Closing(props: Props) {
  const t = getMirrorCopy(props.language)
  const saving = props.state === 'saving'
  return (
    <View style={styles.conversationScene}>
      <ConversationHeader label={saving ? t.savingHeader : t.goodbyeHeader} />
      <View style={styles.centerContent}>
        <AriaPortrait active={!saving} mode={saving ? 'thinking' : 'speaking'} size={194} />
        <Text style={styles.sceneTitle}>
          {saving ? t.savingTitle : props.assistantText || t.goodbyeFallback}
        </Text>
        <Text style={styles.sceneCaption}>
          {saving ? t.savingCaption : t.goodbyeCaption}
        </Text>
        {saving ? <ActivityIndicator color={c.goldDeep} style={styles.spinner} /> : <Waveform mode="speaking" />}
      </View>
    </View>
  )
}

function Problem(props: Props) {
  const t = getMirrorCopy(props.language)
  const offline = props.state === 'offline'
  const microphone = props.state === 'microphone_error'
  const icon = offline ? 'cloud-offline-outline' : microphone ? 'mic-off-outline' : 'alert-circle-outline'
  const title = offline ? t.offlineTitle : microphone ? t.micErrorTitle : t.serviceTitle
  const body = offline ? t.offlineBody : microphone ? t.micErrorBody : props.statusText || t.serviceBodyFallback
  return (
    <View style={styles.problemScene}>
      <Brand lockup />
      <View style={[styles.problemIcon, microphone && styles.problemIconError]}>
        <Ionicons name={icon} size={50} color={microphone ? c.coral : c.goldDeep} />
      </View>
      <Text style={styles.problemTitle}>{title}</Text>
      <Text style={styles.problemBody}>{body}</Text>
      {offline ? (
        <View style={styles.offlineAssurance}>
          <Ionicons name="shield-checkmark-outline" size={23} color={c.sageDeep} />
          <Text style={styles.offlineAssuranceText}>{t.offlineAssurance}</Text>
        </View>
      ) : null}
      {props.onRetry ? (
        <Pressable accessibilityRole="button" onPress={props.onRetry} style={styles.retryButton}>
          <Text style={styles.retryText}>{offline ? t.retryOffline : t.retry}</Text>
        </Pressable>
      ) : null}
    </View>
  )
}

function ConversationHeader({ label }: { label: string }) {
  return (
    <View style={styles.conversationHeader}>
      <Brand />
      <Text style={styles.stateLabel}>{label}</Text>
    </View>
  )
}

function Brand({ lockup = false }: { lockup?: boolean }) {
  return (
    <View style={styles.brandRow}>
      <View style={styles.brandPetals}>
        <View style={[styles.brandPetal, styles.brandPetalLeft]} />
        <View style={[styles.brandPetal, styles.brandPetalCenter]} />
        <View style={[styles.brandPetal, styles.brandPetalRight]} />
      </View>
      {lockup ? <Text style={styles.brandText}>REFLEXION</Text> : null}
    </View>
  )
}

function MicPill({ label, quiet = false }: { label: string; quiet?: boolean }) {
  return (
    <View style={[styles.micPill, quiet && styles.micPillQuiet]}>
      <View style={styles.micDot} />
      <Text style={styles.micText}>{label}</Text>
    </View>
  )
}

function HeardCard({ text, language }: { text: string; language?: string }) {
  const t = getMirrorCopy(language)
  return (
    <View style={styles.heardCard}>
      <View style={styles.heardCheck}>
        <Ionicons name="checkmark" size={18} color={c.white} />
      </View>
      <View style={styles.heardCopy}>
        <Text style={styles.heardLabel}>{t.heardLabel}</Text>
        <Text numberOfLines={3} style={styles.heardText}>{text}</Text>
      </View>
    </View>
  )
}

function EndHint({ onPress, language }: { onPress?: () => void; language?: string }) {
  const t = getMirrorCopy(language)
  return (
    <Pressable accessibilityRole="button" onPress={onPress} style={styles.endHint}>
      <Text style={styles.endHintText}>{t.endHint}</Text>
      <Ionicons name="close" size={18} color={c.textSecondary} />
    </Pressable>
  )
}

function AriaPortrait({ active, mode, size }: { active: boolean; mode: 'listening' | 'thinking' | 'speaking'; size: number }) {
  const pulse = usePulse(active, mode === 'speaking' ? 820 : 1500)
  const scale = pulse.interpolate({ inputRange: [0, 1], outputRange: [0.98, mode === 'speaking' ? 1.045 : 1.025] })
  const ringColor = mode === 'listening' ? c.sage : mode === 'speaking' ? c.gold : c.taupe
  return (
    <View style={[styles.portraitFrame, { height: size, width: size }]}>
      <Animated.View
        style={[
          styles.portraitRing,
          { borderColor: ringColor, borderRadius: size / 2, height: size, transform: [{ scale }], width: size },
        ]}
      />
      <View style={[styles.portraitClip, { borderRadius: size / 2, height: size - 14, width: size - 14 }]}>
        <Image resizeMode="cover" source={ARIA_AVATAR} style={styles.portraitImage} />
      </View>
      {mode === 'listening' ? (
        <View style={styles.portraitMic}>
          <Ionicons name="mic" size={23} color={c.white} />
        </View>
      ) : null}
    </View>
  )
}

function ReadyOrb({ active }: { active: boolean }) {
  const pulse = usePulse(active, 1500)
  const scale = pulse.interpolate({ inputRange: [0, 1], outputRange: [0.94, 1.08] })
  return (
    <Animated.View style={[styles.readyOrbOuter, { transform: [{ scale }] }]}>
      <View style={styles.readyOrbInner}>
        <Ionicons name="sparkles" size={27} color={c.white} />
      </View>
    </Animated.View>
  )
}

function Waveform({ mode }: { mode: 'listening' | 'speaking' }) {
  const active = usePulse(true, mode === 'speaking' ? 680 : 1050)
  const color = mode === 'speaking' ? c.goldDeep : c.sageDeep
  return (
    <View style={styles.waveform}>
      {[8, 13, 20, 29, 18, 34, 23, 13, 9, 16, 26, 17, 10].map((height, index) => (
        <Animated.View
          key={`${height}-${index}`}
          style={[
            styles.waveBar,
            {
              backgroundColor: color,
              height,
              opacity: active.interpolate({
                inputRange: [0, 1],
                outputRange: index % 2 ? [0.35, 0.9] : [0.85, 0.38],
              }),
            },
          ]}
        />
      ))}
    </View>
  )
}

function ThinkingDots() {
  const pulse = usePulse(true, 900)
  return (
    <View style={styles.thinkingDots}>
      {[0, 1, 2].map((dot) => (
        <Animated.View
          key={dot}
          style={[
            styles.thinkingDot,
            { opacity: pulse.interpolate({ inputRange: [0, 0.5, 1], outputRange: dot === 1 ? [0.35, 1, 0.35] : [0.8, 0.35, 0.8] }) },
          ]}
        />
      ))}
    </View>
  )
}

function usePulse(active: boolean, duration: number) {
  const pulse = useRef(new Animated.Value(0)).current
  useEffect(() => {
    if (!active) {
      pulse.stopAnimation()
      pulse.setValue(0)
      return
    }
    const loop = Animated.loop(Animated.sequence([
      Animated.timing(pulse, { duration, easing: Easing.inOut(Easing.sin), toValue: 1, useNativeDriver: true }),
      Animated.timing(pulse, { duration, easing: Easing.inOut(Easing.sin), toValue: 0, useNativeDriver: true }),
    ]))
    loop.start()
    return () => loop.stop()
  }, [active, duration, pulse])
  return pulse
}

const styles = StyleSheet.create({
  shell: { backgroundColor: c.cream, flex: 1, overflow: 'hidden' },
  stage: { flex: 1, minHeight: 640 },
  warmWash: { backgroundColor: 'rgba(231,207,166,0.18)', borderRadius: 360, height: 680, left: -280, position: 'absolute', top: -350, width: 680 },
  reflectionTop: { backgroundColor: 'rgba(255,255,255,0.62)', borderRadius: 280, height: 580, position: 'absolute', right: -330, top: -210, transform: [{ rotate: '-17deg' }], width: 470 },
  reflectionBottom: { backgroundColor: 'rgba(171,197,161,0.09)', borderRadius: 300, bottom: -360, height: 650, left: -300, position: 'absolute', transform: [{ rotate: '19deg' }], width: 500 },
  ambient: { flex: 1, paddingBottom: 38, paddingHorizontal: 32, paddingTop: 28 },
  homeHeader: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between' },
  ambientGreeting: { alignItems: 'center', marginTop: '15%' },
  greeting: { color: c.textSecondary, fontFamily: f.body, fontSize: 28, letterSpacing: 0.2 },
  name: { color: c.text, fontFamily: f.display, fontSize: 58, lineHeight: 69, marginTop: 2, maxWidth: '100%' },
  date: { color: c.textSecondary, fontFamily: f.body, fontSize: 26, lineHeight: 36, marginTop: 32 },
  time: { color: c.text, fontFamily: f.bodyMedium, fontSize: 48, fontVariant: ['tabular-nums'], includeFontPadding: false, letterSpacing: -1.5, lineHeight: 58, marginTop: 2 },
  widgetRow: { flexDirection: 'row', gap: 10, justifyContent: 'center', marginTop: 28 },
  widgetCard: { alignItems: 'center', backgroundColor: 'rgba(255,255,255,0.66)', borderColor: c.lineWarm, borderRadius: 18, borderWidth: 1, flex: 1, flexDirection: 'row', gap: 10, maxWidth: 210, minHeight: 74, paddingHorizontal: 14, paddingVertical: 12, shadowColor: c.shadow, shadowOffset: { height: 6, width: 0 }, shadowOpacity: 0.18, shadowRadius: 14 },
  widgetCopy: { flex: 1 },
  widgetValue: { color: c.text, fontFamily: f.bodyMedium, fontSize: 17 },
  widgetLabel: { color: c.textSecondary, fontFamily: f.body, fontSize: 12, marginTop: 2 },
  ambientPrompt: { alignItems: 'center', bottom: 38, left: 28, position: 'absolute', right: 28 },
  prompt: { color: c.text, fontFamily: f.bodyMedium, fontSize: 28, marginTop: 18 },
  promptSub: { color: c.textSecondary, fontFamily: f.body, fontSize: 21, marginTop: 2 },
  wakeNote: { color: c.amber, fontFamily: f.body, fontSize: 13, marginTop: 8, textAlign: 'center' },
  conversationScene: { flex: 1, paddingBottom: 26, paddingHorizontal: 28, paddingTop: 24 },
  conversationHeader: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between', minHeight: 40 },
  stateLabel: { color: c.goldDeep, fontFamily: f.bodyMedium, fontSize: 12, letterSpacing: 1.8 },
  centerContent: { alignItems: 'center', flex: 1, justifyContent: 'center', paddingBottom: 28 },
  sceneTitle: { color: c.text, fontFamily: f.display, fontSize: 39, lineHeight: 49, marginTop: 22, maxWidth: 560, textAlign: 'center' },
  sceneTitleCompact: { color: c.text, fontFamily: f.display, fontSize: 35, lineHeight: 44, maxWidth: 560, textAlign: 'center' },
  sceneCaption: { color: c.textSecondary, fontFamily: f.body, fontSize: 23, lineHeight: 33, marginTop: 14, maxWidth: 520, textAlign: 'center' },
  question: { color: c.text, fontFamily: f.display, fontSize: 32, lineHeight: 43, marginBottom: 24, maxWidth: 570, textAlign: 'center' },
  ariaLabel: { color: c.goldDeep, fontFamily: f.bodyMedium, fontSize: 17, letterSpacing: 2.5, marginBottom: 18, textTransform: 'uppercase' },
  ariaSpeech: { color: c.text, fontFamily: f.display, fontSize: 32, lineHeight: 44, marginTop: 24, maxWidth: 570, textAlign: 'center' },
  spinner: { marginTop: 25 },
  brandRow: { alignItems: 'center', flexDirection: 'row', gap: 9 },
  brandPetals: { height: 28, position: 'relative', width: 37 },
  brandPetal: { backgroundColor: c.goldDeep, borderBottomLeftRadius: 14, borderBottomRightRadius: 3, borderTopLeftRadius: 3, borderTopRightRadius: 14, height: 22, position: 'absolute', top: 3, width: 10 },
  brandPetalLeft: { left: 5, transform: [{ rotate: '-34deg' }] },
  brandPetalCenter: { left: 14, top: 0, transform: [{ rotate: '45deg' }] },
  brandPetalRight: { right: 5, transform: [{ rotate: '124deg' }] },
  brandText: { color: c.text, fontFamily: f.display, fontSize: 14, letterSpacing: 3.5 },
  micPill: { alignItems: 'center', backgroundColor: 'rgba(171,197,161,0.22)', borderColor: 'rgba(99,123,95,0.24)', borderRadius: 19, borderWidth: 1, flexDirection: 'row', gap: 8, marginTop: 20, paddingHorizontal: 13, paddingVertical: 8 },
  micPillQuiet: { marginTop: 6 },
  micDot: { backgroundColor: c.sageDeep, borderRadius: 4, height: 8, width: 8 },
  micText: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 10, letterSpacing: 1.3 },
  heardCard: { alignItems: 'flex-start', backgroundColor: 'rgba(255,255,255,0.72)', borderColor: 'rgba(99,123,95,0.24)', borderRadius: 20, borderWidth: 1, flexDirection: 'row', gap: 13, marginTop: 22, maxWidth: 540, paddingHorizontal: 18, paddingVertical: 16, width: '100%' },
  heardCheck: { alignItems: 'center', backgroundColor: c.sageDeep, borderRadius: 15, height: 30, justifyContent: 'center', width: 30 },
  heardCopy: { flex: 1, gap: 4 },
  heardLabel: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 10, letterSpacing: 1.5 },
  heardText: { color: c.text, fontFamily: f.body, fontSize: 18, lineHeight: 26 },
  interruptHint: { alignItems: 'center', flexDirection: 'row', gap: 9, marginTop: 16 },
  interruptText: { color: c.textSecondary, fontFamily: f.body, fontSize: 17 },
  endHint: { alignItems: 'center', alignSelf: 'center', backgroundColor: 'rgba(255,255,255,0.55)', borderColor: c.lineWarm, borderRadius: 24, borderWidth: 1, flexDirection: 'row', gap: 8, paddingHorizontal: 18, paddingVertical: 11 },
  endHintText: { color: c.textSecondary, fontFamily: f.bodyMedium, fontSize: 14 },
  portraitFrame: { alignItems: 'center', justifyContent: 'center' },
  portraitRing: { backgroundColor: 'rgba(255,255,255,0.62)', borderWidth: 2, position: 'absolute', shadowColor: c.goldDeep, shadowOffset: { height: 5, width: 0 }, shadowOpacity: 0.18, shadowRadius: 18 },
  portraitClip: { backgroundColor: c.sand, borderColor: c.white, borderWidth: 4, overflow: 'hidden' },
  portraitImage: { height: '100%', width: '100%' },
  portraitMic: { alignItems: 'center', backgroundColor: c.goldDeep, borderColor: c.white, borderRadius: 24, borderWidth: 3, bottom: -5, height: 48, justifyContent: 'center', position: 'absolute', width: 48 },
  readyOrbOuter: { alignItems: 'center', backgroundColor: 'rgba(231,207,166,0.28)', borderColor: 'rgba(185,137,84,0.35)', borderRadius: 44, borderWidth: 1, height: 88, justifyContent: 'center', width: 88 },
  readyOrbInner: { alignItems: 'center', backgroundColor: c.goldDeep, borderRadius: 28, height: 56, justifyContent: 'center', shadowColor: c.goldDeep, shadowOpacity: 0.34, shadowRadius: 17, width: 56 },
  waveform: { alignItems: 'center', flexDirection: 'row', gap: 4, height: 40, justifyContent: 'center', marginTop: 22 },
  waveBar: { borderRadius: 2, width: 3 },
  thinkingDots: { flexDirection: 'row', gap: 10, marginTop: 28 },
  thinkingDot: { backgroundColor: c.goldDeep, borderRadius: 4, height: 8, width: 8 },
  problemScene: { alignItems: 'center', flex: 1, justifyContent: 'center', paddingHorizontal: 34 },
  problemIcon: { alignItems: 'center', backgroundColor: 'rgba(231,207,166,0.24)', borderColor: c.lineWarm, borderRadius: 52, borderWidth: 1, height: 104, justifyContent: 'center', marginTop: 50, width: 104 },
  problemIconError: { backgroundColor: 'rgba(201,120,110,0.10)', borderColor: 'rgba(201,120,110,0.22)' },
  problemTitle: { color: c.text, fontFamily: f.display, fontSize: 39, lineHeight: 49, marginTop: 28, maxWidth: 560, textAlign: 'center' },
  problemBody: { color: c.textSecondary, fontFamily: f.body, fontSize: 21, lineHeight: 31, marginTop: 18, maxWidth: 520, textAlign: 'center' },
  offlineAssurance: { alignItems: 'center', backgroundColor: 'rgba(171,197,161,0.18)', borderRadius: 20, flexDirection: 'row', gap: 10, marginTop: 28, paddingHorizontal: 15, paddingVertical: 12 },
  offlineAssuranceText: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 15 },
  retryButton: { backgroundColor: c.text, borderRadius: 27, marginTop: 30, paddingHorizontal: 27, paddingVertical: 15 },
  retryText: { color: c.cream, fontFamily: f.bodyMedium, fontSize: 16 },
})
