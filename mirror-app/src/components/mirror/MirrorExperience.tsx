import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import {
  ActivityIndicator,
  Animated,
  Easing,
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native'

import { mirrorColors as c, mirrorFonts as f } from '../../theme/mirrorTheme'
import { getMirrorCopy } from './mirrorStrings'
import { MirrorIcon, type MirrorIconName } from './MirrorIcon'

const WAKE_PHRASE = (process.env.EXPO_PUBLIC_WAKE_WORD_PHRASE || 'Hello Aria').trim()
const ARIA_AVATAR = require('../../../assets/images/aria-avatar.png')

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

export type MirrorHomeStatus = 'ready' | 'partial' | 'complete' | 'paused' | 'away' | 'offline'

export type MirrorHomeWidget = {
  icon: MirrorIconName
  label: string
  value: string
}

export type MirrorHomeMessage = {
  sender: string
  preview: string
  kind?: 'text' | 'photo' | 'voice'
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
  /** Safe field-debug detail. The caller controls whether it is enabled. */
  problemDetail?: string
  progressText?: string
  language?: string
  homeWidgets?: MirrorHomeWidget[]
  homeStatus?: MirrorHomeStatus
  homeMessage?: MirrorHomeMessage | null
  wakeListening?: boolean
  wakeError?: string
  bargeInActive?: boolean
  microphoneActive?: boolean
  /** Live browser-demo microphone level. Omitted for the native/production path. */
  microphoneLevel?: number
  onInterrupt?: () => void
  onRepeat?: () => void
  onContinue?: () => void
  onStop?: () => void
  onOpenStatus?: () => void
  onOpenRoutine?: () => void
  onOpenMessage?: () => void
  onOpenConsent?: () => void
  onOpenResearch?: () => void
  onEnd?: () => void
  onRetry?: () => void
  /** Tap and wake-word activation share the same session router. */
  onBegin?: () => void
  /** Kept as an invisible long-press affordance for technician testing; no menu is shown. */
  onBeginCheckin?: () => void
}

export function MirrorExperience(props: Props) {
  const [confirmingEnd, setConfirmingEnd] = useState(false)
  const conversationProps: Props = {
    ...props,
    onStop: props.onStop ? () => setConfirmingEnd(true) : undefined,
  }

  return (
    <View style={styles.shell}>
      <View style={styles.stage}>
        <Atmosphere />
        {props.state === 'ambient' ? <Ambient {...props} /> : null}
        {props.state === 'connecting' ? <Connecting {...props} /> : null}
        {props.state === 'listening' || props.state === 'heard' ? <Listening {...conversationProps} /> : null}
        {props.state === 'thinking' ? <Thinking {...conversationProps} /> : null}
        {props.state === 'speaking' ? <Speaking {...conversationProps} /> : null}
        {props.state === 'closing' || props.state === 'saving' ? <Closing {...props} /> : null}
        {props.state === 'offline' || props.state === 'microphone_error' || props.state === 'service_error'
          ? <Problem {...props} />
          : null}
        {confirmingEnd ? (
          <View style={styles.endConfirmBackdrop}>
            <View accessibilityViewIsModal style={styles.endConfirmCard}>
              <Text style={styles.endConfirmTitle}>End conversation?</Text>
              <Text style={styles.endConfirmBody}>Your progress will be saved and you can continue later.</Text>
              <View style={styles.endConfirmActions}>
                <Pressable accessibilityRole="button" onPress={() => setConfirmingEnd(false)} style={styles.endConfirmSecondary}>
                  <Text style={styles.endConfirmSecondaryText}>Keep talking</Text>
                </Pressable>
                <Pressable
                  accessibilityRole="button"
                  onPress={() => {
                    setConfirmingEnd(false)
                    props.onStop?.()
                  }}
                  style={styles.endConfirmPrimary}
                >
                  <Text style={styles.endConfirmPrimaryText}>End conversation now</Text>
                </Pressable>
              </View>
            </View>
          </View>
        ) : null}
      </View>
    </View>
  )
}

function Atmosphere() {
  return (
    <View pointerEvents="none" style={StyleSheet.absoluteFill}>
      <View style={styles.topWash} />
      <View style={styles.bottomWash} />
      <View style={styles.cornerLeafOne} />
      <View style={styles.cornerLeafTwo} />
    </View>
  )
}

function Ambient(props: Props) {
  const t = getMirrorCopy(props.language)
  const widgets = props.homeWidgets?.slice(0, 2) ?? []
  const status = props.homeStatus || 'ready'
  const homeCopy = status === 'complete'
    ? { title: `Hi ${props.patientName}`, body: 'Your daily check-in is complete.', prompt: `Say “${WAKE_PHRASE}” anytime to chat.` }
    : status === 'partial'
      ? { title: `Welcome back, ${props.patientName}`, body: 'We can continue where we left off.', prompt: `Say “${WAKE_PHRASE}” when you’re ready.` }
    : status === 'paused'
      ? { title: `Hello, ${props.patientName}`, body: 'Reflexion is paused for now.', prompt: 'Ask your caregiver to resume Reflexion when you are ready.' }
      : status === 'away'
        ? { title: `Hello, ${props.patientName}`, body: 'You are marked away right now.', prompt: `Say “${WAKE_PHRASE}” if you would like to talk.` }
      : status === 'offline'
          ? { title: `Hello, ${props.patientName}`, body: 'Aria is temporarily offline.', prompt: 'Some features may be unavailable.' }
          : { title: `${props.greeting}, ${props.patientName}`, body: 'Your daily check-in is ready.', prompt: `Say “${WAKE_PHRASE}” when you’re ready.` }

  return (
    <ScrollView contentContainerStyle={styles.homeScroll} showsVerticalScrollIndicator={false}>
      <MirrorHeader {...props} statusLabel={status === 'offline' ? 'Mirror offline' : 'Mirror ready'} onOpenStatus={props.onOpenStatus} />

      <View style={styles.homeHero}>
        <View style={styles.avatarFrameLarge}>
          <View style={styles.avatarRingLarge} />
          <Image resizeMode="cover" source={ARIA_AVATAR} style={styles.avatarLarge} />
          <View style={styles.ariaNameBadge}>
            <BrandMark small />
            <Text style={styles.ariaNameText}>Aria</Text>
          </View>
        </View>
        <Text adjustsFontSizeToFit numberOfLines={1} style={styles.homeGreeting}>{homeCopy.title}</Text>
        <Text style={styles.homeBody}>{homeCopy.body}</Text>
      </View>

      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Start a conversation with Aria"
        disabled={status === 'offline' || status === 'paused'}
        onLongPress={props.onBeginCheckin}
        onPress={props.onBegin}
        delayLongPress={500}
        style={({ pressed }) => [styles.startCard, pressed && styles.pressed, status === 'offline' && styles.disabledCard]}
      >
        <ReadyOrb active={Boolean(props.wakeListening)} />
        <Text style={styles.startTitle}>{status === 'paused' ? 'Reflexion is paused' : status === 'complete' ? `Talk to Aria` : `Say “${WAKE_PHRASE}”`}</Text>
        <Text style={styles.startSubtitle}>{homeCopy.prompt}</Text>
        {props.wakeError ? <Text style={styles.wakeNote}>{props.wakeError}</Text> : null}
      </Pressable>

      {widgets.length ? (
        <View style={styles.homeCardRow}>
          {widgets.map((widget) => {
            const isRoutine = widget.label.toLowerCase().includes('medication') || widget.label.toLowerCase().includes('routine')
            const card = (
              <View style={styles.homeCard}>
                <View style={styles.homeCardIcon}><MirrorIcon name={widget.icon} size={25} color={c.sageDeep} /></View>
                <View style={styles.homeCardCopy}>
                  <Text numberOfLines={2} style={styles.homeCardTitle}>{widget.value}</Text>
                  <Text numberOfLines={1} style={styles.homeCardLabel}>{widget.label}</Text>
                </View>
                {isRoutine && props.onOpenRoutine ? <MirrorIcon name="chevron-forward" size={22} color={c.sageDeep} /> : null}
              </View>
            )
            return isRoutine && props.onOpenRoutine
              ? <Pressable key={`${widget.label}-${widget.value}`} accessibilityRole="button" onPress={props.onOpenRoutine}>{card}</Pressable>
              : <View key={`${widget.label}-${widget.value}`}>{card}</View>
          })}
        </View>
      ) : null}

      {props.homeMessage ? (
        <Pressable
          accessibilityRole="button"
          accessibilityLabel={`Open family message from ${props.homeMessage.sender}`}
          onPress={props.onOpenMessage}
          style={({ pressed }) => [styles.messageCard, pressed && styles.pressed]}
        >
          <View style={[styles.messageCardIcon, styles.messageIcon]}>
            <MirrorIcon name={props.homeMessage.kind === 'voice' ? 'mic-outline' : 'mail-outline'} size={38} color={c.goldDeep} />
          </View>
          <View style={styles.messageCardCopy}>
            <Text numberOfLines={2} style={styles.messageCardTitle}>A message from {props.homeMessage.sender}</Text>
            <Text numberOfLines={3} style={styles.messageCardLabel}>{props.homeMessage.preview}</Text>
          </View>
          <MirrorIcon name="chevron-forward" size={32} color={c.goldDeep} />
        </Pressable>
      ) : null}

      <View style={styles.homeFooterActions}>
        <Pressable onPress={props.onOpenStatus} style={styles.footerAction}>
          <MirrorIcon name="help-circle-outline" size={22} color={c.sageDeep} />
          <Text style={styles.footerActionText}>Device status & help</Text>
        </Pressable>
        <Pressable onPress={props.onOpenConsent} style={styles.footerAction}>
          <MirrorIcon name="shield-checkmark-outline" size={22} color={c.sageDeep} />
          <Text style={styles.footerActionText}>Consent & control</Text>
        </Pressable>
      </View>
    </ScrollView>
  )
}

function Connecting(props: Props) {
  const t = getMirrorCopy(props.language)
  return (
    <ConversationFrame {...props} label="GETTING READY">
      <AriaPortrait active mode="thinking" size={244} />
      <Text style={styles.conversationTitle}>{t.hi(props.patientName)}</Text>
      <Text style={styles.conversationSubtitle}>{props.statusText || 'Aria is getting ready to listen.'}</Text>
      <ActivityIndicator color={c.goldDeep} size="small" style={styles.spinner} />
    </ConversationFrame>
  )
}

function Listening(props: Props) {
  const t = getMirrorCopy(props.language)
  const heard = props.state === 'heard'
  return (
    <ConversationFrame {...props} label="MIRROR READY">
      <View style={styles.promptCard}>
        <View style={styles.promptQuote}><Text style={styles.promptQuoteText}>“</Text></View>
        <Text style={styles.promptLabel}>Aria asked</Text>
        <ScrollView style={styles.promptScroll} showsVerticalScrollIndicator={false}>
          <Text style={styles.promptText}>{props.assistantText || 'How are you feeling today?'}</Text>
        </ScrollView>
      </View>
      <View style={[styles.listenOrb, heard && styles.listenOrbHeard]}>
        <MirrorIcon name="mic" size={92} color={c.sageDeep} />
      </View>
      <Text style={styles.conversationTitle}>{heard ? 'I can hear you.' : 'I’m listening…'}</Text>
      <Text style={styles.conversationSubtitle}>{props.bargeInActive ? t.bargeCaption : 'Take your time. I’ll wait until you finish.'}</Text>
      <Waveform mode="listening" microphoneLevel={props.microphoneLevel} />
      <ConversationActions onRepeat={props.onRepeat} onStop={props.onStop} repeatLabel="Repeat question" />
    </ConversationFrame>
  )
}

function Thinking(props: Props) {
  return (
    <ConversationFrame {...props} label="ONE MOMENT">
      <AriaPortrait active mode="thinking" size={224} />
      <Text style={styles.conversationTitle}>One moment…</Text>
      <Text style={styles.conversationSubtitle}>Aria is thinking about what you said.</Text>
      <ThinkingDots />
      <ConversationActions onStop={props.onStop} />
    </ConversationFrame>
  )
}

function Speaking(props: Props) {
  const t = getMirrorCopy(props.language)
  return (
    <ConversationFrame {...props} label="MIRROR READY">
      <AriaPortrait active mode="speaking" size={260} />
      <View style={styles.speakingPill}>
        <Waveform mode="speaking" compact />
        <Text style={styles.speakingPillText}>Aria is speaking</Text>
      </View>
      <ScrollView style={styles.speechScroll} showsVerticalScrollIndicator={false}>
        <Text style={styles.speechText}>{props.assistantText || t.ariaSpeakingFallback}</Text>
      </ScrollView>
      <ConversationActions onRepeat={props.onRepeat} onStop={props.onStop} repeatLabel="Repeat" />
      {props.onInterrupt ? (
        <Pressable accessibilityRole="button" onPress={props.onInterrupt} style={styles.interruptButton}>
          <MirrorIcon name="mic-outline" size={24} color={c.sageDeep} />
          <Text style={styles.interruptText}>You can speak at any time</Text>
        </Pressable>
      ) : null}
    </ConversationFrame>
  )
}

function Closing(props: Props) {
  const saving = props.state === 'saving'
  const incomplete = props.homeStatus === 'partial' || props.assistantText?.includes('continue another time')
  return (
    <ConversationFrame {...props} label={saving ? 'SAVING' : 'CONVERSATION COMPLETE'}>
      <AriaPortrait active={!saving} mode={saving ? 'thinking' : 'speaking'} size={264} />
      <Text style={styles.conversationTitle}>{saving ? 'Saving your conversation…' : props.assistantText || `Thank you, ${props.patientName}.`}</Text>
      <Text style={styles.conversationSubtitle}>{saving ? 'Your information will sync securely.' : incomplete ? 'Your progress is saved for another time.' : 'Your check-in is complete for today.'}</Text>
      {saving ? <ActivityIndicator color={c.goldDeep} style={styles.spinner} /> : <Waveform mode="speaking" />}
    </ConversationFrame>
  )
}

function Problem(props: Props) {
  const t = getMirrorCopy(props.language)
  const offline = props.state === 'offline'
  const microphone = props.state === 'microphone_error'
  const icon = offline ? 'cloud-offline-outline' : microphone ? 'mic-off-outline' : 'alert-circle-outline'
  const title = offline ? t.offlineTitle : microphone ? t.micErrorTitle : 'Reflexion service unavailable'
  const body = offline ? t.offlineBody : microphone ? t.micErrorBody : 'Your mirror is connected, but Aria cannot be reached right now.'
  return (
    <View style={styles.problemScene}>
      <MirrorHeader {...props} statusLabel="Needs attention" onOpenStatus={props.onOpenStatus} />
      <View style={styles.problemContent}>
        <View style={[styles.problemIcon, microphone && styles.problemIconError]}>
          <MirrorIcon name={icon} size={54} color={microphone ? c.coral : c.goldDeep} />
        </View>
        <Text style={styles.problemTitle}>{title}</Text>
        <Text style={styles.problemBody}>{body}</Text>
        {offline ? <Text style={styles.problemNote}>Wi-Fi is needed for conversations.</Text> : null}
        {props.onRetry ? (
          <Pressable accessibilityRole="button" onPress={props.onRetry} style={styles.primaryButton}>
            <Text style={styles.primaryButtonText}>{offline ? 'Open connection help' : 'Try again'}</Text>
          </Pressable>
        ) : null}
        {props.problemDetail ? <Text selectable style={styles.problemDetail}>{props.problemDetail}</Text> : null}
      </View>
    </View>
  )
}

function ConversationFrame(props: Props & { label: string; children: ReactNode }) {
  return (
    <View style={styles.conversationScene}>
      <MirrorHeader {...props} statusLabel={props.label === 'ONE MOMENT' ? 'Mirror ready' : 'Mirror ready'} onOpenStatus={props.onOpenStatus} />
      <View style={styles.conversationContent}>{props.children}</View>
      <Pressable onPress={props.onEnd} style={styles.endHint}>
        <Text style={styles.endHintText}>Say “goodbye” to finish</Text>
        <MirrorIcon name="close" size={18} color={c.textSecondary} />
      </Pressable>
    </View>
  )
}

function MirrorHeader(props: Props & { statusLabel?: string; onOpenStatus?: () => void }) {
  const weather = props.homeWidgets?.find((widget) => widget.label.toLowerCase().includes('weather') || widget.icon.includes('sunny'))
  return (
    <View style={styles.header}>
      <View style={styles.headerSide}>
        <Text style={styles.headerTime}>{props.time}</Text>
        <Text style={styles.headerDate}>{props.date}</Text>
      </View>
      <Brand lockup />
      <Pressable accessibilityRole="button" accessibilityLabel="Open device status and help" onPress={props.onOpenStatus} style={styles.headerSideRight}>
        <View style={styles.headerWeather}>
          <MirrorIcon name={weather?.icon || 'wifi'} size={25} color={c.text} />
          {weather ? <Text style={styles.weatherValue}>{weather.value}</Text> : null}
        </View>
        <Text style={styles.headerStatus}>{props.statusLabel || 'Mirror ready'}</Text>
      </Pressable>
    </View>
  )
}

function ConversationActions({ onRepeat, onStop, repeatLabel = 'Repeat' }: { onRepeat?: () => void; onStop?: () => void; repeatLabel?: string }) {
  return (
    <View style={styles.conversationActions}>
      {onRepeat ? (
        <Pressable accessibilityRole="button" onPress={onRepeat} style={styles.secondaryButton}>
          <MirrorIcon name="refresh-outline" size={26} color={c.sageDeep} />
          <Text style={styles.secondaryButtonText}>{repeatLabel}</Text>
        </Pressable>
      ) : null}
      {onStop ? (
        <Pressable accessibilityRole="button" onPress={onStop} style={styles.stopButton}>
          <MirrorIcon name="stop-circle-outline" size={27} color={c.white} />
          <Text style={styles.stopButtonText}>End conversation</Text>
        </Pressable>
      ) : null}
    </View>
  )
}

function Brand({ lockup = false }: { lockup?: boolean }) {
  return (
    <View style={styles.brandLockup}>
      <BrandMark />
      {lockup ? <Text style={styles.brandText}>Reflexion</Text> : null}
    </View>
  )
}

function BrandMark({ small = false }: { small?: boolean }) {
  return (
    <View style={[styles.brandMark, small && styles.brandMarkSmall]}>
      <View style={[styles.brandPetal, styles.brandPetalLeft, small && styles.brandPetalSmall]} />
      <View style={[styles.brandPetal, styles.brandPetalCenter, small && styles.brandPetalSmall]} />
      <View style={[styles.brandPetal, styles.brandPetalRight, small && styles.brandPetalSmall]} />
      <View style={[styles.brandSpark, small && styles.brandSparkSmall]} />
    </View>
  )
}

function AriaPortrait({ active, mode, size }: { active: boolean; mode: 'listening' | 'thinking' | 'speaking'; size: number }) {
  const pulse = usePulse(active, mode === 'speaking' ? 820 : 1500)
  const scale = pulse.interpolate({ inputRange: [0, 1], outputRange: [0.985, mode === 'speaking' ? 1.035 : 1.018] })
  const ringColor = mode === 'listening' ? c.sageDeep : mode === 'speaking' ? c.goldDeep : c.taupe
  return (
    <View style={[styles.portraitFrame, { height: size, width: size }]}>
      <Animated.View style={[styles.portraitRing, { borderColor: ringColor, borderRadius: size / 2, height: size, transform: [{ scale }], width: size }]} />
      <View style={[styles.portraitClip, { borderRadius: size / 2, height: size - 18, width: size - 18 }]}>
        <Image resizeMode="cover" source={ARIA_AVATAR} style={styles.portraitImage} />
      </View>
    </View>
  )
}

function ReadyOrb({ active }: { active: boolean }) {
  const pulse = usePulse(active, 1500)
  const scale = pulse.interpolate({ inputRange: [0, 1], outputRange: [0.96, 1.04] })
  return (
    <Animated.View style={[styles.readyOrb, { transform: [{ scale }] }]}>
      <MirrorIcon name="mic" size={42} color={c.white} />
    </Animated.View>
  )
}

function Waveform({ mode, compact = false, microphoneLevel }: { mode: 'listening' | 'speaking'; compact?: boolean; microphoneLevel?: number }) {
  const liveLevel = typeof microphoneLevel === 'number'
  const pulse = usePulse(!liveLevel, mode === 'speaking' ? 680 : 1050)
  const color = mode === 'speaking' ? c.goldDeep : c.sageDeep
  const bars = compact ? [8, 15, 23, 13, 8] : [7, 13, 20, 30, 18, 36, 24, 14, 9, 17, 27, 18, 11, 19, 28, 12, 8]
  return (
    <View style={[styles.waveform, compact && styles.waveformCompact]}>
      {bars.map((height, index) => (
        <Animated.View key={`${height}-${index}`} style={[styles.waveBar, compact && styles.waveBarCompact, {
          backgroundColor: color,
          height: liveLevel ? Math.max(3, Math.round(height * (0.16 + Math.min(1, microphoneLevel || 0) * (0.72 + (index % 3) * 0.1)))) : height,
          opacity: liveLevel ? Math.min(1, 0.22 + (microphoneLevel || 0) * 1.6) : pulse.interpolate({ inputRange: [0, 1], outputRange: index % 2 ? [0.35, 0.9] : [0.85, 0.38] }),
        }]} />
      ))}
    </View>
  )
}

function ThinkingDots() {
  const pulse = usePulse(true, 900)
  return (
    <View style={styles.thinkingDots}>
      {[0, 1, 2].map((dot) => <Animated.View key={dot} style={[styles.thinkingDot, { opacity: pulse.interpolate({ inputRange: [0, 0.5, 1], outputRange: dot === 1 ? [0.35, 1, 0.35] : [0.8, 0.35, 0.8] }) }]} />)}
    </View>
  )
}

function usePulse(active: boolean, duration: number) {
  const pulse = useRef(new Animated.Value(0)).current
  useEffect(() => {
    if (!active) { pulse.stopAnimation(); pulse.setValue(0); return }
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
  stage: { backgroundColor: c.cream, flex: 1, minHeight: 640 },
  endConfirmBackdrop: { alignItems: 'center', backgroundColor: 'rgba(23,37,37,0.34)', bottom: 0, justifyContent: 'center', left: 0, padding: 28, position: 'absolute', right: 0, top: 0, zIndex: 20 },
  endConfirmCard: { backgroundColor: c.cream, borderColor: c.lineWarm, borderRadius: 24, borderWidth: 1, maxWidth: 540, padding: 28, width: '100%' },
  endConfirmTitle: { color: c.text, fontFamily: f.display, fontSize: 30, lineHeight: 38, textAlign: 'center' },
  endConfirmBody: { color: c.textSecondary, fontFamily: f.body, fontSize: 17, lineHeight: 25, marginTop: 9, textAlign: 'center' },
  endConfirmActions: { gap: 11, marginTop: 23 },
  endConfirmSecondary: { alignItems: 'center', borderColor: c.sageDeep, borderRadius: 25, borderWidth: 1.5, minHeight: 52, justifyContent: 'center', paddingHorizontal: 18 },
  endConfirmSecondaryText: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 16 },
  endConfirmPrimary: { alignItems: 'center', backgroundColor: c.coral, borderRadius: 25, minHeight: 52, justifyContent: 'center', paddingHorizontal: 18 },
  endConfirmPrimaryText: { color: c.white, fontFamily: f.bodyMedium, fontSize: 16 },
  topWash: { backgroundColor: 'rgba(230,215,194,0.30)', borderRadius: 260, height: 300, position: 'absolute', right: -160, top: -155, transform: [{ rotate: '-24deg' }], width: 480 },
  bottomWash: { backgroundColor: 'rgba(219,226,211,0.30)', borderRadius: 260, bottom: -210, height: 420, left: -240, position: 'absolute', transform: [{ rotate: '17deg' }], width: 650 },
  cornerLeafOne: { borderColor: 'rgba(102,126,104,0.25)', borderLeftWidth: 2, borderTopWidth: 2, borderTopLeftRadius: 180, height: 185, position: 'absolute', right: -80, top: -16, transform: [{ rotate: '24deg' }], width: 240 },
  cornerLeafTwo: { borderColor: 'rgba(102,126,104,0.22)', borderBottomWidth: 2, borderRightWidth: 2, borderBottomRightRadius: 180, bottom: -70, height: 220, left: -100, position: 'absolute', transform: [{ rotate: '20deg' }], width: 280 },
  header: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between', minHeight: 84, paddingHorizontal: 34, paddingTop: 23 },
  headerSide: { minWidth: 150 },
  headerSideRight: { alignItems: 'flex-end', minWidth: 170 },
  headerTime: { color: c.text, fontFamily: f.bodyMedium, fontSize: 24, fontVariant: ['tabular-nums'] },
  headerDate: { color: c.text, fontFamily: f.body, fontSize: 14, marginTop: 3 },
  headerWeather: { alignItems: 'center', flexDirection: 'row', gap: 8 },
  weatherValue: { color: c.text, fontFamily: f.body, fontSize: 17 },
  headerStatus: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 14, marginTop: 3 },
  brandLockup: { alignItems: 'center', flexDirection: 'row', gap: 9 },
  brandMark: { height: 38, position: 'relative', width: 42 },
  brandMarkSmall: { height: 22, width: 24 },
  brandPetal: { backgroundColor: c.sageDeep, borderBottomLeftRadius: 20, borderBottomRightRadius: 4, borderTopLeftRadius: 4, borderTopRightRadius: 20, height: 28, position: 'absolute', top: 8, width: 12 },
  brandPetalLeft: { left: 4, transform: [{ rotate: '-34deg' }] },
  brandPetalCenter: { backgroundColor: c.goldDeep, left: 15, top: 1, transform: [{ rotate: '45deg' }] },
  brandPetalRight: { backgroundColor: c.coral, right: 4, transform: [{ rotate: '124deg' }] },
  brandPetalSmall: { borderBottomLeftRadius: 11, borderTopRightRadius: 11, height: 16, top: 4, width: 7 },
  brandSpark: { backgroundColor: c.goldDeep, height: 7, left: 18, position: 'absolute', top: 14, transform: [{ rotate: '45deg' }], width: 7 },
  brandSparkSmall: { height: 4, left: 10, top: 7, width: 4 },
  brandText: { color: c.text, fontFamily: f.display, fontSize: 31, letterSpacing: -1 },
  homeScroll: { paddingBottom: 28 },
  homeHero: { alignItems: 'center', marginTop: 22, paddingHorizontal: 32 },
  avatarFrameLarge: { alignItems: 'center', height: 285, justifyContent: 'center', width: 285 },
  avatarRingLarge: { borderColor: c.sage, borderRadius: 142, borderWidth: 3, height: 285, position: 'absolute', width: 285 },
  avatarLarge: { borderRadius: 132, height: 264, width: 264 },
  ariaNameBadge: { alignItems: 'center', backgroundColor: c.white, borderColor: c.sage, borderRadius: 20, borderWidth: 1, bottom: -8, flexDirection: 'row', gap: 7, paddingHorizontal: 20, paddingVertical: 8, position: 'absolute' },
  ariaNameText: { color: c.sageDeep, fontFamily: f.display, fontSize: 24 },
  homeGreeting: { color: c.text, fontFamily: f.display, fontSize: 45, lineHeight: 56, marginTop: 34, maxWidth: '90%', textAlign: 'center' },
  homeBody: { color: c.text, fontFamily: f.body, fontSize: 21, lineHeight: 30, marginTop: 7, textAlign: 'center' },
  startCard: { alignItems: 'center', alignSelf: 'center', backgroundColor: c.glassOverlayStrong, borderColor: c.lineWarm, borderRadius: 25, borderWidth: 1, marginHorizontal: 35, marginTop: 28, maxWidth: 700, paddingHorizontal: 26, paddingVertical: 22, width: '88%' },
  pressed: { opacity: 0.86, transform: [{ scale: 0.995 }] },
  disabledCard: { opacity: 0.65 },
  readyOrb: { alignItems: 'center', backgroundColor: c.sageDeep, borderRadius: 44, elevation: 4, height: 88, justifyContent: 'center', shadowColor: c.sageDeep, shadowOpacity: 0.25, shadowRadius: 12, width: 88 },
  startTitle: { color: c.sageDeep, fontFamily: f.display, fontSize: 28, marginTop: 15, textAlign: 'center' },
  startSubtitle: { color: c.text, fontFamily: f.body, fontSize: 18, marginTop: 3, textAlign: 'center' },
  wakeNote: { color: c.coral, fontFamily: f.body, fontSize: 13, marginTop: 8, textAlign: 'center' },
  homeCardRow: { alignSelf: 'center', flexDirection: 'row', flexWrap: 'wrap', gap: 14, justifyContent: 'center', marginHorizontal: 35, marginTop: 22, maxWidth: 760, width: '88%' },
  homeCard: { alignItems: 'center', backgroundColor: c.glassOverlay, borderColor: c.lineWarm, borderRadius: 22, borderWidth: 1, flexDirection: 'row', flexGrow: 1, gap: 12, minHeight: 84, minWidth: 230, paddingHorizontal: 17, paddingVertical: 14 },
  messageCard: { alignItems: 'center', alignSelf: 'center', backgroundColor: c.glassOverlay, borderColor: 'rgba(201,109,66,0.28)', borderRadius: 26, borderWidth: 1, flexDirection: 'row', gap: 18, marginHorizontal: 35, marginTop: 22, maxWidth: 760, minHeight: 150, paddingHorizontal: 26, paddingVertical: 22, width: '88%' },
  homeCardIcon: { alignItems: 'center', backgroundColor: c.beige, borderRadius: 27, height: 54, justifyContent: 'center', width: 54 },
  messageCardIcon: { alignItems: 'center', backgroundColor: c.beige, borderRadius: 40, height: 80, justifyContent: 'center', width: 80 },
  messageIcon: { backgroundColor: 'rgba(231,210,180,0.65)' },
  homeCardCopy: { flex: 1 },
  homeCardTitle: { color: c.text, fontFamily: f.bodyMedium, fontSize: 17, lineHeight: 22 },
  homeCardLabel: { color: c.textSecondary, fontFamily: f.body, fontSize: 13, lineHeight: 18, marginTop: 2 },
  messageCardCopy: { flex: 1, justifyContent: 'center' },
  messageCardTitle: { color: c.text, fontFamily: f.bodyMedium, fontSize: 25, lineHeight: 31 },
  messageCardLabel: { color: c.textSecondary, fontFamily: f.body, fontSize: 18, lineHeight: 25, marginTop: 7 },
  homeFooterActions: { alignItems: 'center', flexDirection: 'row', flexWrap: 'wrap', gap: 12, justifyContent: 'center', marginHorizontal: 35, marginTop: 22 },
  footerAction: { alignItems: 'center', flexDirection: 'row', gap: 7, paddingHorizontal: 12, paddingVertical: 9 },
  footerActionText: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 13 },
  conversationScene: { flex: 1, paddingBottom: 21, paddingHorizontal: 34 },
  conversationContent: { alignItems: 'center', flex: 1, justifyContent: 'center', paddingBottom: 20, paddingTop: 4 },
  conversationTitle: { color: c.text, fontFamily: f.display, fontSize: 42, lineHeight: 53, marginTop: 18, maxWidth: 760, textAlign: 'center' },
  conversationSubtitle: { color: c.text, fontFamily: f.body, fontSize: 20, lineHeight: 29, marginTop: 8, maxWidth: 620, textAlign: 'center' },
  spinner: { marginTop: 20 },
  promptCard: { alignItems: 'center', alignSelf: 'stretch', backgroundColor: c.glassOverlay, borderColor: c.lineWarm, borderRadius: 25, borderWidth: 1, flexDirection: 'row', gap: 14, marginBottom: 27, maxWidth: 760, minHeight: 105, paddingHorizontal: 25, paddingVertical: 18 },
  promptQuote: { alignSelf: 'flex-start', height: 45, width: 33 },
  promptQuoteText: { color: c.sageDeep, fontFamily: f.display, fontSize: 70, lineHeight: 70 },
  promptLabel: { alignSelf: 'flex-start', color: c.textSecondary, fontFamily: f.bodyMedium, fontSize: 15, marginTop: 8 },
  promptScroll: { flex: 1, maxHeight: 100 },
  promptText: { color: c.text, fontFamily: f.display, fontSize: 25, lineHeight: 34 },
  listenOrb: { alignItems: 'center', backgroundColor: 'rgba(183,197,175,0.28)', borderColor: c.sage, borderRadius: 120, borderWidth: 2, height: 238, justifyContent: 'center', width: 238 },
  listenOrbHeard: { backgroundColor: 'rgba(79,112,103,0.16)', borderColor: c.sageDeep },
  speakingPill: { alignItems: 'center', backgroundColor: 'rgba(231,231,218,0.75)', borderColor: c.sage, borderRadius: 22, flexDirection: 'row', gap: 10, marginTop: 16, paddingHorizontal: 18, paddingVertical: 8 },
  speakingPillText: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 15 },
  speechScroll: { maxHeight: 185, maxWidth: 760, marginTop: 18 },
  speechText: { color: c.text, fontFamily: f.display, fontSize: 34, lineHeight: 43, textAlign: 'center' },
  waveform: { alignItems: 'center', flexDirection: 'row', gap: 4, height: 44, justifyContent: 'center', marginTop: 17 },
  waveformCompact: { height: 28, marginTop: 0 },
  waveBar: { borderRadius: 2, width: 4 },
  waveBarCompact: { width: 3 },
  conversationActions: { alignItems: 'center', flexDirection: 'row', flexWrap: 'wrap', gap: 13, justifyContent: 'center', marginTop: 22 },
  secondaryButton: { alignItems: 'center', backgroundColor: c.glassOverlayStrong, borderColor: c.sage, borderRadius: 25, borderWidth: 1.5, flexDirection: 'row', gap: 9, minHeight: 55, paddingHorizontal: 22 },
  secondaryButtonText: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 16 },
  stopButton: { alignItems: 'center', backgroundColor: c.coral, borderRadius: 25, flexDirection: 'row', gap: 9, minHeight: 55, paddingHorizontal: 27 },
  stopButtonText: { color: c.white, fontFamily: f.bodyMedium, fontSize: 16 },
  interruptButton: { alignItems: 'center', flexDirection: 'row', gap: 8, marginTop: 17, padding: 8 },
  interruptText: { color: c.sageDeep, fontFamily: f.body, fontSize: 15 },
  endHint: { alignItems: 'center', alignSelf: 'center', flexDirection: 'row', gap: 8, paddingHorizontal: 18, paddingVertical: 10 },
  endHintText: { color: c.textSecondary, fontFamily: f.body, fontSize: 14 },
  portraitFrame: { alignItems: 'center', justifyContent: 'center' },
  portraitRing: { backgroundColor: 'rgba(255,255,255,0.46)', borderWidth: 3, position: 'absolute' },
  portraitClip: { borderColor: c.sage, borderWidth: 2, overflow: 'hidden' },
  portraitImage: { height: '100%', width: '100%' },
  thinkingDots: { flexDirection: 'row', gap: 10, marginTop: 25 },
  thinkingDot: { backgroundColor: c.goldDeep, borderRadius: 4, height: 8, width: 8 },
  problemScene: { flex: 1, paddingHorizontal: 34 },
  problemContent: { alignItems: 'center', flex: 1, justifyContent: 'center', paddingBottom: 48 },
  problemIcon: { alignItems: 'center', backgroundColor: 'rgba(231,210,180,0.38)', borderColor: c.lineWarm, borderRadius: 55, borderWidth: 1, height: 110, justifyContent: 'center', width: 110 },
  problemIconError: { backgroundColor: 'rgba(201,109,66,0.10)', borderColor: 'rgba(201,109,66,0.28)' },
  problemTitle: { color: c.text, fontFamily: f.display, fontSize: 40, lineHeight: 51, marginTop: 27, maxWidth: 650, textAlign: 'center' },
  problemBody: { color: c.text, fontFamily: f.body, fontSize: 21, lineHeight: 31, marginTop: 15, maxWidth: 600, textAlign: 'center' },
  problemNote: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 16, marginTop: 15, textAlign: 'center' },
  primaryButton: { alignItems: 'center', backgroundColor: c.sageDeep, borderRadius: 30, marginTop: 29, minHeight: 58, justifyContent: 'center', paddingHorizontal: 32 },
  primaryButtonText: { color: c.white, fontFamily: f.bodyMedium, fontSize: 17 },
  problemDetail: { color: c.textSecondary, fontFamily: 'monospace', fontSize: 12, lineHeight: 17, marginTop: 27, maxWidth: 650, opacity: 0.75, textAlign: 'center' },
})
