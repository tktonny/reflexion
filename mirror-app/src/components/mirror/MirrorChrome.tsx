import { useMemo } from 'react'
import type { ReactNode } from 'react'
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'

import { mirrorColors as c, mirrorFonts as f } from '../../theme/mirrorTheme'
import { MirrorIcon, type MirrorIconName } from './MirrorIcon'

export function MirrorBrand({ lockup = true }: { lockup?: boolean }) {
  return (
    <View style={styles.brandLockup}>
      <View style={styles.brandMark}>
        <View style={[styles.brandPetal, styles.brandPetalLeft]} />
        <View style={[styles.brandPetal, styles.brandPetalCenter]} />
        <View style={[styles.brandPetal, styles.brandPetalRight]} />
        <View style={styles.brandSpark} />
      </View>
      {lockup ? <Text style={styles.brandText}>Reflexion</Text> : null}
    </View>
  )
}

export function MirrorAtmosphere() {
  return (
    <View pointerEvents="none" style={StyleSheet.absoluteFill}>
      <View style={styles.topWash} />
      <View style={styles.bottomWash} />
      <View style={styles.cornerLeafOne} />
      <View style={styles.cornerLeafTwo} />
    </View>
  )
}

export function MirrorPage({
  children,
  headerStatus = 'Mirror ready',
  onHelp,
  scroll = true,
}: {
  children: ReactNode
  headerStatus?: string
  onHelp?: () => void
  scroll?: boolean
}) {
  const { time, date } = useMemo(() => {
    const now = new Date()
    return {
      time: new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(now),
      date: new Intl.DateTimeFormat(undefined, { weekday: 'long', month: 'long', day: 'numeric' }).format(now),
    }
  }, [])
  const content = scroll ? <ScrollView contentContainerStyle={styles.pageContent} showsVerticalScrollIndicator={false}>{children}</ScrollView> : <View style={styles.pageContent}>{children}</View>
  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.page}>
        <MirrorAtmosphere />
        <View style={styles.header}>
          <View style={styles.headerSide}>
            <Text style={styles.headerTime}>{time}</Text>
            <Text style={styles.headerDate}>{date}</Text>
          </View>
          <MirrorBrand />
          <Pressable accessibilityRole="button" accessibilityLabel="Open device status and help" onPress={onHelp} style={styles.headerSideRight}>
            <MirrorIcon name="wifi" size={25} color={c.text} />
            <Text style={styles.headerStatus}>{headerStatus}</Text>
          </Pressable>
        </View>
        {content}
      </View>
    </SafeAreaView>
  )
}

export function PageHeading({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <View style={styles.heading}>
      <Text style={styles.pageTitle}>{title}</Text>
      {subtitle ? <Text style={styles.pageSubtitle}>{subtitle}</Text> : null}
    </View>
  )
}

export function MirrorCard({ children, accent = false }: { children: ReactNode; accent?: boolean }) {
  return <View style={[styles.card, accent && styles.cardAccent]}>{children}</View>
}

export function PrimaryButton({ label, onPress, icon }: { label: string; onPress?: () => void; icon?: MirrorIconName }) {
  return (
    <Pressable accessibilityRole="button" disabled={!onPress} onPress={onPress} style={[styles.primaryButton, !onPress && styles.disabled]}>
      {icon ? <MirrorIcon name={icon} size={25} color={c.white} /> : null}
      <Text style={styles.primaryButtonText}>{label}</Text>
    </Pressable>
  )
}

export function OutlineButton({ label, onPress, icon }: { label: string; onPress?: () => void; icon?: MirrorIconName }) {
  return (
    <Pressable accessibilityRole="button" disabled={!onPress} onPress={onPress} style={[styles.outlineButton, !onPress && styles.disabled]}>
      {icon ? <MirrorIcon name={icon} size={24} color={c.sageDeep} /> : null}
      <Text style={styles.outlineButtonText}>{label}</Text>
    </Pressable>
  )
}

export function StatusRow({ icon, label, value, state = 'good', onPress }: { icon: MirrorIconName; label: string; value: string; state?: 'good' | 'waiting' | 'error'; onPress?: () => void }) {
  const content = (
    <View style={styles.statusRow}>
      <View style={styles.statusIcon}><MirrorIcon name={icon} size={27} color={state === 'error' ? c.coral : c.sageDeep} /></View>
      <Text style={styles.statusLabel}>{label}</Text>
      <View style={[styles.statusBadge, state === 'error' && styles.statusBadgeError, state === 'waiting' && styles.statusBadgeWaiting]}>
        <MirrorIcon name={state === 'good' ? 'checkmark-circle' : state === 'waiting' ? 'time-outline' : 'close-circle'} size={21} color={state === 'error' ? c.coral : state === 'waiting' ? c.goldDeep : c.sageDeep} />
        <Text style={[styles.statusValue, state === 'error' && styles.statusValueError]}>{value}</Text>
      </View>
      {onPress ? <MirrorIcon name="chevron-forward" size={23} color={c.textSecondary} /> : null}
    </View>
  )
  return onPress ? <Pressable accessibilityRole="button" onPress={onPress}>{content}</Pressable> : content
}

const styles = StyleSheet.create({
  safeArea: { backgroundColor: c.cream, flex: 1 },
  page: { backgroundColor: c.cream, flex: 1, overflow: 'hidden' },
  topWash: { backgroundColor: 'rgba(230,215,194,0.30)', borderRadius: 260, height: 300, position: 'absolute', right: -160, top: -155, transform: [{ rotate: '-24deg' }], width: 480 },
  bottomWash: { backgroundColor: 'rgba(219,226,211,0.30)', borderRadius: 260, bottom: -210, height: 420, left: -240, position: 'absolute', transform: [{ rotate: '17deg' }], width: 650 },
  cornerLeafOne: { borderColor: 'rgba(102,126,104,0.25)', borderLeftWidth: 2, borderTopWidth: 2, borderTopLeftRadius: 180, height: 185, position: 'absolute', right: -80, top: -16, transform: [{ rotate: '24deg' }], width: 240 },
  cornerLeafTwo: { borderColor: 'rgba(102,126,104,0.22)', borderBottomWidth: 2, borderRightWidth: 2, borderBottomRightRadius: 180, bottom: -70, height: 220, left: -100, position: 'absolute', transform: [{ rotate: '20deg' }], width: 280 },
  header: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between', minHeight: 84, paddingHorizontal: 34, paddingTop: 23 },
  headerSide: { minWidth: 150 },
  headerSideRight: { alignItems: 'flex-end', minWidth: 170 },
  headerTime: { color: c.text, fontFamily: f.bodyMedium, fontSize: 23, fontVariant: ['tabular-nums'] },
  headerDate: { color: c.text, fontFamily: f.body, fontSize: 14, marginTop: 3 },
  headerStatus: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 14, marginTop: 3 },
  brandLockup: { alignItems: 'center', flexDirection: 'row', gap: 9 },
  brandMark: { height: 38, position: 'relative', width: 42 },
  brandPetal: { backgroundColor: c.sageDeep, borderBottomLeftRadius: 20, borderBottomRightRadius: 4, borderTopLeftRadius: 4, borderTopRightRadius: 20, height: 28, position: 'absolute', top: 8, width: 12 },
  brandPetalLeft: { left: 4, transform: [{ rotate: '-34deg' }] },
  brandPetalCenter: { backgroundColor: c.goldDeep, left: 15, top: 1, transform: [{ rotate: '45deg' }] },
  brandPetalRight: { backgroundColor: c.coral, right: 4, transform: [{ rotate: '124deg' }] },
  brandSpark: { backgroundColor: c.goldDeep, height: 7, left: 18, position: 'absolute', top: 14, transform: [{ rotate: '45deg' }], width: 7 },
  brandText: { color: c.text, fontFamily: f.display, fontSize: 31, letterSpacing: -1 },
  pageContent: { alignItems: 'center', flexGrow: 1, paddingBottom: 42, paddingHorizontal: 34, paddingTop: 24 },
  heading: { alignItems: 'center', marginBottom: 25, maxWidth: 820 },
  pageTitle: { color: c.text, fontFamily: f.display, fontSize: 47, lineHeight: 58, textAlign: 'center' },
  pageSubtitle: { color: c.text, fontFamily: f.body, fontSize: 20, lineHeight: 30, marginTop: 7, textAlign: 'center' },
  card: { backgroundColor: c.glassOverlay, borderColor: c.lineWarm, borderRadius: 24, borderWidth: 1, marginBottom: 17, maxWidth: 850, padding: 22, width: '100%' },
  cardAccent: { borderColor: c.sageDeep, borderWidth: 1.5 },
  primaryButton: { alignItems: 'center', backgroundColor: c.sageDeep, borderRadius: 30, flexDirection: 'row', gap: 11, justifyContent: 'center', marginTop: 8, minHeight: 58, paddingHorizontal: 28, width: '100%' },
  primaryButtonText: { color: c.white, fontFamily: f.bodyMedium, fontSize: 18 },
  outlineButton: { alignItems: 'center', borderColor: c.sageDeep, borderRadius: 30, borderWidth: 1.5, flexDirection: 'row', gap: 10, justifyContent: 'center', marginTop: 10, minHeight: 56, paddingHorizontal: 26, width: '100%' },
  outlineButtonText: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 17 },
  disabled: { opacity: 0.5 },
  statusRow: { alignItems: 'center', borderBottomColor: c.lineWarm, borderBottomWidth: StyleSheet.hairlineWidth, flexDirection: 'row', gap: 15, minHeight: 84, paddingVertical: 13 },
  statusIcon: { alignItems: 'center', backgroundColor: c.beige, borderRadius: 27, height: 54, justifyContent: 'center', width: 54 },
  statusLabel: { color: c.text, flex: 1, fontFamily: f.bodyMedium, fontSize: 20 },
  statusBadge: { alignItems: 'center', backgroundColor: 'rgba(183,197,175,0.25)', borderRadius: 16, flexDirection: 'row', gap: 6, paddingHorizontal: 12, paddingVertical: 8 },
  statusBadgeError: { backgroundColor: 'rgba(201,109,66,0.12)' },
  statusBadgeWaiting: { backgroundColor: 'rgba(231,210,180,0.38)' },
  statusValue: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 15 },
  statusValueError: { color: c.coral },
})

export const mirrorChromeStyles = styles
