import { router, useLocalSearchParams } from 'expo-router'
import { Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native'

import { mirrorColors as c, mirrorFonts as f } from '../src/theme/mirrorTheme'

const ARIA_AVATAR = require('../assets/images/aria-avatar.png')

type ClosingParams = {
  completed?: string | string[]
  date?: string | string[]
  language?: string | string[]
  patientName?: string | string[]
  sync?: string | string[]
  time?: string | string[]
}

function value(value: string | string[] | undefined): string {
  return Array.isArray(value) ? value[0] || '' : value || ''
}

export default function ConversationClosingScreen() {
  const params = useLocalSearchParams<ClosingParams>()
  const patientName = value(params.patientName).trim() || 'there'
  const completed = value(params.completed) !== 'false'
  const time = value(params.time) || ' '
  const date = value(params.date) || ' '
  const queued = value(params.sync) === 'queued'

  return (
    <View style={styles.shell}>
      <ScrollView contentContainerStyle={styles.scene} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <View style={styles.headerSide}>
            <Text style={styles.headerTime}>{time}</Text>
            <Text style={styles.headerDate}>{date}</Text>
          </View>
          <View style={styles.brandLockup}>
            <BrandMark />
            <Text style={styles.brandText}>Reflexion</Text>
          </View>
          <View style={styles.headerSideRight}>
            <Text style={styles.wifiIcon}>⌁</Text>
            <Text style={styles.headerStatus}>Mirror Ready</Text>
          </View>
        </View>

        <View style={styles.hero}>
          <View style={styles.portraitFrame}>
            <View style={styles.portraitRing} />
            <Image resizeMode="cover" source={ARIA_AVATAR} style={styles.portrait} />
            <View style={styles.ariaBadge}>
              <BrandMark small />
              <Text style={styles.ariaBadgeText}>Aria</Text>
            </View>
          </View>
          <Text adjustsFontSizeToFit numberOfLines={1} style={styles.title}>Thank you, {patientName}.</Text>
          <Text style={styles.subtitle}>
            {completed ? 'Your check-in is complete for today.' : 'Your progress has been saved for another time.'}
          </Text>

          <Pressable
            accessibilityHint="Return to the mirror home screen"
            accessibilityRole="button"
            onPress={() => router.replace('/conversation')}
            style={({ pressed }) => [styles.returnButton, pressed && styles.pressed]}
          >
            <Text style={styles.returnButtonText}>Return home</Text>
          </Pressable>

          <View style={styles.divider}>
            <View style={styles.dividerLine} />
            <Text style={styles.dividerStar}>✦</Text>
            <View style={styles.dividerLine} />
          </View>
          <Text style={styles.encouragement}>
            {completed
              ? 'You showed up for yourself today.\nThat’s something to be proud of.'
              : 'We can continue together another time.'}
          </Text>
          {queued ? <Text style={styles.syncNote}>Your conversation is syncing securely.</Text> : null}
        </View>
      </ScrollView>
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

const styles = StyleSheet.create({
  shell: { backgroundColor: c.cream, flex: 1, overflow: 'hidden' },
  scene: { minHeight: '100%', paddingBottom: 55 },
  header: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between', minHeight: 110, paddingHorizontal: 34, paddingTop: 24 },
  headerSide: { minWidth: 170 },
  headerSideRight: { alignItems: 'flex-end', minWidth: 170 },
  headerTime: { color: c.text, fontFamily: f.bodyMedium, fontSize: 25, fontVariant: ['tabular-nums'] },
  headerDate: { color: c.text, fontFamily: f.body, fontSize: 15, marginTop: 3 },
  wifiIcon: { color: c.text, fontFamily: f.body, fontSize: 36, lineHeight: 30, transform: [{ rotate: '90deg' }] },
  headerStatus: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 16, marginTop: 5 },
  brandLockup: { alignItems: 'center', gap: 5 },
  brandText: { color: c.text, fontFamily: f.display, fontSize: 40, letterSpacing: -1.2 },
  brandMark: { height: 45, position: 'relative', width: 50 },
  brandMarkSmall: { height: 22, width: 24 },
  brandPetal: { backgroundColor: c.sageDeep, borderBottomLeftRadius: 20, borderBottomRightRadius: 4, borderTopLeftRadius: 4, borderTopRightRadius: 20, height: 32, position: 'absolute', top: 10, width: 14 },
  brandPetalLeft: { left: 4, transform: [{ rotate: '-34deg' }] },
  brandPetalCenter: { backgroundColor: c.goldDeep, left: 18, top: 1, transform: [{ rotate: '45deg' }] },
  brandPetalRight: { backgroundColor: c.coral, right: 4, transform: [{ rotate: '124deg' }] },
  brandPetalSmall: { borderBottomLeftRadius: 11, borderTopRightRadius: 11, height: 16, top: 4, width: 7 },
  brandSpark: { backgroundColor: c.goldDeep, height: 7, left: 21, position: 'absolute', top: 17, transform: [{ rotate: '45deg' }], width: 7 },
  brandSparkSmall: { height: 4, left: 10, top: 7, width: 4 },
  hero: { alignItems: 'center', paddingHorizontal: 30, paddingTop: 55 },
  portraitFrame: { alignItems: 'center', height: 480, justifyContent: 'center', width: 480 },
  portraitRing: { borderColor: c.sage, borderRadius: 240, borderWidth: 4, height: 480, position: 'absolute', width: 480 },
  portrait: { borderRadius: 222, height: 444, width: 444 },
  ariaBadge: { alignItems: 'center', backgroundColor: c.white, borderColor: c.sage, borderRadius: 24, borderWidth: 2, bottom: -10, flexDirection: 'row', gap: 10, paddingHorizontal: 25, paddingVertical: 9, position: 'absolute' },
  ariaBadgeText: { color: c.sageDeep, fontFamily: f.display, fontSize: 32 },
  title: { color: c.text, fontFamily: f.display, fontSize: 58, lineHeight: 70, marginTop: 76, maxWidth: '96%', textAlign: 'center' },
  subtitle: { color: c.text, fontFamily: f.body, fontSize: 26, lineHeight: 37, marginTop: 12, maxWidth: 800, textAlign: 'center' },
  returnButton: { alignItems: 'center', alignSelf: 'stretch', backgroundColor: c.sageDeep, borderRadius: 42, marginTop: 65, minHeight: 84, justifyContent: 'center', maxWidth: 760 },
  returnButtonText: { color: c.white, fontFamily: f.body, fontSize: 27 },
  pressed: { opacity: 0.86, transform: [{ scale: 0.995 }] },
  divider: { alignItems: 'center', flexDirection: 'row', gap: 18, marginTop: 65, width: 450 },
  dividerLine: { backgroundColor: 'rgba(200,121,67,0.38)', flex: 1, height: 1 },
  dividerStar: { color: c.goldDeep, fontSize: 34 },
  encouragement: { color: c.text, fontFamily: f.body, fontSize: 24, lineHeight: 35, marginTop: 34, textAlign: 'center' },
  syncNote: { color: c.textSecondary, fontFamily: f.body, fontSize: 14, marginTop: 22 },
})
