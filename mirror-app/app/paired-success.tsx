import { router, useLocalSearchParams } from 'expo-router'
import { useEffect } from 'react'
import { Image, StyleSheet, Text, View } from 'react-native'

import { MirrorCard, MirrorPage, PrimaryButton } from '../src/components/mirror/MirrorChrome'
import { mirrorColors as c, mirrorFonts as f } from '../src/theme/mirrorTheme'
import { isDemoRoute } from '../src/demo/demoConfig'

const ARIA_AVATAR = require('../assets/images/aria-avatar.png')

export default function PairedSuccessScreen({ demoName }: { demoName?: string } = {}) {
  const { demo, name } = useLocalSearchParams<{ demo?: string; name?: string }>()
  const demoScreen = Boolean(demoName) || isDemoRoute(demo)
  const patientName = String(demoName || name || 'your loved one').trim() || 'your loved one'

  useEffect(() => {
    const timer = setTimeout(() => router.replace(demoScreen ? '/demo' : '/conversation'), 4200)
    return () => clearTimeout(timer)
  }, [demoScreen])

  return (
    <MirrorPage headerStatus="Mirror ready" scroll={false}>
      <View style={styles.content}>
        <View style={styles.checkCircle}><Text style={styles.checkmark}>✓</Text></View>
        <Text style={styles.title}>Device paired successfully</Text>
        <Text style={styles.subtitle}>Ready for {patientName} to use.</Text>
        <MirrorCard>
          <Text style={styles.cardText}>Your settings have been applied successfully.</Text>
          <Image resizeMode="cover" source={ARIA_AVATAR} style={styles.avatar} />
        </MirrorCard>
        <PrimaryButton label="Go to home" icon="arrow-forward" onPress={() => router.replace(demoScreen ? '/demo' : '/conversation')} />
        <Text style={styles.note}>Returning to your home screen…</Text>
      </View>
    </MirrorPage>
  )
}

const styles = StyleSheet.create({
  content: { alignItems: 'center', flex: 1, justifyContent: 'center', maxWidth: 760, paddingHorizontal: 24, width: '100%' },
  checkCircle: { alignItems: 'center', backgroundColor: 'rgba(183,197,175,0.32)', borderColor: c.sage, borderRadius: 105, borderWidth: 3, height: 160, justifyContent: 'center', width: 160 },
  checkmark: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 108, lineHeight: 125 },
  title: { color: c.text, fontFamily: f.display, fontSize: 47, lineHeight: 58, marginTop: 32, textAlign: 'center' },
  subtitle: { color: c.sageDeep, fontFamily: f.body, fontSize: 23, marginTop: 8, textAlign: 'center' },
  cardText: { color: c.text, fontFamily: f.body, fontSize: 19, lineHeight: 29, textAlign: 'center' },
  avatar: { borderColor: c.sage, borderRadius: 70, borderWidth: 2, height: 125, marginTop: 20, width: 125 },
  note: { color: c.textSecondary, fontFamily: f.body, fontSize: 14, marginTop: 14 },
})
