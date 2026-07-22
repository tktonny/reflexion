import { Ionicons } from '@expo/vector-icons'
import { router, useLocalSearchParams } from 'expo-router'
import { useEffect } from 'react'
import { Pressable, StyleSheet, Text, View } from 'react-native'

import { mirrorColors as c, mirrorFonts as f } from '../src/theme/mirrorTheme'

export default function ConversationClosingScreen() {
  const params = useLocalSearchParams<{ nurseName?: string; sync?: string }>()
  const nurseName = typeof params.nurseName === 'string' && params.nurseName.trim()
    ? params.nurseName.trim()
    : 'your caregiver'
  const queued = params.sync === 'queued'

  useEffect(() => {
    const returnToHome = setTimeout(() => router.replace('/conversation'), 9000)
    return () => clearTimeout(returnToHome)
  }, [])

  return (
    <View style={styles.shell}>
      <View pointerEvents="none" style={styles.reflection} />
      <Pressable
        accessibilityHint="Return to the mirror home screen"
        accessibilityRole="button"
        onPress={() => router.replace('/conversation')}
        style={styles.scene}
      >
        <View style={styles.checkCircle}>
          <Ionicons name="checkmark" size={42} color={c.ink} />
        </View>
        <Text style={styles.eyebrow}>CONVERSATION COMPLETE</Text>
        <Text style={styles.title}>Thank you for chatting with me.</Text>
        <Text style={styles.body}>I’ll let {nurseName} know you checked in today.</Text>
        <Text style={styles.goodbye}>Have a lovely day.</Text>
        <View style={styles.syncRow}>
          <View style={[styles.syncDot, queued && styles.syncDotQueued]} />
          <Text style={styles.syncText}>
            {queued ? 'Saved safely — will sync when connection returns' : 'Saved securely'}
          </Text>
        </View>
        <Text style={styles.returnHint}>Returning to your home screen…</Text>
      </Pressable>
    </View>
  )
}

const styles = StyleSheet.create({
  shell: { backgroundColor: c.cream, flex: 1, overflow: 'hidden' },
  reflection: { backgroundColor: 'rgba(231,207,166,0.22)', borderRadius: 300, height: 620, position: 'absolute', right: -310, top: -250, transform: [{ rotate: '-20deg' }], width: 510 },
  scene: { alignItems: 'center', flex: 1, justifyContent: 'center', paddingHorizontal: 38 },
  checkCircle: { alignItems: 'center', backgroundColor: c.sage, borderRadius: 48, height: 96, justifyContent: 'center', marginBottom: 32, shadowColor: c.sageDeep, shadowOpacity: 0.18, shadowRadius: 18, width: 96 },
  eyebrow: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 13, letterSpacing: 2 },
  title: { color: c.text, fontFamily: f.display, fontSize: 39, lineHeight: 50, marginTop: 20, maxWidth: 520, textAlign: 'center' },
  body: { color: c.textSecondary, fontFamily: f.body, fontSize: 22, lineHeight: 33, marginTop: 22, maxWidth: 500, textAlign: 'center' },
  goodbye: { color: c.text, fontFamily: f.display, fontSize: 28, marginTop: 17 },
  syncRow: { alignItems: 'center', backgroundColor: 'rgba(255,255,255,0.62)', borderColor: c.lineWarm, borderRadius: 22, borderWidth: 1, flexDirection: 'row', gap: 9, marginTop: 42, paddingHorizontal: 15, paddingVertical: 10 },
  syncDot: { backgroundColor: c.sageDeep, borderRadius: 4, height: 7, width: 7 },
  syncDotQueued: { backgroundColor: c.amber },
  syncText: { color: c.textSecondary, fontFamily: f.body, fontSize: 13 },
  returnHint: { bottom: 44, color: c.textSecondary, fontFamily: f.body, fontSize: 15, opacity: 0.8, position: 'absolute' },
})
