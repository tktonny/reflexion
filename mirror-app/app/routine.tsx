import { router, useLocalSearchParams } from 'expo-router'
import { useMemo, useState } from 'react'
import { StyleSheet, Text, View } from 'react-native'

import { MirrorCard, MirrorPage, OutlineButton, PrimaryButton } from '../src/components/mirror/MirrorChrome'
import { MirrorIcon } from '../src/components/mirror/MirrorIcon'
import { dataOrThrow } from '../src/api/devicePairing'
import { deviceFetch, randomIdempotencyKey } from '../src/storage/deviceCredentials'
import { mirrorColors as c, mirrorFonts as f } from '../src/theme/mirrorTheme'
import { isDemoRoute } from '../src/demo/demoConfig'
import { updateDemoMirrorState, type DemoRoutineResponse } from '../src/demo/demoRepository'

type ResponseChoice = 'taken' | 'snoozed' | 'skipped' | 'unknown'

export default function RoutineScreen() {
  const params = useLocalSearchParams<{ demo?: string; occurrenceId?: string; displayText?: string; scheduledAt?: string }>()
  const demo = isDemoRoute(params.demo)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState('')
  const title = String(params.displayText || 'Your routine reminder').trim() || 'Your routine reminder'
  const time = useMemo(() => {
    const parsed = new Date(String(params.scheduledAt || ''))
    return Number.isNaN(parsed.getTime()) ? '' : new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(parsed)
  }, [params.scheduledAt])

  async function respond(status: ResponseChoice, message: string) {
    if (saving) return
    setSaving(true)
    try {
      if (demo) {
        const routineResponse: DemoRoutineResponse = status === 'taken' ? 'complete' : status === 'snoozed' ? 'deferred' : 'declined'
        await updateDemoMirrorState({ routineResponse })
        setSaved(`Demo only — ${message}`)
        setTimeout(() => router.replace('/demo'), 900)
        return
      }
      if (params.occurrenceId) {
        const response = await deviceFetch(`/api/v1/reminder-occurrences/${encodeURIComponent(String(params.occurrenceId))}/responses`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Idempotency-Key': randomIdempotencyKey() },
          body: JSON.stringify({ status, respondedAt: new Date().toISOString(), note: 'Reported by the loved one on the Mirror.' }),
        })
        await dataOrThrow(response)
      }
      setSaved(message)
      setTimeout(() => router.replace('/conversation'), 900)
    } catch {
      setSaved('We could not save that response. Please try again.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <MirrorPage headerStatus="Mirror ready" onHelp={() => router.push(demo ? '/status?demo=1' : '/status')}>
      <View style={styles.content}>
        <Text style={styles.eyebrow}>ROUTINE REMINDER</Text>
        <Text style={styles.title}>A gentle reminder</Text>
        {time ? <Text style={styles.time}>Scheduled for {time}</Text> : null}
        <MirrorCard accent>
          <View style={styles.iconCircle}><MirrorIcon name="calendar-outline" size={56} color={c.sageDeep} /></View>
          <Text style={styles.reminderTitle}>{title}</Text>
          <Text style={styles.reminderBody}>You can tell me what you would like to do.</Text>
        </MirrorCard>
        {saved ? <Text style={styles.saved}>{saved}</Text> : null}
        <PrimaryButton label="Reported complete" icon="checkmark-circle-outline" onPress={() => void respond('taken', 'Thanks — I recorded that you reported it complete.')} />
        <OutlineButton label="Remind me later" icon="time-outline" onPress={() => void respond('snoozed', 'Okay. I will remind you later.')} />
        <OutlineButton label="Not today" icon="close-circle-outline" onPress={() => void respond('skipped', 'Okay. I will leave it for today.')} />
        <Text style={styles.note}>Your response is a report from you. Reflexion does not independently verify it.</Text>
      </View>
    </MirrorPage>
  )
}

const styles = StyleSheet.create({
  content: { alignItems: 'center', maxWidth: 760, width: '100%' },
  eyebrow: { color: c.goldDeep, fontFamily: f.bodyMedium, fontSize: 13, letterSpacing: 2, marginBottom: 7 },
  title: { color: c.text, fontFamily: f.display, fontSize: 48, lineHeight: 58, textAlign: 'center' },
  time: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 18, marginTop: 8 },
  iconCircle: { alignItems: 'center', alignSelf: 'center', backgroundColor: 'rgba(183,197,175,0.30)', borderRadius: 50, height: 100, justifyContent: 'center', width: 100 },
  reminderTitle: { color: c.text, fontFamily: f.display, fontSize: 34, lineHeight: 43, marginTop: 20, textAlign: 'center' },
  reminderBody: { color: c.textSecondary, fontFamily: f.body, fontSize: 19, marginTop: 8, textAlign: 'center' },
  saved: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 15, marginBottom: 4, textAlign: 'center' },
  note: { color: c.textSecondary, fontFamily: f.body, fontSize: 13, lineHeight: 19, marginTop: 17, maxWidth: 620, textAlign: 'center' },
})
