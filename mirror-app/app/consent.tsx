import { router } from 'expo-router'
import { useEffect, useState } from 'react'
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'

import { dataOrThrow, type DeviceConfiguration } from '../src/api/devicePairing'
import { recordMirrorConsent, type MirrorConsentStatus } from '../src/api/deviceConsent'
import { deviceFetch, getDeviceCredential } from '../src/storage/deviceCredentials'
import { mirrorColors as c, mirrorFonts as f } from '../src/theme/mirrorTheme'

type PublicStatus = NonNullable<NonNullable<DeviceConfiguration['patient']>['consent']>['status']

export default function ConsentScreen() {
  const [patientName, setPatientName] = useState('your loved one')
  const [status, setStatus] = useState<PublicStatus>('pending')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const load = async () => {
    setError('')
    try {
      const credential = await getDeviceCredential()
      if (!credential) throw new Error('device_not_paired')
      const response = await deviceFetch(`/api/v1/devices/${encodeURIComponent(credential.deviceId)}/configuration`)
      const configuration = await dataOrThrow<DeviceConfiguration>(response)
      if (!configuration.patient) throw new Error('paired_patient_configuration_missing')
      setPatientName(configuration.patient.displayName || 'your loved one')
      const next = configuration.patient.consent?.status || 'pending'
      setStatus(next)
      if (next === 'accepted') router.replace('/conversation')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not load consent status.')
    }
  }

  useEffect(() => { void load() }, [])

  const choose = async (next: Extract<MirrorConsentStatus, 'granted' | 'declined'>) => {
    setBusy(true)
    setError('')
    try {
      await recordMirrorConsent(next)
      setStatus(next === 'granted' ? 'accepted' : 'declined')
      if (next === 'granted') router.replace('/conversation')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not save your choice. Please try again.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.eyebrow}>REFLEXION · CARE CONNECTED</Text>
        <Text accessibilityRole="header" style={styles.title}>A quick choice before we begin</Text>
        <Text style={styles.copy}>Reflexion can use conversations and routine responses to keep {patientName} connected with their caregiver. This is not a diagnosis or an emergency service.</Text>
        <View style={styles.card}>
          <Text style={styles.cardTitle}>You are in control</Text>
          <Text style={styles.copy}>You can change or withdraw this choice later with your care team. Nothing is shared for optional research unless that is chosen separately.</Text>
        </View>
        {status === 'declined' ? <View style={styles.notice}><Text style={styles.noticeTitle}>Consent is currently declined</Text><Text style={styles.copy}>Daily check-ins will stay paused. Your caregiver can help you review this choice.</Text></View> : null}
        {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
        {busy ? <ActivityIndicator color={c.linen} /> : <>
          <Pressable accessibilityRole="button" accessibilityLabel="Agree and continue" onPress={() => void choose('granted')} style={styles.primary}><Text style={styles.primaryText}>Agree and continue</Text></Pressable>
          <Pressable accessibilityRole="button" accessibilityLabel="Decline for now" onPress={() => void choose('declined')} style={styles.secondary}><Text style={styles.secondaryText}>Decline for now</Text></Pressable>
        </>}
      </ScrollView>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  safe: { backgroundColor: c.cream, flex: 1 },
  content: { flexGrow: 1, justifyContent: 'center', padding: 42, gap: 20 },
  eyebrow: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 12, letterSpacing: 1.5 },
  title: { color: c.ink, fontFamily: f.display, fontSize: 38, lineHeight: 44 },
  copy: { color: c.textSecondary, fontFamily: f.body, fontSize: 17, lineHeight: 25 },
  card: { backgroundColor: 'rgba(255,255,255,0.72)', borderColor: c.line, borderRadius: 22, borderWidth: 1, gap: 8, padding: 22 },
  cardTitle: { color: c.ink, fontFamily: f.bodyMedium, fontSize: 19 },
  notice: { backgroundColor: 'rgba(245,235,208,0.86)', borderColor: c.line, borderRadius: 18, gap: 8, padding: 18 },
  noticeTitle: { color: c.ink, fontFamily: f.bodyMedium, fontSize: 18 },
  primary: { alignItems: 'center', backgroundColor: c.ink, borderRadius: 28, paddingVertical: 17 },
  primaryText: { color: c.cream, fontFamily: f.bodyMedium, fontSize: 17 },
  secondary: { alignItems: 'center', borderColor: c.sageDeep, borderRadius: 28, borderWidth: 1, paddingVertical: 16 },
  secondaryText: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 16 },
  error: { color: '#9D3D34', fontFamily: f.bodyMedium, fontSize: 15, lineHeight: 21 },
})
