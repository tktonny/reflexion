import { router, useLocalSearchParams } from 'expo-router'
import { useEffect, useState } from 'react'
import { StyleSheet, Text, View } from 'react-native'

import { dataOrThrow, type DeviceConfiguration } from '../src/api/devicePairing'
import { deviceFetch, getDeviceCredential } from '../src/storage/deviceCredentials'
import { loadJson } from '../src/storage/mirrorStorage'
import { NURSE_PATIENT_CONFIG_STORAGE_KEY } from '../src/constants/nursePatientConfig'
import { MirrorCard, MirrorPage, OutlineButton, PageHeading } from '../src/components/mirror/MirrorChrome'
import { MirrorIcon } from '../src/components/mirror/MirrorIcon'
import { mirrorColors as c, mirrorFonts as f } from '../src/theme/mirrorTheme'
import { isDemoRoute } from '../src/demo/demoConfig'
import { getDemoMirrorState } from '../src/demo/demoRepository'

type ResearchState = 'Not invited' | 'Invitation pending' | 'Consented' | 'Declined' | 'Withdrawn' | 'Study closed'

export default function ResearchScreen() {
  const params = useLocalSearchParams<{ demo?: string }>()
  const demo = isDemoRoute(params.demo)
  const [state, setState] = useState<ResearchState>('Not invited')
  useEffect(() => {
    let mounted = true
    async function load() {
      if (demo) {
        const fixture = await getDemoMirrorState()
        if (mounted) setState(fixture.research)
        return
      }
      const cached = await loadJson<DeviceConfiguration>(NURSE_PATIENT_CONFIG_STORAGE_KEY)
      let configuration = cached
      try {
        const credential = await getDeviceCredential()
        if (credential) {
          const response = await deviceFetch(`/api/v1/devices/${encodeURIComponent(credential.deviceId)}/configuration`)
          configuration = await dataOrThrow<DeviceConfiguration>(response)
        }
      } catch { /* cached configuration remains safe to display */ }
      if (mounted) setState(resolveResearch(configuration))
    }
    void load()
    return () => { mounted = false }
  }, [demo])

  return (
    <MirrorPage headerStatus="Mirror ready" onHelp={() => router.push(demo ? '/status?demo=1' : '/status')}>
      <PageHeading title="Research participation" subtitle="Optional and separate from everyday Reflexion use." />
      <MirrorCard accent>
        <View style={styles.heroRow}>
          <View style={styles.clipboard}><MirrorIcon name="clipboard-outline" size={54} color={c.sageDeep} /></View>
          <View style={styles.heroCopy}>
            <Text style={styles.heroTitle}>{state}</Text>
            <Text style={styles.heroBody}>{researchDescription(state)}</Text>
          </View>
        </View>
      </MirrorCard>
      <MirrorCard>
        <Text style={styles.cardTitle}>Help shape the future of healthy aging.</Text>
        <Text style={styles.cardBody}>Research participation is optional. Choosing not to join will not affect normal mirror use.</Text>
      </MirrorCard>
      <Text style={styles.protocol}>{demo ? 'Demo only — study state is local sample data.' : 'Research details are shown only when an approved study requires them for this participant.'}</Text>
      <OutlineButton label="Back to consent & control" icon="chevron-back" onPress={() => router.replace(demo ? '/consent?demo=1' : '/consent')} />
    </MirrorPage>
  )
}

function resolveResearch(configuration: DeviceConfiguration | null): ResearchState {
  const value = configuration?.consent?.research || configuration?.desired?.researchState
  switch (String(value).toLowerCase()) {
    case 'invitation_pending': return 'Invitation pending'
    case 'consented': return 'Consented'
    case 'declined': return 'Declined'
    case 'withdrawn': return 'Withdrawn'
    case 'study_closed': return 'Study closed'
    default: return 'Not invited'
  }
}

function researchDescription(state: ResearchState) {
  switch (state) {
    case 'Invitation pending': return 'Your caregiver can review the study information before you decide.'
    case 'Consented': return 'You have chosen to participate in the approved study.'
    case 'Declined': return 'You have chosen not to participate. Normal Reflexion use continues.'
    case 'Withdrawn': return 'Research participation has been withdrawn. Normal Reflexion use continues.'
    case 'Study closed': return 'This study is closed. Normal Reflexion use continues.'
    default: return 'No approved study invitation is active for this mirror.'
  }
}

const styles = StyleSheet.create({
  heroRow: { alignItems: 'center', flexDirection: 'row', gap: 22 },
  clipboard: { alignItems: 'center', backgroundColor: c.beige, borderRadius: 76, height: 142, justifyContent: 'center', width: 142 },
  heroCopy: { flex: 1 },
  heroTitle: { color: c.text, fontFamily: f.display, fontSize: 37 },
  heroBody: { color: c.textSecondary, fontFamily: f.body, fontSize: 18, lineHeight: 27, marginTop: 8 },
  cardTitle: { color: c.text, fontFamily: f.display, fontSize: 31, lineHeight: 39 },
  cardBody: { color: c.textSecondary, fontFamily: f.body, fontSize: 18, lineHeight: 27, marginTop: 10 },
  protocol: { color: c.textSecondary, fontFamily: f.body, fontSize: 15, lineHeight: 22, maxWidth: 760, textAlign: 'center' },
})
