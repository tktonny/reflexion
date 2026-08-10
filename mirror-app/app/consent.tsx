import { router, useLocalSearchParams } from 'expo-router'
import { useEffect, useState } from 'react'
import { StyleSheet, Text, View } from 'react-native'

import { dataOrThrow, type DeviceConfiguration } from '../src/api/devicePairing'
import { deviceFetch, getDeviceCredential } from '../src/storage/deviceCredentials'
import { loadJson } from '../src/storage/mirrorStorage'
import { NURSE_PATIENT_CONFIG_STORAGE_KEY } from '../src/constants/nursePatientConfig'
import { MirrorCard, MirrorPage, OutlineButton, PageHeading, StatusRow } from '../src/components/mirror/MirrorChrome'
import { MirrorIcon } from '../src/components/mirror/MirrorIcon'
import { mirrorColors as c, mirrorFonts as f } from '../src/theme/mirrorTheme'
import { isDemoRoute } from '../src/demo/demoConfig'
import { getDemoMirrorState } from '../src/demo/demoRepository'

type ConsentState = 'Pending' | 'Accepted' | 'Declined' | 'Withdrawn'
type ControlState = 'Active' | 'Paused'
type ResearchState = 'not_invited' | 'invitation_pending' | 'consented' | 'declined' | 'withdrawn' | 'study_closed'

export default function ConsentScreen() {
  const params = useLocalSearchParams<{ demo?: string }>()
  const demo = isDemoRoute(params.demo)
  const [consent, setConsent] = useState<ConsentState>('Pending')
  const [control, setControl] = useState<ControlState>('Active')
  const [research, setResearch] = useState<ResearchState>('not_invited')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let mounted = true
    async function load() {
      if (demo) {
        const fixture = await getDemoMirrorState()
        if (!mounted) return
        setConsent(fixture.consent)
        setControl(fixture.consent === 'Accepted' ? 'Active' : 'Paused')
        setResearch(fixture.research === 'Not invited' ? 'not_invited' : fixture.research.toLowerCase().replaceAll(' ', '_') as ResearchState)
        setLoading(false)
        return
      }
      const cached = await loadJson<DeviceConfiguration>(NURSE_PATIENT_CONFIG_STORAGE_KEY)
      const next = await refreshConfiguration(cached)
      if (!mounted) return
      setConsent(resolveConsent(next))
      setControl(resolveControl(next))
      setResearch(resolveResearch(next))
      setLoading(false)
    }
    void load()
    return () => { mounted = false }
  }, [demo])

  return (
    <MirrorPage headerStatus="Mirror ready" onHelp={() => router.push(demo ? '/status?demo=1' : '/status')}>
      <PageHeading title="Your choice, always" subtitle="Product consent & control" />
      <Text style={styles.intro}>Reflexion conversations, reminders, and family messages continue according to your choice.</Text>
      <MirrorCard accent>
        <View style={styles.stateHeader}>
          <View style={styles.shield}><MirrorIcon name="shield-checkmark" size={42} color={c.sageDeep} /></View>
          <View style={styles.stateCopy}>
            <Text style={styles.stateLabel}>Current status</Text>
            <Text style={styles.stateTitle}>{loading ? 'Checking…' : consent}</Text>
            <Text style={styles.stateBody}>{loading ? 'Reading the latest caregiver settings.' : consentDescription(consent)}</Text>
          </View>
        </View>
      </MirrorCard>
      <MirrorCard>
        <StatusRow icon="shield-checkmark-outline" label="Product consent" value={loading ? 'Checking' : consent} state={loading ? 'waiting' : consent === 'Accepted' ? 'good' : consent === 'Pending' ? 'waiting' : 'error'} />
        <StatusRow icon="pause-circle-outline" label="Product control" value={loading ? 'Checking' : control} state={loading ? 'waiting' : control === 'Active' ? 'good' : 'waiting'} />
        {research !== 'not_invited' ? <StatusRow icon="flask-outline" label="Research participation" value={researchLabel(research)} state="waiting" onPress={() => router.push(demo ? '/research?demo=1' : '/research')} /> : null}
      </MirrorCard>
      <Text style={styles.authority}>{demo ? 'Demo only — changes are local and never sync to a caregiver app.' : 'Your caregiver manages these settings. This mirror never changes consent on its own.'}</Text>
      <OutlineButton label="Back to home" icon="home-outline" onPress={() => router.replace(demo ? '/demo' : '/conversation')} />
    </MirrorPage>
  )
}

async function refreshConfiguration(cached: DeviceConfiguration | null) {
  try {
    const credential = await getDeviceCredential()
    if (!credential) return cached
    const response = await deviceFetch(`/api/v1/devices/${encodeURIComponent(credential.deviceId)}/configuration`)
    return await dataOrThrow<DeviceConfiguration>(response)
  } catch {
    return cached
  }
}

function resolveConsent(configuration: DeviceConfiguration | null): ConsentState {
  const value = configuration?.consent?.product || configuration?.desired?.productConsent
  switch (String(value).toLowerCase()) {
    case 'accepted':
    case 'granted': return 'Accepted'
    case 'declined': return 'Declined'
    case 'withdrawn': return 'Withdrawn'
    default: return 'Pending'
  }
}

function resolveControl(configuration: DeviceConfiguration | null): ControlState {
  const value = configuration?.consent?.control || configuration?.desired?.productControl
  return String(value).toLowerCase() === 'paused' ? 'Paused' : 'Active'
}

function resolveResearch(configuration: DeviceConfiguration | null): ResearchState {
  const value = configuration?.consent?.research || configuration?.desired?.researchState
  const normalized = String(value || 'not_invited').toLowerCase() as ResearchState
  return ['not_invited', 'invitation_pending', 'consented', 'declined', 'withdrawn', 'study_closed'].includes(normalized)
    ? normalized
    : 'not_invited'
}

function researchLabel(value: ResearchState) {
  return {
    not_invited: 'Not invited',
    invitation_pending: 'Invitation pending',
    consented: 'Consented',
    declined: 'Declined',
    withdrawn: 'Withdrawn',
    study_closed: 'Study closed',
  }[value]
}

function consentDescription(value: ConsentState) {
  switch (value) {
    case 'Accepted': return 'You have given consent for Reflexion features to work for you.'
    case 'Declined': return 'Reflexion product features are not active.'
    case 'Withdrawn': return 'Product consent has been withdrawn.'
    default: return 'Your caregiver has not yet confirmed product consent.'
  }
}

const styles = StyleSheet.create({
  intro: { color: c.textSecondary, fontFamily: f.body, fontSize: 20, lineHeight: 30, marginBottom: 22, maxWidth: 760, textAlign: 'center' },
  stateHeader: { alignItems: 'center', flexDirection: 'row', gap: 20 },
  shield: { alignItems: 'center', backgroundColor: c.beige, borderRadius: 62, height: 112, justifyContent: 'center', width: 112 },
  stateCopy: { flex: 1 },
  stateLabel: { color: c.sageDeep, fontFamily: f.bodyMedium, fontSize: 18 },
  stateTitle: { color: c.text, fontFamily: f.display, fontSize: 38, marginTop: 4 },
  stateBody: { color: c.textSecondary, fontFamily: f.body, fontSize: 17, lineHeight: 24, marginTop: 5 },
  authority: { color: c.textSecondary, fontFamily: f.body, fontSize: 15, lineHeight: 22, marginBottom: 14, maxWidth: 700, textAlign: 'center' },
})
