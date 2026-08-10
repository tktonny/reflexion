import { router, useLocalSearchParams } from 'expo-router'
import { useCallback, useEffect, useState } from 'react'
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native'

import { currentUpdateLabel } from '../src/lib/otaUpdates'
import { getMirrorBuildInfo } from '../src/lib/buildInfo'
import { runHardwareChecks, type HardwareCheck, type HardwareReport } from '../src/lib/hardwareCheck'
import { getDeviceCredential } from '../src/storage/deviceCredentials'
import { MirrorCard, MirrorPage, OutlineButton, PageHeading, PrimaryButton, StatusRow } from '../src/components/mirror/MirrorChrome'
import { MirrorIcon } from '../src/components/mirror/MirrorIcon'
import { mirrorColors as c, mirrorFonts as f } from '../src/theme/mirrorTheme'
import { isDemoRoute } from '../src/demo/demoConfig'

export default function StatusScreen() {
  const params = useLocalSearchParams<{ demo?: string }>()
  const demo = isDemoRoute(params.demo)
  const [report, setReport] = useState<HardwareReport | null>(null)
  const [deviceId, setDeviceId] = useState<string | null>(null)
  const [checkedAt, setCheckedAt] = useState<Date | null>(null)
  const refresh = useCallback(async () => {
    if (demo) {
      setReport({
        platform: 'android-demo',
        checks: [
          { key: 'api', label: 'Backend', status: 'unknown', detail: 'Skipped in local demo' },
          { key: 'network', label: 'Network', status: 'unknown', detail: 'Skipped in local demo' },
          { key: 'identity', label: 'Device identity', status: 'unknown', detail: 'Demo identity only' },
          { key: 'mic', label: 'Microphone', status: 'warn', detail: 'Run the real hardware test from the test build' },
          { key: 'speaker', label: 'Speaker', status: 'warn', detail: 'Run the real hardware test from the test build' },
        ],
        recommendedMode: 'none',
        recommendedReason: 'Local demo does not use a backend',
        configuredMode: 'ws',
      })
      setDeviceId('demo-local')
      setCheckedAt(new Date())
      return
    }
    const [next, credential] = await Promise.all([runHardwareChecks(), getDeviceCredential()])
    setReport(next)
    setDeviceId(credential?.deviceId || null)
    setCheckedAt(new Date())
  }, [demo])
  useEffect(() => { void refresh() }, [refresh])

  const build = getMirrorBuildInfo()
  const api = find(report, 'api')
  const network = find(report, 'network')
  const identity = find(report, 'identity')
  const mic = find(report, 'mic')
  const speaker = find(report, 'speaker')
  const serviceGood = !demo && (api?.status === 'ok' || api?.status === 'warn')
  return (
    <MirrorPage headerStatus="Mirror ready" onHelp={() => undefined}>
      <PageHeading title="Device status & help" subtitle="Everything looks good when each technical check is ready." />
      <MirrorCard>
        <StatusRow icon="wifi" label="Wi-Fi connected" value={network?.status === 'fail' ? 'Unavailable' : report ? 'Connected' : 'Checking'} state={network?.status === 'fail' ? 'error' : report ? 'good' : 'waiting'} onPress={() => router.push('/wifi-setup')} />
        <StatusRow icon="globe-outline" label="Internet connected" value={serviceGood ? 'Connected' : report ? 'Unavailable' : 'Checking'} state={serviceGood ? 'good' : report ? 'error' : 'waiting'} />
        <StatusRow icon="cloud-done-outline" label="Reflexion service reachable" value={serviceGood ? 'Reachable' : report ? 'Unavailable' : 'Checking'} state={serviceGood ? 'good' : report ? 'error' : 'waiting'} />
        <StatusRow icon="shield-checkmark-outline" label="Device authentication" value={identity?.status === 'fail' ? 'Needs attention' : identity ? 'Ready' : 'Checking'} state={identity?.status === 'fail' ? 'error' : identity ? 'good' : 'waiting'} />
        <StatusRow icon="mic-outline" label="Microphone working" value={statusLabel(mic)} state={statusState(mic)} onPress={() => router.push('/hardware-check')} />
        <StatusRow icon="volume-high-outline" label="Speaker working" value={statusLabel(speaker)} state={statusState(speaker)} onPress={() => router.push('/hardware-check')} />
      </MirrorCard>
      <MirrorCard>
        <Text style={styles.sectionTitle}>Mirror details</Text>
        <Text style={styles.detail}>App version · {build.appVersion} (build {build.buildNumber})</Text>
        <Text style={styles.detail}>Update state · {currentUpdateLabel()}</Text>
        <Text style={styles.detail}>Device assignment · {demo ? 'Demo local only' : deviceId ? 'Connected' : 'Not paired'}</Text>
        <Text style={styles.detail}>Last service check · {demo ? 'Not used in Demo Mode' : checkedAt ? checkedAt.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : 'Checking'}</Text>
      </MirrorCard>
      <PrimaryButton label={report ? 'Check again' : 'Checking…'} icon="refresh-outline" onPress={report ? () => { void refresh() } : undefined} />
      <OutlineButton label="Back to home" icon="home-outline" onPress={() => router.replace(demo ? '/demo' : '/conversation')} />
      {!report ? <ActivityIndicator color={c.goldDeep} style={styles.spinner} /> : null}
      <View style={styles.supportRow}><MirrorIcon name="people-outline" size={23} color={c.sageDeep} /><Text style={styles.supportText}>Need extra help? Ask your caregiver for assistance if needed.</Text></View>
    </MirrorPage>
  )
}

function find(report: HardwareReport | null, key: string): HardwareCheck | undefined {
  return report?.checks.find((check) => check.key === key)
}

function statusLabel(check?: HardwareCheck) {
  if (!check) return 'Checking'
  if (check.status === 'ok') return 'OK'
  if (check.status === 'fail') return 'Needs attention'
  return 'Check recommended'
}

function statusState(check?: HardwareCheck): 'good' | 'waiting' | 'error' {
  if (!check) return 'waiting'
  if (check.status === 'ok') return 'good'
  if (check.status === 'fail') return 'error'
  return 'waiting'
}

const styles = StyleSheet.create({
  sectionTitle: { color: c.text, fontFamily: f.display, fontSize: 29, marginBottom: 10 },
  detail: { color: c.textSecondary, fontFamily: f.body, fontSize: 16, lineHeight: 25 },
  spinner: { marginTop: 16 },
  supportRow: { alignItems: 'center', flexDirection: 'row', gap: 10, marginTop: 21, maxWidth: 760 },
  supportText: { color: c.textSecondary, flex: 1, fontFamily: f.body, fontSize: 15, lineHeight: 22 },
})
