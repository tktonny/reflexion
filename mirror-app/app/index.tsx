import AsyncStorage from '@react-native-async-storage/async-storage'
import { router } from 'expo-router'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ActivityIndicator, Alert, Animated, Easing, Platform, Pressable, Share, StyleSheet, Text, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import qrcode from 'qrcode-generator'

import { getApiUrl, hasConfiguredApiBase } from '../src/config/apiUrl'
import { MirrorIcon } from '../src/components/mirror/MirrorIcon'
import {
  createDevicePairing,
  dataOrThrow,
  exchangeDeviceCredential,
  getDevicePairing,
  type DeviceConfiguration,
} from '../src/api/devicePairing'
import { DEFAULT_LANGUAGE } from '../src/config/conversationMode'
import {
  ACTIVE_MIRROR_ID_STORAGE_KEY,
  ACTIVE_NURSE_ID_STORAGE_KEY,
  ACTIVE_PATIENT_ID_STORAGE_KEY,
  DEVICE_ID_STORAGE_KEY,
  MIRROR_LANGUAGE_STORAGE_KEY,
  MIRROR_TIMEZONE_STORAGE_KEY,
  NURSE_PATIENT_CONFIG_STORAGE_KEY,
} from '../src/constants/nursePatientConfig'
import {
  clearDeviceCredential,
  deviceFetch,
  getDeviceCredential,
  getDevicePairingIdentity,
  getPendingDevicePairing,
  getPairingIdempotencyKey,
  clearPendingDevicePairing,
  savePendingDevicePairing,
} from '../src/storage/deviceCredentials'
import type { ReadinessVerdict } from '../src/lib/readiness'
import {
  pairingFailureFromError,
  recordPairingDiagnostic,
  type PairingDiagnosticEvent,
  type PairingDiagnosticReasonCode,
  type PairingDiagnosticStage,
} from '../src/lib/pairingDiagnostics'
import { getMirrorBuildInfo, logMirrorBuildInfo, type SafeBuildInfo } from '../src/lib/buildInfo'
import { WifiSetupView } from '../src/components/mirror/WifiSetupView'
import { PhoneSetupInstructions } from '../src/components/mirror/PhoneSetupInstructions'
import { getSetupModeState, networkSetupAvailable, subscribeSetupMode, type SetupModeState } from '../src/native/networkSetup'
import { mirrorColors as palette, mirrorFonts as fonts } from '../src/theme/mirrorTheme'
import { DEMO_BUILD_ENABLED } from '../src/demo/demoConfig'

export type BootCheck = { key: string; label: string; ok: boolean }
export type PairingSetup = {
  current: PairingDiagnosticStage | null
  completed: PairingDiagnosticStage[]
  failure: PairingDiagnosticEvent | null
}
export type PairingDetails = {
  deviceId: string
  pairingId: string
  pairingCode: string
  pairingToken?: string
  qrPayload: string
  expiresAt?: string
}

const INSTALLER_SETUP_ENABLED = __DEV__ || process.env.EXPO_PUBLIC_ENABLE_INSTALLER_SETUP === 'true'
const BACKEND_CONFIGURED = __DEV__ || hasConfiguredApiBase()
const SETUP_STEPS: Array<{ key: PairingDiagnosticStage; label: string }> = [
  { key: 'WIFI_CONNECTED', label: 'Connected to Wi-Fi' },
  { key: 'INTERNET_AVAILABLE', label: 'Internet connected' },
  { key: 'SERVICE_REACHABLE', label: 'Reflexion service' },
  { key: 'AUTHENTICATION_READY', label: 'Device authentication' },
  { key: 'READY_TO_PAIR', label: 'Pairing ready' },
  { key: 'CLAIM_IN_PROGRESS', label: 'Assignment' },
  { key: 'PAIRED', label: 'Ready to use' },
]

// The one blocking verdict the boot flow can both detect and fix in place.
const MICROPHONE_BLOCKED: ReadinessVerdict = {
  blocked: true,
  problem: 'microphone',
  title: 'Aria cannot hear you yet',
  body: 'The microphone is switched off for this app, so Aria would not hear your answers.',
  actionLabel: 'Turn on the microphone',
}

export default function BootScreen() {
  const [booting, setBooting] = useState(true)
  const [checks, setChecks] = useState<BootCheck[]>([])
  const [pairing, setPairing] = useState<PairingDetails | null>(null)
  const [pairingError, setPairingError] = useState('')
  const [wifiRequired, setWifiRequired] = useState(false)
  const [showPairing, setShowPairing] = useState(false)
  const [recoveryRequired, setRecoveryRequired] = useState(false)
  const [offlineHome, setOfflineHome] = useState(false)
  const [blocked, setBlocked] = useState<ReadinessVerdict | null>(null)
  const [setup, setSetup] = useState<PairingSetup>({ current: null, completed: [], failure: null })
  const [latestDiagnostic, setLatestDiagnostic] = useState<PairingDiagnosticEvent | null>(null)
  const [setupMode, setSetupMode] = useState<SetupModeState | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const buildInfo = useMemo(() => getMirrorBuildInfo(), [])

  useEffect(() => { logMirrorBuildInfo() }, [])

  const note = useCallback(async (
    stage: PairingDiagnosticStage,
    ok: boolean,
    reasonCode?: PairingDiagnosticReasonCode,
    options?: Parameters<typeof recordPairingDiagnostic>[3],
  ) => {
    const event = await recordPairingDiagnostic(stage, ok, reasonCode, options)
    setLatestDiagnostic(event)
    setSetup((current) => ({
      current: stage,
      completed: ok && !current.completed.includes(stage) ? [...current.completed, stage] : current.completed,
      failure: ok ? null : event,
    }))
    return event
  }, [])

  // Hoisted out of the mount effect so the readiness screen can re-run it after the elder fixes the
  // problem (permission granted -> straight into the conversation, no restart).
  const boot = useCallback(async () => {
    setWifiRequired(false)
    setShowPairing(false)
    const result = await runBootChecks(note)
    setChecks(result.checks)
    if (result.paired && !result.backendReachable) {
      setOfflineHome(true)
      setBooting(false)
      return
    }
    if (result.credentialRejected) {
      setRecoveryRequired(true)
      setPairingError(result.failure?.userMessage || 'Reflexion could not verify this mirror.')
      setBooting(false)
      return
    }
    if (result.paired && result.backendReachable) {
      // Readiness gate: never hand an elder a conversation that cannot work. A denied microphone used
      // to sail through here and capture silence.
      if (!result.microphoneGranted) {
        setBlocked(MICROPHONE_BLOCKED)
        setBooting(false)
        return
      }
      try {
        await refreshAndPersistDeviceProfile(result.deviceId)
        await note('PAIRED', true)
        router.replace('/conversation')
        return
      } catch (error) {
        const failure = pairingFailureFromError(error, 'AUTHENTICATION_FAILED')
        await note(failure.stage, false, failure.reasonCode, failure)
      }
    }
    if (!result.wifiConnected) {
      setWifiRequired(true)
      setPairingError(result.failure?.userMessage || 'Your mirror is not connected to Wi-Fi.')
      setBooting(false)
      return
    }
    if (!result.backendReachable) {
      setPairingError(result.failure?.userMessage || 'The mirror is online, but Reflexion cannot be reached.')
      setBooting(false)
      return
    }
    await loadPairingCode(result.deviceId)
    setBooting(false)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [note])

  useEffect(() => {
    // The browser demo is an explicit local build. Send web testers straight to the fixture
    // surface instead of making them pass through the production pairing boot flow first.
    if (DEMO_BUILD_ENABLED && Platform.OS === 'web') {
      router.replace('/demo')
      return
    }
    if (DEMO_BUILD_ENABLED) {
      setPairingError('This is a local demo build. Production connectivity is intentionally disabled.')
      setBooting(false)
      return
    }
    if (!BACKEND_CONFIGURED) {
      setBooting(false)
      return
    }
    void boot()
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [boot])

  // Electron pushes setup-hotspot changes across the preload bridge. Keep this screen reactive so a
  // successful phone-configured Wi-Fi join returns the appliance to its normal readiness flow.
  useEffect(() => {
    void getSetupModeState().then((state) => { if (state) setSetupMode(state) })
    return subscribeSetupMode((state) => {
      setSetupMode(state)
      if (state.lastResult?.ok && !DEMO_BUILD_ENABLED) { setBooting(true); void boot() }
    })
  }, [boot])

  async function loadPairingCode(knownDeviceId = '', forceRefresh = false) {
    setSetup((current) => ({ ...current, current: 'PAIRING_SESSION_CREATING', failure: null }))
    try {
      const details = await requestPairingCode(forceRefresh)
      setPairing(details)
      await note('AUTHENTICATION_READY', true)
      await note('PAIRING_SESSION_CREATING', true)
      await note('READY_TO_PAIR', true)
      setRecoveryRequired(false)
      setPairingError('')
    } catch (error) {
      setPairing(null)
      const failure = pairingFailureFromError(error)
      await note(failure.stage, false, failure.reasonCode, failure)
      if (failure.stage === 'DEVICE_ALREADY_CLAIMED' || failure.reasonCode === 'DEVICE_RECOVERY_REQUIRED') {
        setRecoveryRequired(true)
        setPairingError(failure.userMessage)
      } else {
        setPairingError(failure.userMessage || (knownDeviceId
          ? 'The mirror is online, but Reflexion cannot be reached.'
          : 'Reflexion could not prepare this mirror for pairing.'))
      }
    }
  }

  useEffect(() => {
    if (!pairing) return
    const activePairing = pairing
    async function pollStatus() {
      try {
        const status = await getDevicePairing(activePairing.pairingId)
        if (status.state === 'expired' || status.state === 'cancelled') {
          if (pollRef.current) clearInterval(pollRef.current)
          await note('PAIRING_CODE_EXPIRED', false, 'PAIRING_CODE_EXPIRED')
          await loadPairingCode(activePairing.deviceId, true)
          return
        }
        if (status.state !== 'paired') return
        await note('CLAIM_IN_PROGRESS', true)
        await exchangeDeviceCredential(status)
        await refreshAndPersistDeviceProfile(activePairing.deviceId)
        await note('PAIRED', true)
        if (pollRef.current) clearInterval(pollRef.current)
        router.replace({ pathname: '/paired-success', params: { name: status.patientDisplayName || 'your loved one' } })
      } catch (error) {
        const failure = pairingFailureFromError(error, 'CREDENTIAL_ISSUANCE_FAILED')
        await note(failure.stage, false, failure.reasonCode, failure)
        setPairingError(failure.userMessage)
        if (!failure.retryable && pollRef.current) clearInterval(pollRef.current)
      }
    }
    void pollStatus()
    pollRef.current = setInterval(() => { void pollStatus() }, 3500)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pairing?.pairingId])

  if (setupMode?.active) return <PhoneSetupScreen state={setupMode} />
  if (!BACKEND_CONFIGURED && !DEMO_BUILD_ENABLED) return <StagingBackendNotConfiguredScreen buildInfo={buildInfo} />
  if (booting) return <BootLoadingScreen checks={checks} />
  if (wifiRequired) {
    return (
      <WifiSetupView
        error={pairingError}
        onRetry={() => { setWifiRequired(false); setBooting(true); void boot() }}
      />
    )
  }
  if (blocked) {
    // The action re-requests the OS permission; either way we re-run boot so a fixed mirror proceeds
    // straight into the conversation without the elder needing to restart anything.
    const retryBoot = () => { setBlocked(null); setBooting(true); void boot() }
    return (
      <NotReadyScreen
        verdict={blocked}
        onAction={() => { void checkMicrophonePermission().then(retryBoot) }}
        onRetry={retryBoot}
      />
    )
  }
  if (offlineHome) return <OfflineHomeScreen onRetry={() => router.replace('/conversation')} />
  return (
    <PairingScreen
      error={pairingError}
      onRefresh={() => { setShowPairing(true); void loadPairingCode('', true) }}
      onRetry={() => void loadPairingCode()}
      pairing={pairing}
      recoveryRequired={recoveryRequired}
      buildInfo={buildInfo}
      setup={setup}
      latestDiagnostic={latestDiagnostic}
      showPairing={showPairing}
      onShowPairing={() => setShowPairing(true)}
      onBackToReadiness={() => setShowPairing(false)}
    />
  )
}

function StagingBackendNotConfiguredScreen({ buildInfo }: { buildInfo: SafeBuildInfo }) {
  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.stage}>
        <View pointerEvents="none" style={styles.reflection} />
        <View style={styles.pairScene}>
          <Text style={styles.eyebrow}>MIRROR STAGING</Text>
          <Text style={styles.pairTitle}>Staging backend not configured</Text>
          <Text style={styles.pairBody}>
            This staging build is ready for OTA validation. Pairing and conversations will remain unavailable until the staging API is configured.
          </Text>
          <View style={styles.offlineIcon}>
            <MirrorIcon name="cloud-offline-outline" size={44} color={palette.linen} />
          </View>
          <BuildSummary buildInfo={buildInfo} />
          {networkSetupAvailable() ? (
            <Pressable onPress={() => router.push('/network-setup')} style={styles.retryButton}>
              <Text style={styles.retryText}>Set up the connection</Text>
            </Pressable>
          ) : null}
          {INSTALLER_SETUP_ENABLED ? (
            <Pressable onPress={() => router.push('/test-device')} style={styles.retryButton}>
              <Text style={styles.retryText}>Installer setup</Text>
            </Pressable>
          ) : null}
        </View>
      </View>
    </SafeAreaView>
  )
}

type DiagnosticSink = (
  stage: PairingDiagnosticStage,
  ok: boolean,
  reasonCode?: PairingDiagnosticReasonCode,
  options?: Parameters<typeof recordPairingDiagnostic>[3],
) => Promise<PairingDiagnosticEvent>

type ProbeResult = {
  ok: boolean
  status?: number
  requestId?: string
  reasonCode?: PairingDiagnosticReasonCode
}

async function runBootChecks(note: DiagnosticSink) {
  let credential: Awaited<ReturnType<typeof getDeviceCredential>> = null
  let savedDeviceId: string | null = null
  let identityFailure: PairingDiagnosticEvent | null = null
  try {
    ;[credential, savedDeviceId] = await Promise.all([
      getDeviceCredential(),
      AsyncStorage.getItem(DEVICE_ID_STORAGE_KEY),
    ])
  } catch {
    identityFailure = await note('DEVICE_IDENTITY_FAILED', false, 'SECURE_STORAGE_FAILURE')
  }
  let identity: Awaited<ReturnType<typeof getDevicePairingIdentity>> | null = null
  try {
    identity = await getDevicePairingIdentity()
  } catch {
    identityFailure ||= await note('DEVICE_IDENTITY_FAILED', false, 'SECURE_STORAGE_FAILURE')
  }
  const deviceId = credential?.deviceId || identity?.deviceId || savedDeviceId || ''
  const paired = Boolean(credential && credential.deviceId === deviceId)
  if (!paired) await clearLegacyPairingState()
  const network = await checkConnection()
  const wifiEvent = await note(
    network.wifiConnected ? 'WIFI_CONNECTED' : 'WIFI_DISCONNECTED',
    network.wifiConnected,
    network.wifiConnected ? undefined : 'NETWORK_UNAVAILABLE',
  )
  const internetEvent = await note(
    network.internetAvailable ? 'INTERNET_AVAILABLE' : 'INTERNET_UNAVAILABLE',
    network.internetAvailable,
    network.internetAvailable ? undefined : (network.internetProbe.reasonCode || 'NETWORK_UNAVAILABLE'),
    network.internetProbe.status ? { httpStatus: network.internetProbe.status } : undefined,
  )
  const serviceEvent = await note(
    network.backendReachable ? 'SERVICE_REACHABLE' : 'SERVICE_UNREACHABLE',
    network.backendReachable,
    network.backendReachable ? undefined : (network.backendProbe.reasonCode || 'SERVICE_UNREACHABLE'),
    {
      ...(network.backendProbe.status ? { httpStatus: network.backendProbe.status } : {}),
      ...(network.backendProbe.requestId ? { requestId: network.backendProbe.requestId } : {}),
    },
  )
  let failure = identityFailure || (!network.wifiConnected ? wifiEvent : !network.internetAvailable ? internetEvent : !network.backendReachable ? serviceEvent : null)
  let authenticated = false
  let credentialRejected = false
  if (paired && network.backendReachable) {
    try {
      const response = await deviceFetch(`/api/v1/devices/${encodeURIComponent(deviceId)}`)
      authenticated = response.ok
      if (authenticated) await note('AUTHENTICATION_READY', true)
      if (!response.ok) {
        credentialRejected = response.status === 401
        const authEvent = await note('AUTHENTICATION_FAILED', false, response.status === 401 ? 'CREDENTIAL_REJECTED' : 'HTTP_ERROR', { httpStatus: response.status })
        failure ||= authEvent
      }
    } catch {
      credentialRejected = true
      const authEvent = await note('AUTHENTICATION_FAILED', false, 'CREDENTIAL_REJECTED')
      failure ||= authEvent
    }
  }
  const microphoneGranted = await checkMicrophonePermission()
  const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone
  if (timezone) await AsyncStorage.setItem(MIRROR_TIMEZONE_STORAGE_KEY, timezone)
  return {
    deviceId,
    ...network,
    paired,
    authenticated,
    credentialRejected,
    microphoneGranted,
    failure,
    checks: [
      { key: 'wifi', label: 'Wi-Fi connected', ok: network.wifiConnected },
      { key: 'internet', label: 'Internet available', ok: network.internetAvailable },
      { key: 'backend', label: 'Reflexion service reachable', ok: network.backendReachable },
      { key: 'authenticated', label: 'Mirror authenticated', ok: authenticated || !credential },
      { key: 'paired', label: 'Mirror paired with household', ok: paired && authenticated },
      { key: 'assigned', label: 'Mirror assigned to loved one', ok: paired && authenticated },
      { key: 'microphone', label: 'Microphone ready', ok: microphoneGranted },
      { key: 'timezone', label: 'Time and timezone ready', ok: Boolean(timezone) },
    ] satisfies BootCheck[],
  }
}

async function clearLegacyPairingState() {
  await clearDeviceCredential({ preserveBootstrap: true })
  await AsyncStorage.multiRemove([
    ACTIVE_MIRROR_ID_STORAGE_KEY,
    ACTIVE_NURSE_ID_STORAGE_KEY,
    ACTIVE_PATIENT_ID_STORAGE_KEY,
    NURSE_PATIENT_CONFIG_STORAGE_KEY,
    MIRROR_LANGUAGE_STORAGE_KEY,
  ])
}

async function pingBackend(): Promise<ProbeResult> {
  const controller = new AbortController()
  // The production health route can take just over three seconds on a cold/slow mirror link.
  // Keep the boot check bounded, but do not turn a reachable backend into a false offline state.
  const timer = setTimeout(() => controller.abort(), 8000)
  try {
    const response = await fetch(getApiUrl('/health'), { signal: controller.signal })
    return {
      ok: response.ok,
      status: response.status,
      requestId: response.headers.get('x-request-id') || undefined,
      ...(response.ok ? {} : { reasonCode: response.status === 404 ? 'ROUTE_NOT_FOUND' : 'HEALTH_CHECK_FAILED' }),
    }
  } catch (error) {
    return { ok: false, reasonCode: error instanceof Error && error.name === 'AbortError' ? 'TIMEOUT' : 'SERVICE_UNREACHABLE' }
  } finally {
    clearTimeout(timer)
  }
}

async function checkConnection() {
  const reportedOnline = typeof navigator === 'undefined' || navigator.onLine !== false
  const [internetProbe, backendProbe] = await Promise.all([
    pingInternet(),
    pingBackend(),
  ])
  // React Native's navigator.onLine can remain false on a Wi-Fi-connected Android device. The
  // probes above are the authoritative signal for this app; a reachable production backend also
  // proves the mirror has working internet even if the generic connectivity endpoint is blocked.
  const internetAvailable = internetProbe.ok || backendProbe.ok
  return {
    wifiConnected: reportedOnline || internetProbe.ok || backendProbe.ok,
    internetAvailable,
    backendReachable: backendProbe.ok,
    internetProbe,
    backendProbe,
  }
}

async function pingInternet(): Promise<ProbeResult> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 2500)
  try {
    const response = await fetch('https://connectivitycheck.gstatic.com/generate_204', { signal: controller.signal })
    return { ok: response.ok, status: response.status, ...(response.ok ? {} : { reasonCode: 'HEALTH_CHECK_FAILED' }) }
  } catch (error) {
    return { ok: false, reasonCode: error instanceof Error && error.name === 'AbortError' ? 'TIMEOUT' : 'NETWORK_UNAVAILABLE' }
  } finally {
    clearTimeout(timer)
  }
}

async function checkMicrophonePermission() {
  if (Platform.OS !== 'web') {
    // This used to `return true` unconditionally on native, so the boot screen showed the elder
    // "Microphone ready ✓" on a mirror whose mic permission was denied — the one check that most needed
    // to be honest was the one that was hardcoded. Ask the OS, and request once if undetermined.
    try {
      const audio: any = await import('expo-audio')
      const current = await audio.getRecordingPermissionsAsync?.()
      if (current?.granted) return true
      const requested = await audio.requestRecordingPermissionsAsync?.()
      return Boolean(requested?.granted)
    } catch {
      return false
    }
  }
  if (!navigator.mediaDevices?.getUserMedia) return false
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
    stream.getTracks().forEach((track) => track.stop())
    return true
  } catch {
    return false
  }
}

async function requestPairingCode(forceRefresh = false): Promise<PairingDetails> {
  const identity = await getDevicePairingIdentity()
  const existing = forceRefresh ? null : await getPendingDevicePairing()
  if (existing && existing.deviceId === identity.deviceId && Date.parse(existing.expiresAt) > Date.now()) {
    return {
      deviceId: existing.deviceId,
      pairingId: existing.pairingId,
      pairingCode: existing.pairingCode,
      pairingToken: existing.pairingToken,
      qrPayload: JSON.stringify({ type: 'reflexion_device_pairing_v2', pairingToken: existing.pairingToken }),
      expiresAt: existing.expiresAt,
    }
  }
  if (existing || forceRefresh) await clearPendingDevicePairing()
  const idempotencyKey = await getPairingIdempotencyKey()
  const body = await createDevicePairing(idempotencyKey)
  const details = {
    deviceId: body.deviceId || identity.deviceId,
    pairingId: body.pairingId,
    pairingCode: body.displayCode,
    pairingToken: body.pairingToken,
    qrPayload: JSON.stringify({ type: 'reflexion_device_pairing_v2', pairingToken: body.pairingToken }),
    expiresAt: body.expiresAt,
  }
  await savePendingDevicePairing({ ...details, idempotencyKey })
  return details
}

async function refreshAndPersistDeviceProfile(deviceId: string) {
  const response = await deviceFetch(`/api/v1/devices/${encodeURIComponent(deviceId)}/configuration`)
  const configuration = await dataOrThrow<DeviceConfiguration>(response)
  if (!configuration.patient) throw new Error('paired_patient_configuration_missing')
  await AsyncStorage.multiSet([
    [DEVICE_ID_STORAGE_KEY, deviceId],
    [ACTIVE_MIRROR_ID_STORAGE_KEY, deviceId],
    [ACTIVE_PATIENT_ID_STORAGE_KEY, configuration.patient.patientId],
    [NURSE_PATIENT_CONFIG_STORAGE_KEY, JSON.stringify(configuration)],
    [MIRROR_LANGUAGE_STORAGE_KEY, configuration.patient.preferredLanguage || DEFAULT_LANGUAGE],
    [MIRROR_TIMEZONE_STORAGE_KEY, configuration.patient.timezone || Intl.DateTimeFormat().resolvedOptions().timeZone],
  ])
}

export function BootLoadingScreen({ checks }: { checks: BootCheck[] }) {
  const readyCount = checks.filter((check) => check.ok).length
  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.stage}>
        <View pointerEvents="none" style={styles.reflection} />
        <View style={styles.bootScene}>
          <BrandMark />
          <Text style={styles.brand}>REFLEXION</Text>
          <Text style={styles.loadingText}>Starting Reflexion…</Text>
          <LoadingDots />
          {checks.length ? <Text style={styles.bootStatus}>{readyCount} of {checks.length} checks ready</Text> : null}
        </View>
      </View>
    </SafeAreaView>
  )
}

export function PairingScreen({
  error,
  onRefresh,
  onRetry,
  pairing,
  recoveryRequired,
  buildInfo = getMirrorBuildInfo(),
  setup = { current: null, completed: [], failure: null },
  latestDiagnostic = null,
  showPairing = false,
  onShowPairing,
  onBackToReadiness,
}: {
  error: string
  onRefresh?: () => void
  onRetry: () => void
  pairing: PairingDetails | null
  recoveryRequired?: boolean
  buildInfo?: SafeBuildInfo
  setup?: PairingSetup
  latestDiagnostic?: PairingDiagnosticEvent | null
  showPairing?: boolean
  onShowPairing?: () => void
  onBackToReadiness?: () => void
}) {
  const refresh = onRefresh || onRetry
  const [showDetails, setShowDetails] = useState(false)
  const failure = setup.failure || (latestDiagnostic?.ok ? null : latestDiagnostic)
  const failureMessage = failure?.userMessage || error
  const report = failure ? formatDiagnosticReport(failure, buildInfo) : ''
  const isClaimed = failure?.stage === 'DEVICE_ALREADY_CLAIMED'
  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.stage}>
        <View pointerEvents="none" style={styles.reflection} />
        <View style={styles.pairScene}>
          <Text style={styles.eyebrow}>{showPairing ? 'MIRROR SETUP' : 'MIRROR ONLINE'}</Text>
          <Text style={styles.pairTitle}>{showPairing ? 'Pair your mirror' : 'Mirror setup status'}</Text>
          <Text style={styles.pairBody}>{showPairing
            ? 'Open the Reflexion Caregiver app and scan this QR code or enter the code below.'
            : failureMessage || (pairing ? 'Everything looks good. Your mirror is ready to connect.' : 'We are checking the mirror’s connection.')}</Text>
          <PairingChecklist setup={setup} />
          {pairing && showPairing ? (
            <>
              <Pressable onPress={onBackToReadiness} style={styles.backToReadiness}>
                <MirrorIcon name="chevron-back" size={21} color={palette.sageDeep} />
                <Text style={styles.demoLinkText}>Back to readiness</Text>
              </Pressable>
              <Text style={styles.pairingLabel}>PAIRING CODE</Text>
              <Text style={styles.pairingCode}>{formatPairingCode(pairing.pairingCode)}</Text>
              <QrCode value={pairing.qrPayload} />
              <Text style={styles.scanText}>Scan with the caregiver app to pair this mirror.</Text>
              {pairing.expiresAt ? <Text style={styles.waitingText}>Code refreshes at {formatExpiry(pairing.expiresAt)}.</Text> : null}
              <Text style={styles.waitingText}>Waiting securely for pairing…</Text>
              <Pressable onPress={refresh} style={styles.refreshCodeButton}>
                <MirrorIcon name="refresh-outline" size={23} color={palette.sageDeep} />
                <Text style={styles.demoLinkText}>Refresh code</Text>
              </Pressable>
            </>
          ) : pairing ? (
            <View style={styles.readyCard}>
              <View style={styles.readyIcon}><MirrorIcon name="qr-code-outline" size={42} color={palette.sageDeep} /></View>
              <View style={styles.readyCopy}>
                <Text style={styles.readyTitle}>Ready to pair</Text>
                <Text style={styles.readyBody}>Share the pairing code with the caregiver app to connect your mirror.</Text>
              </View>
              <Pressable accessibilityRole="button" onPress={onShowPairing} style={styles.showCodeButton}>
                <MirrorIcon name="qr-code-outline" size={23} color={palette.cream} />
                <Text style={styles.retryText}>Show pairing code</Text>
              </Pressable>
            </View>
          ) : (
            <View style={styles.offlineIcon}>
              <MirrorIcon name="cloud-offline-outline" size={44} color={palette.linen} />
            </View>
          )}
          {failureMessage && pairing ? <Text style={styles.errorText}>{failureMessage}</Text> : null}
          {failure ? (
            <>
              <Pressable onPress={() => setShowDetails((current) => !current)} style={styles.detailsLink}>
                <Text style={styles.demoLinkText}>{showDetails ? 'Hide details' : 'Show details'}</Text>
              </Pressable>
              {showDetails ? (
                <View style={styles.detailsCard}>
                  <Text style={styles.detailsText}>Support code: {failure.supportCode}</Text>
                  <Text style={styles.detailsText}>Occurred: {new Date(failure.at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</Text>
                  <Text style={styles.detailsText}>Mirror version: {buildInfo.appVersion} (build {buildInfo.buildNumber})</Text>
                  <View style={styles.detailsActions}>
                    {!isClaimed ? <Pressable onPress={onRetry} style={styles.smallAction}><Text style={styles.smallActionText}>Retry</Text></Pressable> : null}
                    <Pressable onPress={() => void Share.share({ title: 'Reflexion pairing diagnostic', message: report })} style={styles.smallAction}>
                      <Text style={styles.smallActionText}>Copy diagnostic report</Text>
                    </Pressable>
                  </View>
                </View>
              ) : null}
            </>
          ) : null}
          {pairing && showPairing ? (
            <Pressable onPress={refresh} style={styles.demoLink}><Text style={styles.demoLinkText}>Generate a new code</Text></Pressable>
          ) : isClaimed ? (
            <Text style={styles.offlineNote}>This mirror needs a reset or transfer before it can be paired again.</Text>
          ) : <Pressable onPress={onRetry} style={styles.retryButton}><Text style={styles.retryText}>{recoveryRequired ? 'Check again' : 'Try again'}</Text></Pressable>}
          {!pairing && networkSetupAvailable() ? (
            <Pressable onPress={() => router.push('/network-setup')} style={styles.demoLink}>
              <Text style={styles.demoLinkText}>Set up the internet connection</Text>
            </Pressable>
          ) : null}
          {!pairing && INSTALLER_SETUP_ENABLED ? (
            <Pressable onPress={() => router.push('/test-device')} style={styles.demoLink}>
              <Text style={styles.demoLinkText}>Installer setup</Text>
            </Pressable>
          ) : null}
          {DEMO_BUILD_ENABLED ? (
            <Pressable
              accessibilityRole="button"
              onPress={() => {
                if (Platform.OS === 'web') {
                  // React Native Alert has no reliable web implementation. The web demo is already
                  // an explicit demo build, so enter it directly from this control.
                  router.push('/demo')
                  return
                }
                Alert.alert(
                  'Enter Demo Mode?',
                  'Demo mode uses local sample data. Changes will not sync to the caregiver app.',
                  [
                    { text: 'Cancel', style: 'cancel' },
                    { text: 'Enter demo mode', onPress: () => router.push('/demo') },
                  ],
                )
              }}
              style={styles.demoLink}
            >
              <Text style={styles.demoLinkText}>Enter local Demo Mode</Text>
            </Pressable>
          ) : null}
        </View>
      </View>
    </SafeAreaView>
  )
}

function PairingChecklist({ setup }: { setup: PairingSetup }) {
  const failure = setup.failure
  return (
    <View accessibilityLabel="Mirror setup checklist" style={styles.checklist}>
      {SETUP_STEPS.map((step) => {
        const completed = setup.completed.includes(step.key)
        const failed = Boolean(failure && failureAffectsStep(failure.stage, step.key))
        const active = !completed && !failed && setup.current === step.key
        return (
          <View key={step.key} style={styles.checklistRow}>
            <View style={[styles.checkIcon, completed && styles.checkIconDone, failed && styles.checkIconFailed]}>
              {completed ? <MirrorIcon name="checkmark" size={14} color={palette.cream} /> : failed ? <MirrorIcon name="close" size={14} color={palette.cream} /> : active ? <ActivityIndicator color={palette.goldDeep} size="small" /> : <View style={styles.checkIconPending} />}
            </View>
            <Text style={[styles.checkLabel, completed && styles.checkLabelDone, failed && styles.checkLabelFailed]}>{step.label}</Text>
          </View>
        )
      })}
      {failure && !SETUP_STEPS.some((step) => failureAffectsStep(failure.stage, step.key)) ? (
        <View style={styles.checklistRow}>
          <View style={[styles.checkIcon, styles.checkIconFailed]}><MirrorIcon name="close" size={14} color={palette.cream} /></View>
          <Text style={styles.checkLabelFailed}>{failure.userMessage}</Text>
        </View>
      ) : null}
    </View>
  )
}

function failureAffectsStep(failure: PairingDiagnosticStage, step: PairingDiagnosticStage) {
  if (failure === step) return true
  if (failure === 'PAIRING_SESSION_FAILED' && (step === 'PAIRING_SESSION_CREATING' || step === 'READY_TO_PAIR')) return true
  if (failure === 'AUTHENTICATION_FAILED') return step === 'AUTHENTICATION_READY' || step === 'READY_TO_PAIR'
  if (failure === 'CREDENTIAL_ISSUANCE_FAILED' || failure === 'CREDENTIAL_STORAGE_FAILED' || failure === 'ASSIGNMENT_FAILED') return step === 'CLAIM_IN_PROGRESS' || step === 'PAIRED'
  return false
}

function formatDiagnosticReport(failure: PairingDiagnosticEvent, buildInfo: SafeBuildInfo) {
  return [
    'Reflexion pairing diagnostic',
    `Failure: ${failure.userMessage}`,
    `Support code: ${failure.supportCode}`,
    `Occurred: ${failure.at}`,
    `Mirror version: ${buildInfo.appVersion} (build ${buildInfo.buildNumber})`,
    `Environment: ${buildInfo.environment}`,
    `Stage: ${failure.stage}`,
    `Reason: ${failure.reasonCode || 'UNKNOWN'}`,
    ...(failure.httpStatus ? [`HTTP status: ${failure.httpStatus}`] : []),
  ].join('\n')
}

function BuildSummary({ buildInfo }: { buildInfo: SafeBuildInfo }) {
  return <Text style={styles.detailsText}>Mirror {buildInfo.appVersion} · build {buildInfo.buildNumber}</Text>
}

/**
 * Shown INSTEAD of starting a conversation when the self-check found a blocking problem. Previously a
 * mirror with a denied microphone went straight into a check-in and silently recorded nothing; the elder
 * saw a normal conversation that simply "did not work", and the caregiver saw a missed check-in rather
 * than a one-tap fix. Same visual language as the offline screen: one large calm line, one supporting
 * sentence, one action.
 */
function NotReadyScreen({ verdict, onAction, onRetry }: { verdict: ReadinessVerdict; onAction: () => void; onRetry: () => void }) {
  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.stage}>
        <View pointerEvents="none" style={styles.reflection} />
        <View style={styles.offlineScene}>
          <View style={styles.offlineIcon}>
            <MirrorIcon name={verdict.problem === 'microphone' ? 'mic-off-outline' : 'construct-outline'} size={44} color={palette.linen} />
          </View>
          <Text style={styles.offlineTitle}>{verdict.title}</Text>
          <Text style={styles.offlineText}>{verdict.body}</Text>
          <Text style={styles.offlineNote}>Your caregiver can help with this.</Text>
          <Pressable onPress={verdict.actionLabel ? onAction : onRetry} style={styles.retryButton}>
            <Text style={styles.retryText}>{verdict.actionLabel ?? 'Check again'}</Text>
          </Pressable>
        </View>
      </View>
    </SafeAreaView>
  )
}

function OfflineHomeScreen({ onRetry }: { onRetry: () => void }) {
  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.stage}>
        <View pointerEvents="none" style={styles.reflection} />
        <View style={styles.offlineScene}>
          <View style={styles.offlineIcon}>
            <MirrorIcon name="cloud-offline-outline" size={44} color={palette.linen} />
          </View>
          <Text style={styles.offlineTitle}>Reflexion is offline right now.</Text>
          <Text style={styles.offlineText}>Saved check-ins will update your caregiver once the mirror is connected again.</Text>
          <Text style={styles.offlineNote}>Nothing already recorded will be lost.</Text>
          <Pressable onPress={onRetry} style={styles.retryButton}><Text style={styles.retryText}>Try connection again</Text></Pressable>
          {networkSetupAvailable() ? (
            <Pressable onPress={() => router.push('/network-setup')} style={styles.demoLink}>
              <Text style={styles.demoLinkText}>Set up the internet connection</Text>
            </Pressable>
          ) : null}
        </View>
      </View>
    </SafeAreaView>
  )
}

function PhoneSetupScreen({ state }: { state: SetupModeState }) {
  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.stage}>
        <View pointerEvents="none" style={styles.reflection} />
        <View style={styles.setupScene}>
          <PhoneSetupInstructions state={state} />
        </View>
      </View>
    </SafeAreaView>
  )
}

function BrandMark() {
  return (
    <View style={styles.brandMark}>
      <View style={[styles.seed, styles.seedLeft]} />
      <View style={[styles.seed, styles.seedCenter]} />
      <View style={[styles.seed, styles.seedRight]} />
    </View>
  )
}

function LoadingDots() {
  const pulse = useRef(new Animated.Value(0)).current
  useEffect(() => {
    const loop = Animated.loop(Animated.sequence([
      Animated.timing(pulse, { duration: 700, easing: Easing.inOut(Easing.quad), toValue: 1, useNativeDriver: true }),
      Animated.timing(pulse, { duration: 700, easing: Easing.inOut(Easing.quad), toValue: 0, useNativeDriver: true }),
    ]))
    loop.start()
    return () => loop.stop()
  }, [pulse])
  return (
    <View style={styles.dots}>
      {[0, 1, 2].map((item) => (
        <Animated.View
          key={item}
          style={[styles.dot, { opacity: pulse.interpolate({ inputRange: [0, 0.5, 1], outputRange: item === 1 ? [0.35, 1, 0.35] : [0.7, 0.35, 0.7] }) }]}
        />
      ))}
    </View>
  )
}

function QrCode({ value }: { value: string }) {
  const grid = useMemo(() => {
    const qr = qrcode(0, 'M')
    qr.addData(value)
    qr.make()
    const count = qr.getModuleCount()
    const rows: boolean[][] = []
    for (let rowIndex = 0; rowIndex < count; rowIndex += 1) {
      const row: boolean[] = []
      for (let columnIndex = 0; columnIndex < count; columnIndex += 1) row.push(qr.isDark(rowIndex, columnIndex))
      rows.push(row)
    }
    return { rows, cell: Math.max(4, Math.floor(180 / count)) }
  }, [value])
  return (
    <View style={styles.qr}>
      {grid.rows.map((row, rowIndex) => (
        <View key={rowIndex} style={styles.qrRow}>
          {row.map((dark, columnIndex) => (
            <View key={columnIndex} style={{ backgroundColor: dark ? palette.ink : 'transparent', height: grid.cell, width: grid.cell }} />
          ))}
        </View>
      ))}
    </View>
  )
}

function formatPairingCode(code: string) {
  const digits = code.replace(/\D/g, '').padEnd(6, '-').slice(0, 6)
  return `${digits.slice(0, 3)} ${digits.slice(3)}`
}

function formatExpiry(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return 'soon'
  return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
}

const styles = StyleSheet.create({
  safeArea: { backgroundColor: palette.cream, flex: 1 },
  stage: { backgroundColor: palette.cream, flex: 1, overflow: 'hidden' },
  reflection: { backgroundColor: 'rgba(231,207,166,0.22)', borderRadius: 300, height: 620, position: 'absolute', right: -310, top: -250, transform: [{ rotate: '-20deg' }], width: 510 },
  bootScene: { alignItems: 'center', flex: 1, justifyContent: 'center', paddingHorizontal: 36 },
  brandMark: { height: 88, position: 'relative', width: 116 },
  seed: { backgroundColor: palette.goldDeep, borderBottomLeftRadius: 30, borderBottomRightRadius: 5, borderTopLeftRadius: 5, borderTopRightRadius: 30, height: 62, position: 'absolute', top: 13, width: 28 },
  seedLeft: { left: 17, transform: [{ rotate: '-32deg' }] },
  seedCenter: { left: 44, top: 2, transform: [{ rotate: '44deg' }] },
  seedRight: { right: 17, transform: [{ rotate: '121deg' }] },
  brand: { color: palette.text, fontFamily: fonts.display, fontSize: 22, letterSpacing: 7, marginLeft: 7, marginTop: 20 },
  loadingText: { color: palette.textSecondary, fontFamily: fonts.body, fontSize: 25, marginTop: 54 },
  bootStatus: { bottom: 38, color: palette.textSecondary, fontFamily: fonts.body, fontSize: 13, opacity: 0.72, position: 'absolute' },
  dots: { flexDirection: 'row', gap: 12, marginTop: 30 },
  dot: { backgroundColor: palette.goldDeep, borderRadius: 4, height: 8, width: 8 },
  pairScene: { alignItems: 'center', flex: 1, justifyContent: 'flex-start', paddingBottom: 24, paddingHorizontal: 34, paddingTop: 46 },
  eyebrow: { color: palette.goldDeep, fontFamily: fonts.bodyMedium, fontSize: 12, letterSpacing: 2.2 },
  pairTitle: { color: palette.text, fontFamily: fonts.display, fontSize: 35, lineHeight: 44, marginTop: 12, maxWidth: 520, textAlign: 'center' },
  pairBody: { color: palette.textSecondary, fontFamily: fonts.body, fontSize: 19, lineHeight: 28, marginTop: 10, maxWidth: 450, textAlign: 'center' },
  checklist: { alignSelf: 'stretch', marginTop: 18, maxWidth: 460 },
  checklistRow: { alignItems: 'center', flexDirection: 'row', minHeight: 28 },
  checkIcon: { alignItems: 'center', borderColor: palette.lineWarm, borderRadius: 10, borderWidth: 1, height: 20, justifyContent: 'center', marginRight: 10, width: 20 },
  checkIconDone: { backgroundColor: palette.sageDeep, borderColor: palette.sageDeep },
  checkIconFailed: { backgroundColor: palette.coral, borderColor: palette.coral },
  checkIconPending: { backgroundColor: palette.lineWarm, borderRadius: 4, height: 7, width: 7 },
  checkLabel: { color: palette.textSecondary, fontFamily: fonts.body, fontSize: 14 },
  checkLabelDone: { color: palette.sageDeep, fontFamily: fonts.bodyMedium },
  checkLabelFailed: { color: palette.coral, fontFamily: fonts.bodyMedium, flex: 1 },
  pairingLabel: { color: palette.goldDeep, fontFamily: fonts.bodyMedium, fontSize: 11, letterSpacing: 1.8, marginTop: 26 },
  pairingCode: { color: palette.text, fontFamily: fonts.display, fontSize: 48, fontVariant: ['tabular-nums'], letterSpacing: 4, lineHeight: 60, marginBottom: 15 },
  qr: { alignItems: 'center', backgroundColor: palette.white, borderColor: palette.lineWarm, borderRadius: 18, borderWidth: 1, justifyContent: 'center', padding: 15, shadowColor: palette.shadow, shadowOpacity: 0.16, shadowRadius: 16 },
  qrRow: { flexDirection: 'row' },
  scanText: { color: palette.text, fontFamily: fonts.bodyMedium, fontSize: 16, lineHeight: 23, marginTop: 16, maxWidth: 400, textAlign: 'center' },
  waitingText: { color: palette.textSecondary, fontFamily: fonts.body, fontSize: 13, marginTop: 7 },
  backToReadiness: { alignItems: 'center', flexDirection: 'row', gap: 3, marginBottom: 4, padding: 6 },
  refreshCodeButton: { alignItems: 'center', borderColor: palette.sageDeep, borderRadius: 24, borderWidth: 1, flexDirection: 'row', gap: 8, marginTop: 17, paddingHorizontal: 22, paddingVertical: 10 },
  readyCard: { alignItems: 'center', alignSelf: 'stretch', backgroundColor: 'rgba(255,255,255,0.72)', borderColor: palette.sageDeep, borderRadius: 18, borderWidth: 1, flexDirection: 'row', flexWrap: 'wrap', gap: 15, marginTop: 18, maxWidth: 560, padding: 20 },
  readyIcon: { alignItems: 'center', backgroundColor: palette.beige, borderRadius: 38, height: 76, justifyContent: 'center', width: 76 },
  readyCopy: { flex: 1, minWidth: 180 },
  readyTitle: { color: palette.text, fontFamily: fonts.display, fontSize: 27 },
  readyBody: { color: palette.textSecondary, fontFamily: fonts.body, fontSize: 15, lineHeight: 21, marginTop: 4 },
  showCodeButton: { alignItems: 'center', backgroundColor: palette.sageDeep, borderRadius: 24, flexDirection: 'row', gap: 8, justifyContent: 'center', minWidth: 235, paddingHorizontal: 20, paddingVertical: 13 },
  errorText: { color: palette.coral, fontFamily: fonts.body, fontSize: 14, lineHeight: 20, marginTop: 8, maxWidth: 470, textAlign: 'center' },
  detailsLink: { marginTop: 4, padding: 6 },
  detailsCard: { backgroundColor: 'rgba(255,255,255,0.72)', borderColor: palette.lineWarm, borderRadius: 12, borderWidth: 1, marginTop: 6, maxWidth: 470, padding: 12, width: '100%' },
  detailsText: { color: palette.textSecondary, fontFamily: fonts.body, fontSize: 12, lineHeight: 18, textAlign: 'center' },
  detailsActions: { alignItems: 'center', flexDirection: 'row', flexWrap: 'wrap', gap: 10, justifyContent: 'center', marginTop: 8 },
  smallAction: { borderColor: palette.lineWarm, borderRadius: 16, borderWidth: 1, paddingHorizontal: 12, paddingVertical: 7 },
  smallActionText: { color: palette.text, fontFamily: fonts.bodyMedium, fontSize: 12 },
  demoLink: { borderBottomColor: palette.lineWarm, borderBottomWidth: 1, marginTop: 12, padding: 6 },
  demoLinkText: { color: palette.textSecondary, fontFamily: fonts.body, fontSize: 13 },
  offlineScene: { alignItems: 'center', flex: 1, justifyContent: 'center', paddingHorizontal: 38 },
  setupScene: { alignItems: 'center', flex: 1, justifyContent: 'flex-start', paddingHorizontal: 28, paddingTop: 34 },
  offlineIcon: { alignItems: 'center', backgroundColor: 'rgba(231,207,166,0.24)', borderColor: palette.lineWarm, borderRadius: 46, borderWidth: 1, height: 92, justifyContent: 'center', width: 92 },
  offlineTitle: { color: palette.text, fontFamily: fonts.display, fontSize: 37, lineHeight: 48, marginTop: 30, textAlign: 'center' },
  offlineText: { color: palette.textSecondary, fontFamily: fonts.body, fontSize: 21, lineHeight: 31, marginTop: 20, maxWidth: 480, textAlign: 'center' },
  offlineNote: { color: palette.sageDeep, fontFamily: fonts.bodyMedium, fontSize: 16, lineHeight: 24, marginTop: 12, maxWidth: 450, textAlign: 'center' },
  retryButton: { backgroundColor: palette.text, borderRadius: 26, marginTop: 28, paddingHorizontal: 24, paddingVertical: 14 },
  retryText: { color: palette.cream, fontFamily: fonts.bodyMedium, fontSize: 15 },
})
