import AsyncStorage from '@react-native-async-storage/async-storage'
import { Ionicons } from '@expo/vector-icons'
import { router } from 'expo-router'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Animated, Easing, Platform, Pressable, StyleSheet, Text, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import qrcode from 'qrcode-generator'

import { getApiUrl } from '../src/config/apiUrl'
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
  getBootstrapCredential,
  getDeviceCredential,
} from '../src/storage/deviceCredentials'
import { mirrorColors as palette, mirrorFonts as fonts } from '../src/theme/mirrorTheme'

type BootCheck = { key: string; label: string; ok: boolean }
type PairingDetails = {
  deviceId: string
  pairingId: string
  pairingCode: string
  qrPayload: string
  expiresAt?: string
}

const INSTALLER_SETUP_ENABLED = __DEV__ || process.env.EXPO_PUBLIC_ENABLE_INSTALLER_SETUP === 'true'

export default function BootScreen() {
  const [booting, setBooting] = useState(true)
  const [checks, setChecks] = useState<BootCheck[]>([])
  const [pairing, setPairing] = useState<PairingDetails | null>(null)
  const [pairingError, setPairingError] = useState('')
  const [offlineHome, setOfflineHome] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    let mounted = true
    async function boot() {
      const result = await runBootChecks()
      if (!mounted) return
      setChecks(result.checks)
      if (result.paired && !result.online) {
        setOfflineHome(true)
        setBooting(false)
        return
      }
      if (result.paired && result.online) {
        try {
          await refreshAndPersistDeviceProfile(result.deviceId)
          router.replace('/conversation')
          return
        } catch { /* revoked credentials fall through to pairing */ }
      }
      await loadPairingCode(result.deviceId)
      if (mounted) setBooting(false)
    }
    void boot()
    return () => {
      mounted = false
      if (pollRef.current) clearInterval(pollRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function loadPairingCode(knownDeviceId = '') {
    const details = await requestPairingCode()
    if (details) {
      setPairing(details)
      setPairingError('')
      return
    }
    setPairing(null)
    setPairingError(knownDeviceId
      ? 'Unable to reach the pairing service. Check the connection and try again.'
      : 'This mirror has not been provisioned with a device credential.')
  }

  useEffect(() => {
    if (!pairing) return
    const activePairing = pairing
    async function pollStatus() {
      try {
        const status = await getDevicePairing(activePairing.pairingId)
        if (status.state === 'expired' || status.state === 'cancelled') {
          if (pollRef.current) clearInterval(pollRef.current)
          await loadPairingCode(activePairing.deviceId)
          return
        }
        if (status.state !== 'paired' || !status.exchangeTicket) return
        await exchangeDeviceCredential(status)
        await refreshAndPersistDeviceProfile(activePairing.deviceId)
        if (pollRef.current) clearInterval(pollRef.current)
        router.replace('/conversation')
      } catch { /* polling resumes when connectivity returns */ }
    }
    void pollStatus()
    pollRef.current = setInterval(() => { void pollStatus() }, 3500)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pairing?.pairingId])

  if (booting) return <BootLoadingScreen checks={checks} />
  if (offlineHome) return <OfflineHomeScreen onRetry={() => router.replace('/conversation')} />
  return <PairingScreen error={pairingError} onRetry={() => void loadPairingCode()} pairing={pairing} />
}

async function runBootChecks() {
  const [bootstrap, credential, savedDeviceId] = await Promise.all([
    getBootstrapCredential(),
    getDeviceCredential(),
    AsyncStorage.getItem(DEVICE_ID_STORAGE_KEY),
  ])
  const deviceId = credential?.deviceId || bootstrap?.deviceId || savedDeviceId || ''
  const paired = Boolean(credential && credential.deviceId === deviceId)
  if (!paired) await clearLegacyPairingState()
  const online = typeof navigator === 'undefined' ? true : navigator.onLine !== false
  const backendReachable = online ? await pingBackend() : false
  const microphoneGranted = await checkMicrophonePermission()
  const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone
  if (timezone) await AsyncStorage.setItem(MIRROR_TIMEZONE_STORAGE_KEY, timezone)
  return {
    deviceId,
    online: online && backendReachable,
    paired,
    checks: [
      { key: 'device', label: 'Device provisioned', ok: Boolean(bootstrap?.token || credential) },
      { key: 'paired', label: 'Caregiver paired', ok: paired },
      { key: 'internet', label: 'Internet connected', ok: online },
      { key: 'backend', label: 'Reflexion service reachable', ok: backendReachable },
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

async function pingBackend() {
  try {
    const response = await fetch(getApiUrl('/health'))
    return response.ok
  } catch {
    return false
  }
}

async function checkMicrophonePermission() {
  if (Platform.OS !== 'web') return true
  if (!navigator.mediaDevices?.getUserMedia) return false
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
    stream.getTracks().forEach((track) => track.stop())
    return true
  } catch {
    return false
  }
}

async function requestPairingCode(): Promise<PairingDetails | null> {
  try {
    const bootstrap = await getBootstrapCredential()
    if (!bootstrap) return null
    const body = await createDevicePairing()
    return {
      deviceId: bootstrap.deviceId,
      pairingId: body.pairingId,
      pairingCode: body.displayCode,
      qrPayload: JSON.stringify({
        type: 'reflexion_device_pairing_v2',
        pairingId: body.pairingId,
        pairingCode: body.displayCode,
      }),
      expiresAt: body.expiresAt,
    }
  } catch {
    return null
  }
}

async function refreshAndPersistDeviceProfile(deviceId: string) {
  const response = await deviceFetch(`/api/v1/devices/${encodeURIComponent(deviceId)}/configuration`)
  const configuration = await dataOrThrow<DeviceConfiguration>(response)
  if (!configuration.patient) throw new Error('paired_patient_configuration_missing')
  await AsyncStorage.multiSet([
    [DEVICE_ID_STORAGE_KEY, deviceId],
    [ACTIVE_MIRROR_ID_STORAGE_KEY, deviceId],
    [ACTIVE_PATIENT_ID_STORAGE_KEY, configuration.patient.patientId],
    [NURSE_PATIENT_CONFIG_STORAGE_KEY, JSON.stringify({ patient: configuration.patient })],
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

export function PairingScreen({ error, onRetry, pairing }: { error: string; onRetry: () => void; pairing: PairingDetails | null }) {
  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.stage}>
        <View pointerEvents="none" style={styles.reflection} />
        <View style={styles.pairScene}>
          <Text style={styles.eyebrow}>SET UP REFLEXION</Text>
          <Text style={styles.pairTitle}>{pairing ? 'This mirror is not linked yet.' : 'Pairing is unavailable.'}</Text>
          <Text style={styles.pairBody}>{pairing
            ? 'Pair this mirror using the caregiver app.'
            : 'Check the mirror’s connection, then try again.'}</Text>
          {pairing ? (
            <>
              <Text style={styles.pairingLabel}>PAIRING CODE</Text>
              <Text style={styles.pairingCode}>{formatPairingCode(pairing.pairingCode)}</Text>
              <QrCode value={pairing.qrPayload} />
              <Text style={styles.scanText}>Scan with the caregiver app to pair this mirror.</Text>
              <Text style={styles.waitingText}>Waiting securely for pairing…</Text>
            </>
          ) : (
            <View style={styles.offlineIcon}>
              <Ionicons name="cloud-offline-outline" size={44} color={palette.linen} />
            </View>
          )}
          {error ? <Text style={styles.errorText}>{error}</Text> : null}
          {!pairing ? <Pressable onPress={onRetry} style={styles.retryButton}><Text style={styles.retryText}>Try again</Text></Pressable> : null}
          {!pairing && INSTALLER_SETUP_ENABLED ? (
            <Pressable onPress={() => router.push('/test-device')} style={styles.demoLink}>
              <Text style={styles.demoLinkText}>Installer setup</Text>
            </Pressable>
          ) : null}
          {__DEV__ && !pairing ? (
            <Pressable onPress={() => router.push('/realtime-test')} style={styles.demoLink}>
              <Text style={styles.demoLinkText}>Open developer conversation</Text>
            </Pressable>
          ) : null}
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
            <Ionicons name="cloud-offline-outline" size={44} color={palette.linen} />
          </View>
          <Text style={styles.offlineTitle}>Reflexion is offline right now.</Text>
          <Text style={styles.offlineText}>Saved check-ins will update your caregiver once the mirror is connected again.</Text>
          <Text style={styles.offlineNote}>Nothing already recorded will be lost.</Text>
          <Pressable onPress={onRetry} style={styles.retryButton}><Text style={styles.retryText}>Try connection again</Text></Pressable>
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
  pairingLabel: { color: palette.goldDeep, fontFamily: fonts.bodyMedium, fontSize: 11, letterSpacing: 1.8, marginTop: 26 },
  pairingCode: { color: palette.text, fontFamily: fonts.display, fontSize: 48, fontVariant: ['tabular-nums'], letterSpacing: 4, lineHeight: 60, marginBottom: 15 },
  qr: { alignItems: 'center', backgroundColor: palette.white, borderColor: palette.lineWarm, borderRadius: 18, borderWidth: 1, justifyContent: 'center', padding: 15, shadowColor: palette.shadow, shadowOpacity: 0.16, shadowRadius: 16 },
  qrRow: { flexDirection: 'row' },
  scanText: { color: palette.text, fontFamily: fonts.bodyMedium, fontSize: 16, lineHeight: 23, marginTop: 16, maxWidth: 400, textAlign: 'center' },
  waitingText: { color: palette.textSecondary, fontFamily: fonts.body, fontSize: 13, marginTop: 7 },
  errorText: { color: palette.coral, fontFamily: fonts.body, fontSize: 14, lineHeight: 20, marginTop: 8, maxWidth: 470, textAlign: 'center' },
  demoLink: { borderBottomColor: palette.lineWarm, borderBottomWidth: 1, marginTop: 12, padding: 6 },
  demoLinkText: { color: palette.textSecondary, fontFamily: fonts.body, fontSize: 13 },
  offlineScene: { alignItems: 'center', flex: 1, justifyContent: 'center', paddingHorizontal: 38 },
  offlineIcon: { alignItems: 'center', backgroundColor: 'rgba(231,207,166,0.24)', borderColor: palette.lineWarm, borderRadius: 46, borderWidth: 1, height: 92, justifyContent: 'center', width: 92 },
  offlineTitle: { color: palette.text, fontFamily: fonts.display, fontSize: 37, lineHeight: 48, marginTop: 30, textAlign: 'center' },
  offlineText: { color: palette.textSecondary, fontFamily: fonts.body, fontSize: 21, lineHeight: 31, marginTop: 20, maxWidth: 480, textAlign: 'center' },
  offlineNote: { color: palette.sageDeep, fontFamily: fonts.bodyMedium, fontSize: 16, lineHeight: 24, marginTop: 12, maxWidth: 450, textAlign: 'center' },
  retryButton: { backgroundColor: palette.text, borderRadius: 26, marginTop: 28, paddingHorizontal: 24, paddingVertical: 14 },
  retryText: { color: palette.cream, fontFamily: fonts.bodyMedium, fontSize: 15 },
})
