import AsyncStorage from '@react-native-async-storage/async-storage'
import { Ionicons } from '@expo/vector-icons'
import { router } from 'expo-router'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Animated, Easing, Platform, StyleSheet, Text, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'

import { getApiUrl } from './apiUrl'
import {
  ACTIVE_MIRROR_ID_STORAGE_KEY,
  ACTIVE_NURSE_ID_STORAGE_KEY,
  ACTIVE_PATIENT_ID_STORAGE_KEY,
  DEVICE_AUTH_TOKEN_STORAGE_KEY,
  DEVICE_ID_STORAGE_KEY,
  MIRROR_LANGUAGE_STORAGE_KEY,
  MIRROR_TIMEZONE_STORAGE_KEY,
  NURSE_PATIENT_CONFIG_STORAGE_KEY,
} from '../src/constants/nursePatientConfig'
import { loadJson, persistNursePatientIds } from '../src/storage/mirrorStorage'
import { generateMirrorId } from '../src/utils/id'

type BootCheck = {
  key: string
  label: string
  ok: boolean
}

type PairingDetails = {
  deviceId: string
  pairingCode: string
  qrPayload: string
  expiresAt?: string
}

type PairingStatusResponse =
  | {
      success: true
      paired: true
      deviceId: string
      authToken: string
      nurseId?: string
      patientId?: string
      language?: string
      timezone?: string
      nursePatientConfig: unknown
    }
  | { success: true; paired: false }
  | { success: false; reason: string }

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

      if (result.paired && result.online) {
        router.replace('/conversation')
        return
      }
      if (result.paired && !result.online) {
        setOfflineHome(true)
        setBooting(false)
        return
      }

      if (result.online) {
        const serverStatus = await getDevicePairingStatus(result.deviceId)
        if (!mounted) return
        if (serverStatus.success && serverStatus.paired) {
          await persistPairedMirror(serverStatus)
          router.replace('/conversation')
          return
        }
      }

      const details = await requestPairingCode(result.deviceId)
      if (!mounted) return
      if (details) {
        setPairing(details)
        setPairingError('')
      } else {
        setPairingError('Unable to request a pairing code. Check internet connection and backend availability.')
      }
      setBooting(false)
    }

    void boot()
    return () => {
      mounted = false
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  useEffect(() => {
    if (!pairing) return
    const activePairing = pairing

    async function pollStatus() {
      const status = await getPairingStatus(activePairing.deviceId, activePairing.pairingCode)
      if (!status.success || !status.paired) return

      await persistPairedMirror(status)
      if (pollRef.current) clearInterval(pollRef.current)
      router.replace('/conversation')
    }

    void pollStatus()
    pollRef.current = setInterval(() => {
      void pollStatus()
    }, 3500)

    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [pairing])

  if (booting) {
    return <BootLoadingScreen checks={checks} />
  }

  if (offlineHome) {
    return <OfflineHomeScreen />
  }

  return <PairingScreen error={pairingError} pairing={pairing} />
}

async function runBootChecks() {
  const savedDeviceId = await AsyncStorage.getItem(DEVICE_ID_STORAGE_KEY)
  const deviceId = savedDeviceId || generateMirrorId()
  if (!savedDeviceId) {
    await AsyncStorage.setItem(DEVICE_ID_STORAGE_KEY, deviceId)
  }

  const [storedConfig, activeMirrorId] = await Promise.all([
    loadJson<unknown>(NURSE_PATIENT_CONFIG_STORAGE_KEY),
    AsyncStorage.getItem(ACTIVE_MIRROR_ID_STORAGE_KEY),
  ])
  const authToken = await AsyncStorage.getItem(DEVICE_AUTH_TOKEN_STORAGE_KEY)
  const paired = Boolean(
    authToken &&
      storedConfig &&
      activeMirrorId &&
      activeMirrorId === deviceId &&
      hasPatientForMirror(storedConfig, deviceId),
  )
  if (!paired) {
    await clearLegacyPairingState()
  }
  const online = typeof navigator === 'undefined' ? true : navigator.onLine !== false
  const backendReachable = online ? await pingBackend() : false
  const microphoneGranted = await checkMicrophonePermission()
  const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone
  if (timezone) {
    await AsyncStorage.setItem(MIRROR_TIMEZONE_STORAGE_KEY, timezone)
  }

  return {
    deviceId,
    online: online && backendReachable,
    paired,
    checks: [
      { key: 'device', label: 'Device ID exists', ok: Boolean(deviceId) },
      { key: 'paired', label: 'Paired to a user', ok: paired },
      { key: 'internet', label: 'Internet connected', ok: online },
      { key: 'backend', label: 'Backend reachable', ok: backendReachable },
      { key: 'microphone', label: 'Microphone permission granted', ok: microphoneGranted },
      { key: 'speaker', label: 'Speaker available', ok: true },
      { key: 'timezone', label: 'Current time/timezone correct', ok: Boolean(timezone) },
    ],
  }
}

async function clearLegacyPairingState() {
  await AsyncStorage.multiRemove([
    ACTIVE_MIRROR_ID_STORAGE_KEY,
    ACTIVE_NURSE_ID_STORAGE_KEY,
    ACTIVE_PATIENT_ID_STORAGE_KEY,
    DEVICE_AUTH_TOKEN_STORAGE_KEY,
    NURSE_PATIENT_CONFIG_STORAGE_KEY,
    MIRROR_LANGUAGE_STORAGE_KEY,
  ])
}

function hasPatientForMirror(config: unknown, mirrorId: string) {
  if (!config || typeof config !== 'object') return false
  const patients = (config as { patients?: Array<{ mirrorId?: unknown }> }).patients
  return Boolean(
    patients?.some((patient) => {
      const id = patient.mirrorId
      if (typeof id === 'string') return id === mirrorId
      if (id && typeof id === 'object' && '$oid' in id) {
        return (id as { $oid?: string }).$oid === mirrorId
      }
      return false
    }),
  )
}

async function pingBackend() {
  try {
    const response = await fetch(getApiUrl('/api/mirror-pairing/request-code'), {
      method: 'OPTIONS',
    })
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

async function requestPairingCode(deviceId: string): Promise<PairingDetails | null> {
  try {
    const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone
    const response = await fetch(getApiUrl('/api/mirror-pairing/request-code'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ deviceId, timezone }),
    })
    const body = await response.json()
    if (!response.ok || !body?.success) return null
    return {
      deviceId: body.deviceId,
      pairingCode: body.pairingCode,
      qrPayload: body.qrPayload,
      expiresAt: body.expiresAt,
    }
  } catch {
    return null
  }
}

async function getPairingStatus(deviceId: string, pairingCode: string): Promise<PairingStatusResponse> {
  try {
    const response = await fetch(getApiUrl('/api/mirror-pairing/status'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ deviceId, pairingCode }),
    })
    return (await response.json()) as PairingStatusResponse
  } catch {
    return { success: false, reason: 'network_error' }
  }
}

async function getDevicePairingStatus(deviceId: string): Promise<PairingStatusResponse> {
  try {
    const response = await fetch(getApiUrl('/api/mirror-pairing/device-status'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ deviceId }),
    })
    return (await response.json()) as PairingStatusResponse
  } catch {
    return { success: false, reason: 'network_error' }
  }
}

async function persistPairedMirror(status: Extract<PairingStatusResponse, { paired: true }>) {
  await AsyncStorage.multiSet([
    [DEVICE_ID_STORAGE_KEY, status.deviceId],
    [ACTIVE_MIRROR_ID_STORAGE_KEY, status.deviceId],
    [DEVICE_AUTH_TOKEN_STORAGE_KEY, status.authToken],
    [NURSE_PATIENT_CONFIG_STORAGE_KEY, JSON.stringify(status.nursePatientConfig)],
    [MIRROR_LANGUAGE_STORAGE_KEY, status.language || 'english'],
    [MIRROR_TIMEZONE_STORAGE_KEY, status.timezone || Intl.DateTimeFormat().resolvedOptions().timeZone],
  ])
  if (status.nurseId && status.patientId) {
    await AsyncStorage.multiSet([
      [ACTIVE_NURSE_ID_STORAGE_KEY, status.nurseId],
      [ACTIVE_PATIENT_ID_STORAGE_KEY, status.patientId],
    ])
  } else {
    await persistNursePatientIds(status.nursePatientConfig, status.deviceId)
  }
}

function BootLoadingScreen({ checks }: { checks: BootCheck[] }) {
  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.stage}>
        <View style={styles.card}>
          <LotusLogo />
          <Text style={styles.brand}>REFLEXION</Text>
          <Text style={styles.loadingText}>Starting Reflexion...</Text>
          <LoadingDots />
          <View style={styles.checkList}>
            {checks.map((check) => (
              <View key={check.key} style={styles.checkRow}>
                <Ionicons
                  name={check.ok ? 'checkmark-circle' : 'ellipse-outline'}
                  size={15}
                  color={check.ok ? colors.sage : colors.taupe}
                />
                <Text style={styles.checkText}>{check.label}</Text>
              </View>
            ))}
          </View>
        </View>
      </View>
    </SafeAreaView>
  )
}

function PairingScreen({
  error,
  pairing,
}: {
  error: string
  pairing: PairingDetails | null
}) {
  const code = pairing?.pairingCode ? formatPairingCode(pairing.pairingCode) : '--- ---'

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.stage}>
        <View style={styles.card}>
          <Text style={styles.pairTitle}>This mirror is{'\n'}not linked yet.</Text>
          <Text style={styles.pairBody}>Pair this mirror using{'\n'}the caregiver app.</Text>
          <Text style={styles.pairingLabel}>Pairing code</Text>
          <Text style={styles.pairingCode}>{code}</Text>
          <QrCode value={pairing?.qrPayload || 'reflexion:pairing:pending'} />
          <Text style={styles.scanText}>Scan with caregiver app{'\n'}to pair this mirror.</Text>
          {error ? <Text style={styles.errorText}>{error}</Text> : null}
        </View>
      </View>
    </SafeAreaView>
  )
}

function OfflineHomeScreen() {
  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.stage}>
        <View style={styles.card}>
          <View style={styles.offlineIcon}>
            <Ionicons name="cloud-offline-outline" size={48} color={colors.goldDark} />
          </View>
          <Text style={styles.offlineTitle}>Reflexion is offline right now.</Text>
          <Text style={styles.offlineText}>You can still complete your check-in.</Text>
          <Text style={styles.offlineText}>I’ll update your caregiver once we’re connected again.</Text>
          <Text style={styles.offlineNote}>Completed check-ins are queued locally and uploaded later.</Text>
        </View>
      </View>
    </SafeAreaView>
  )
}

function LotusLogo() {
  return (
    <View style={styles.logo}>
      <View style={[styles.petal, styles.petalLeft]} />
      <View style={[styles.petal, styles.petalCenter]} />
      <View style={[styles.petal, styles.petalRight]} />
      <View style={styles.logoBase} />
    </View>
  )
}

function LoadingDots() {
  const pulse = useRef(new Animated.Value(0)).current

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, {
          duration: 700,
          easing: Easing.inOut(Easing.quad),
          toValue: 1,
          useNativeDriver: true,
        }),
        Animated.timing(pulse, {
          duration: 700,
          easing: Easing.inOut(Easing.quad),
          toValue: 0,
          useNativeDriver: true,
        }),
      ]),
    )
    loop.start()
    return () => loop.stop()
  }, [pulse])

  return (
    <View style={styles.dots}>
      {[0, 1, 2].map((item) => (
        <Animated.View
          key={item}
          style={[
            styles.dot,
            {
              opacity: pulse.interpolate({
                inputRange: [0, 0.5, 1],
                outputRange: item === 1 ? [0.35, 1, 0.35] : [0.7, 0.35, 0.7],
              }),
            },
          ]}
        />
      ))}
    </View>
  )
}

function QrCode({ value }: { value: string }) {
  const cells = useMemo(() => makeQrCells(value), [value])
  return (
    <View style={styles.qr}>
      {cells.map((filled, index) => (
        <View key={index} style={[styles.qrCell, filled && styles.qrCellFilled]} />
      ))}
    </View>
  )
}

function makeQrCells(value: string) {
  let seed = 0
  for (let index = 0; index < value.length; index += 1) {
    seed = (seed * 31 + value.charCodeAt(index)) >>> 0
  }

  const size = 17
  const cells = Array.from({ length: size * size }, (_, index) => {
    const row = Math.floor(index / size)
    const col = index % size
    const finder =
      (row < 5 && col < 5) ||
      (row < 5 && col >= size - 5) ||
      (row >= size - 5 && col < 5)
    if (finder) {
      const localRow = row < 5 ? row : row - (size - 5)
      const localCol = col < 5 ? col : col - (size - 5)
      return localRow === 0 || localRow === 4 || localCol === 0 || localCol === 4 || (localRow === 2 && localCol === 2)
    }
    seed = (seed * 1664525 + 1013904223) >>> 0
    return seed % 3 !== 0
  })
  return cells
}

function formatPairingCode(code: string) {
  const digits = code.replace(/\D/g, '').padEnd(6, '-').slice(0, 6)
  return `${digits.slice(0, 3)} ${digits.slice(3)}`
}

const colors = {
  background: '#FFF9F1',
  card: '#FFFBF4',
  line: '#F1E5D2',
  gold: '#E7CFA6',
  goldDark: '#C89755',
  sage: '#ABC5A1',
  taupe: '#BBAFA0',
  text: '#282828',
  secondary: '#686868',
  coral: '#C97068',
}

const styles = StyleSheet.create({
  safeArea: {
    backgroundColor: colors.background,
    flex: 1,
  },
  stage: {
    alignItems: 'center',
    backgroundColor: colors.background,
    flex: 1,
    justifyContent: 'center',
    padding: 18,
  },
  card: {
    alignItems: 'center',
    backgroundColor: colors.card,
    borderColor: colors.line,
    borderRadius: 8,
    borderWidth: 1,
    gap: 24,
    height: '100%',
    justifyContent: 'center',
    maxHeight: 760,
    maxWidth: 430,
    minHeight: 620,
    padding: 34,
    shadowColor: '#D8C6A8',
    shadowOffset: { height: 10, width: 0 },
    shadowOpacity: 0.22,
    shadowRadius: 22,
    width: '100%',
  },
  logo: {
    height: 120,
    marginBottom: 2,
    position: 'relative',
    width: 150,
  },
  petal: {
    backgroundColor: 'rgba(231, 207, 166, 0.72)',
    height: 78,
    position: 'absolute',
    top: 8,
    width: 42,
  },
  petalLeft: {
    borderBottomLeftRadius: 42,
    borderTopRightRadius: 42,
    left: 38,
    transform: [{ rotate: '-25deg' }],
  },
  petalCenter: {
    borderTopLeftRadius: 42,
    borderTopRightRadius: 42,
    left: 54,
  },
  petalRight: {
    borderBottomRightRadius: 42,
    borderTopLeftRadius: 42,
    left: 70,
    transform: [{ rotate: '25deg' }],
  },
  logoBase: {
    borderBottomColor: colors.gold,
    borderBottomWidth: 14,
    borderRadius: 80,
    bottom: 12,
    height: 54,
    left: 22,
    position: 'absolute',
    width: 106,
  },
  brand: {
    color: colors.text,
    fontSize: 34,
    fontWeight: '700',
    letterSpacing: 8,
  },
  loadingText: {
    color: colors.secondary,
    fontSize: 24,
    fontWeight: '800',
    marginTop: 42,
  },
  dots: {
    flexDirection: 'row',
    gap: 14,
    marginTop: 42,
  },
  dot: {
    backgroundColor: colors.goldDark,
    borderRadius: 5,
    height: 10,
    width: 10,
  },
  checkList: {
    bottom: 28,
    gap: 5,
    left: 30,
    position: 'absolute',
    right: 30,
  },
  checkRow: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: 7,
  },
  checkText: {
    color: colors.secondary,
    fontSize: 11,
    fontWeight: '700',
  },
  pairTitle: {
    color: colors.text,
    fontSize: 30,
    fontWeight: '900',
    lineHeight: 40,
    textAlign: 'center',
  },
  pairBody: {
    color: colors.secondary,
    fontSize: 22,
    fontWeight: '800',
    lineHeight: 31,
    textAlign: 'center',
  },
  pairingLabel: {
    color: colors.secondary,
    fontSize: 20,
    fontWeight: '800',
    marginTop: 4,
  },
  pairingCode: {
    color: colors.text,
    fontSize: 48,
    fontWeight: '500',
    letterSpacing: 2,
  },
  qr: {
    backgroundColor: '#FFFFFF',
    borderColor: colors.line,
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: 'row',
    flexWrap: 'wrap',
    height: 156,
    padding: 10,
    width: 156,
  },
  qrCell: {
    height: 8,
    width: 8,
  },
  qrCellFilled: {
    backgroundColor: colors.text,
  },
  scanText: {
    color: colors.secondary,
    fontSize: 18,
    fontWeight: '900',
    lineHeight: 26,
    textAlign: 'center',
  },
  errorText: {
    color: colors.coral,
    fontSize: 13,
    fontWeight: '800',
    lineHeight: 18,
    textAlign: 'center',
  },
  offlineIcon: {
    alignItems: 'center',
    backgroundColor: '#FFF4E4',
    borderRadius: 56,
    height: 106,
    justifyContent: 'center',
    width: 106,
  },
  offlineTitle: {
    color: colors.text,
    fontSize: 30,
    fontWeight: '900',
    lineHeight: 38,
    textAlign: 'center',
  },
  offlineText: {
    color: colors.secondary,
    fontSize: 22,
    fontWeight: '800',
    lineHeight: 32,
    textAlign: 'center',
  },
  offlineNote: {
    color: colors.secondary,
    fontSize: 14,
    fontWeight: '800',
    lineHeight: 20,
    marginTop: 8,
    textAlign: 'center',
  },
})
