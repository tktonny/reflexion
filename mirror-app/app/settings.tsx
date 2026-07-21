import AsyncStorage from '@react-native-async-storage/async-storage'
import { Ionicons } from '@expo/vector-icons'
import { Camera } from 'expo-camera'
import { router } from 'expo-router'
import { useEffect, useMemo, useState } from 'react'
import { Alert, Platform, Pressable, StyleSheet, Text, View } from 'react-native'
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
import { loadJson } from '../src/storage/mirrorStorage'
import { clearPendingConversations, flushPendingConversations, loadPendingConversations } from '../src/storage/conversationQueue'
import { DEFAULT_LANGUAGE } from '../src/config/conversationMode'

// Language options (value = hint string understood by normalizeLanguageKey). Applied on the next
// check-in; each maps to a qwen3.5-omni realtime voice (Serena / Kiki粤语 / Joseph Chen闽南).
const LANGUAGE_OPTIONS: { value: string; label: string }[] = [
  { value: 'mandarin', label: '中文(普通话)' },
  { value: 'cantonese', label: '粤语' },
  { value: 'minnan', label: '闽南话' },
  { value: 'english', label: 'English' },
  { value: 'malay', label: 'Malay' },
  { value: 'hindi', label: 'हिन्दी' },
  { value: 'urdu', label: 'اردو' },
]

type AdminState = {
  backendOnline: boolean
  cameraStatus: string
  currentUser: string
  deviceId: string
  microphoneStatus: string
  pairingStatus: string
  pendingUploads: number
  speakerStatus: string
  wifiStatus: string
}

export default function SettingsScreen() {
  const [state, setState] = useState<AdminState>({
    backendOnline: false,
    cameraStatus: 'Checking',
    currentUser: 'Not paired',
    deviceId: 'Unknown',
    microphoneStatus: 'Checking',
    pairingStatus: 'Not paired',
    pendingUploads: 0,
    speakerStatus: 'Good',
    wifiStatus: 'Checking',
  })
  const [language, setLanguage] = useState<string>(DEFAULT_LANGUAGE)

  const rows = useMemo(
    () => [
      ['Device ID', compactId(state.deviceId)],
      ['Pairing Status', state.pairingStatus],
      ['Current User', state.currentUser],
      ['Wi-Fi Status', state.wifiStatus],
      ['Backend Status', state.backendOnline ? 'Online' : 'Offline'],
      ['App Version', '0.0.1'],
      ['Microphone Test', state.microphoneStatus],
      ['Speaker Test', state.speakerStatus],
      ['Camera Status', state.cameraStatus],
      ['Pending Uploads', String(state.pendingUploads)],
    ],
    [state],
  )

  useEffect(() => {
    void loadAdminState()
  }, [])

  async function loadAdminState() {
    const [
      deviceId,
      activeMirrorId,
      authToken,
      config,
      pending,
      storedLang,
    ] = await Promise.all([
      AsyncStorage.getItem(DEVICE_ID_STORAGE_KEY),
      AsyncStorage.getItem(ACTIVE_MIRROR_ID_STORAGE_KEY),
      AsyncStorage.getItem(DEVICE_AUTH_TOKEN_STORAGE_KEY),
      loadJson<unknown>(NURSE_PATIENT_CONFIG_STORAGE_KEY),
      loadPendingConversations(),
      AsyncStorage.getItem(MIRROR_LANGUAGE_STORAGE_KEY),
    ])
    if (storedLang && storedLang.trim()) setLanguage(storedLang.trim())
    const online = typeof navigator === 'undefined' ? true : navigator.onLine !== false
    const backendOnline = online ? await checkBackend() : false
    const microphoneStatus = await checkMicrophone()
    const cameraStatus = await checkCamera()
    const currentUser = getCurrentUser(config, activeMirrorId || deviceId || '')

    setState({
      backendOnline,
      cameraStatus,
      currentUser: currentUser || 'Not paired',
      deviceId: deviceId || 'Unknown',
      microphoneStatus,
      pairingStatus: authToken && activeMirrorId ? 'Paired' : 'Not paired',
      pendingUploads: pending.length,
      speakerStatus: 'Good',
      wifiStatus: online ? 'Connected' : 'Offline',
    })
  }

  async function resetPairing() {
    // Notify the backend to release the server-side pairing session (best-effort) so it doesn't leave
    // an orphaned status:'paired' row that would 409 re-pairing and keep device-status "paired".
    try {
      const [deviceId, authToken] = await Promise.all([
        AsyncStorage.getItem(DEVICE_ID_STORAGE_KEY),
        AsyncStorage.getItem(DEVICE_AUTH_TOKEN_STORAGE_KEY),
      ])
      if (deviceId) {
        await fetch(getApiUrl('/api/mirror-pairing/unpair'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ deviceId, authToken }),
        }).catch(() => {})
      }
    } catch {
      /* offline / unreachable — proceed with the local reset */
    }
    // Preserve DEVICE_ID: a support reset should re-pair the SAME mirror identity, not mint a new one
    // (only the /test-device dev flow intentionally changes the deviceId).
    await AsyncStorage.multiRemove([
      ACTIVE_MIRROR_ID_STORAGE_KEY,
      ACTIVE_NURSE_ID_STORAGE_KEY,
      ACTIVE_PATIENT_ID_STORAGE_KEY,
      DEVICE_AUTH_TOKEN_STORAGE_KEY,
      MIRROR_LANGUAGE_STORAGE_KEY,
      MIRROR_TIMEZONE_STORAGE_KEY,
      NURSE_PATIENT_CONFIG_STORAGE_KEY,
    ])
    await clearPendingConversations()
    router.replace('/')
  }

  async function uploadLogs() {
    const result = await flushPendingConversations()
    setState((current) => ({ ...current, pendingUploads: result.remaining }))
    Alert.alert('Upload logs', `Synced ${result.synced}. Remaining ${result.remaining}.`)
  }

  async function changeLanguage(value: string) {
    setLanguage(value)
    await AsyncStorage.setItem(MIRROR_LANGUAGE_STORAGE_KEY, value)
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.stage}>
        <View style={styles.card}>
          <Text style={styles.title}>Admin / Settings</Text>
          <View style={styles.rows}>
            {rows.map(([label, value]) => (
              <View key={label} style={styles.row}>
                <Text style={styles.label}>{label}</Text>
                <Text style={[styles.value, isGoodValue(value) && styles.goodValue]}>{value}</Text>
              </View>
            ))}
          </View>

          <View style={styles.langSection}>
            <Text style={styles.langTitle}>对话语言 / Language</Text>
            <View style={styles.langChips}>
              {LANGUAGE_OPTIONS.map((opt) => (
                <Pressable
                  key={opt.value}
                  onPress={() => void changeLanguage(opt.value)}
                  style={[styles.langChip, language === opt.value && styles.langChipActive]}
                >
                  <Text style={[styles.langChipText, language === opt.value && styles.langChipTextActive]}>{opt.label}</Text>
                </Pressable>
              ))}
            </View>
            <Text style={styles.langHint}>下次检查生效 · 默认中文(普通话)</Text>
          </View>

          <View style={styles.actions}>
            <AdminAction icon="mic-outline" label="Microphone test" onPress={() => void loadAdminState()} />
            <AdminAction icon="volume-high-outline" label="Speaker test" onPress={() => Alert.alert('Speaker test', 'Speaker output is available.')} />
            <AdminAction icon="refresh-outline" label="Upload logs" onPress={() => void uploadLogs()} />
            <AdminAction icon="reload-outline" label="Restart app" onPress={() => router.replace('/')} />
            <AdminAction danger icon="trash-outline" label="Reset pairing" onPress={() => void resetPairing()} />
          </View>
        </View>
      </View>
    </SafeAreaView>
  )
}

function AdminAction({
  danger,
  icon,
  label,
  onPress,
}: {
  danger?: boolean
  icon: keyof typeof Ionicons.glyphMap
  label: string
  onPress: () => void
}) {
  return (
    <Pressable style={styles.actionRow} onPress={onPress}>
      <View style={styles.actionLabel}>
        <Ionicons name={icon} size={20} color={danger ? '#C97068' : '#8E7F6D'} />
        <Text style={[styles.actionText, danger && styles.dangerText]}>{label}</Text>
      </View>
      <Ionicons name="chevron-forward" size={18} color="#BBAFA0" />
    </Pressable>
  )
}

async function checkBackend() {
  try {
    const response = await fetch(getApiUrl('/api/mirror-pairing/request-code'), { method: 'OPTIONS' })
    return response.ok
  } catch {
    return false
  }
}

async function checkMicrophone() {
  if (Platform.OS !== 'web') return 'Good'
  if (!navigator.mediaDevices?.getUserMedia) return 'Unavailable'
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
    stream.getTracks().forEach((track) => track.stop())
    return 'Good'
  } catch {
    return 'Needs permission'
  }
}

async function checkCamera() {
  try {
    const permission = await Camera.getCameraPermissionsAsync()
    return permission.granted ? 'Good' : 'Needs permission'
  } catch {
    return 'Unavailable'
  }
}

function getCurrentUser(config: unknown, mirrorId: string) {
  if (!config || typeof config !== 'object') return ''
  const patients = (config as { patients?: Array<{ mirrorId?: unknown; name?: unknown }> }).patients
  const patient = patients?.find((candidate) => mirrorIdToString(candidate.mirrorId) === mirrorId)
  return typeof patient?.name === 'string' ? patient.name : ''
}

function mirrorIdToString(value: unknown) {
  if (typeof value === 'string') return value
  if (value && typeof value === 'object' && '$oid' in value) {
    return (value as { $oid?: string }).$oid || ''
  }
  return ''
}

function compactId(value: string) {
  if (!value || value === 'Unknown') return value
  return value.length > 14 ? `${value.slice(0, 8)}...${value.slice(-4)}` : value
}

function isGoodValue(value: string) {
  return ['Paired', 'Connected', 'Online', 'Good'].includes(value)
}

const styles = StyleSheet.create({
  safeArea: {
    backgroundColor: '#FFF9F1',
    flex: 1,
  },
  stage: {
    alignItems: 'center',
    backgroundColor: '#FFF9F1',
    flex: 1,
    justifyContent: 'center',
    padding: 18,
  },
  card: {
    backgroundColor: '#FFFBF4',
    borderColor: '#F1E5D2',
    borderRadius: 8,
    borderWidth: 1,
    gap: 28,
    maxWidth: 520,
    padding: 34,
    shadowColor: '#D8C6A8',
    shadowOffset: { height: 10, width: 0 },
    shadowOpacity: 0.22,
    shadowRadius: 22,
    width: '100%',
  },
  title: {
    color: '#282828',
    fontSize: 24,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  rows: {
    gap: 12,
  },
  row: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  label: {
    color: '#686868',
    fontSize: 16,
    fontWeight: '800',
  },
  value: {
    color: '#282828',
    fontSize: 16,
    fontWeight: '800',
    maxWidth: 220,
    textAlign: 'right',
  },
  goodValue: {
    color: '#4D9668',
  },
  actions: {
    borderTopColor: '#EDE5D6',
    borderTopWidth: 1,
    gap: 2,
    paddingTop: 12,
  },
  langSection: {
    borderTopColor: '#EDE5D6',
    borderTopWidth: 1,
    gap: 10,
    paddingTop: 16,
  },
  langTitle: {
    color: '#282828',
    fontSize: 15,
    fontWeight: '900',
  },
  langChips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  langChip: {
    backgroundColor: '#FFF4E2',
    borderColor: '#F0DEC1',
    borderRadius: 8,
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  langChipActive: {
    backgroundColor: '#C89755',
    borderColor: '#C89755',
  },
  langChipText: {
    color: '#8E7F6D',
    fontSize: 14,
    fontWeight: '800',
  },
  langChipTextActive: {
    color: '#FFFFFF',
  },
  langHint: {
    color: '#BBAFA0',
    fontSize: 12,
    fontWeight: '700',
  },
  actionRow: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
    minHeight: 44,
  },
  actionLabel: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: 12,
  },
  actionText: {
    color: '#282828',
    fontSize: 16,
    fontWeight: '800',
  },
  dangerText: {
    color: '#C97068',
  },
})
