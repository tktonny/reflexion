import AsyncStorage from '@react-native-async-storage/async-storage'
import { Ionicons } from '@expo/vector-icons'
import Constants from 'expo-constants'
import { router } from 'expo-router'
import { useEffect, useMemo, useState } from 'react'
import { ActivityIndicator, Alert, Platform, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'

import { getApiUrl } from '../src/config/apiUrl'
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
import { runHardwareChecks, type HardwareCheck } from '../src/lib/hardwareCheck'
import { clearPendingConversations, flushPendingConversations, loadPendingConversations } from '../src/storage/conversationQueue'
import { clearDeviceCredential, getDeviceCredential } from '../src/storage/deviceCredentials'
import { loadJson } from '../src/storage/mirrorStorage'
import { mirrorColors as c, mirrorFonts as f } from '../src/theme/mirrorTheme'

const LANGUAGE_OPTIONS: { value: string; label: string }[] = [
  { value: 'mandarin', label: '中文（普通话）' },
  { value: 'cantonese', label: '粤语' },
  { value: 'minnan', label: '闽南话' },
  { value: 'english', label: 'English' },
  { value: 'malay', label: 'Malay' },
  { value: 'hindi', label: 'हिन्दी' },
  { value: 'urdu', label: 'اردو' },
]

type AdminState = {
  backendOnline: boolean
  currentUser: string
  deviceId: string
  pairingStatus: string
  pendingUploads: number
  checks: HardwareCheck[]
}

const EMPTY_STATE: AdminState = {
  backendOnline: false,
  currentUser: 'Not paired',
  deviceId: 'Unknown',
  pairingStatus: 'Not paired',
  pendingUploads: 0,
  checks: [],
}

export default function SettingsScreen() {
  const [state, setState] = useState<AdminState>(EMPTY_STATE)
  const [language, setLanguage] = useState(DEFAULT_LANGUAGE)
  const [loading, setLoading] = useState(true)

  const rows = useMemo(() => {
    const check = (key: string) => state.checks.find((item) => item.key === key)
    return [
      ['Device ID', compactId(state.deviceId), state.deviceId !== 'Unknown'],
      ['Pairing', state.pairingStatus, state.pairingStatus === 'Paired'],
      ['Patient', state.currentUser, state.currentUser !== 'Not paired'],
      ['Backend', state.backendOnline ? 'Online' : 'Offline', state.backendOnline],
      ['Network', check('network')?.detail || 'Unknown', check('network')?.status === 'ok'],
      ['Microphone', check('mic')?.detail || 'Unknown', check('mic')?.status === 'ok'],
      ['Speaker path', check('speaker')?.detail || 'Unknown', check('speaker')?.status === 'ok'],
      ['Camera', check('camera')?.detail || 'Unknown', check('camera')?.status === 'ok'],
      ['Realtime PCM', check('rtaudio')?.detail || 'Unknown', check('rtaudio')?.status === 'ok'],
      ['Pending uploads', String(state.pendingUploads), state.pendingUploads === 0],
      ['App version', Constants.expoConfig?.version || 'Unknown', true],
      ['Android version', Platform.OS === 'android' ? `API ${String(Platform.Version)}` : `Development: ${Platform.OS}`, true],
      ['Volume', 'Managed by Android', true],
      ['Brightness', 'Managed by Android', true],
    ] as Array<[string, string, boolean]>
  }, [state])

  useEffect(() => { void loadAdminState() }, [])

  async function loadAdminState() {
    setLoading(true)
    try {
      const [deviceId, activeMirrorId, credential, config, pending, storedLanguage, hardware] = await Promise.all([
        AsyncStorage.getItem(DEVICE_ID_STORAGE_KEY),
        AsyncStorage.getItem(ACTIVE_MIRROR_ID_STORAGE_KEY),
        getDeviceCredential(),
        loadJson<unknown>(NURSE_PATIENT_CONFIG_STORAGE_KEY),
        loadPendingConversations(),
        AsyncStorage.getItem(MIRROR_LANGUAGE_STORAGE_KEY),
        runHardwareChecks(),
      ])
      if (storedLanguage?.trim()) setLanguage(storedLanguage.trim())
      const online = typeof navigator === 'undefined' ? true : navigator.onLine !== false
      const backendOnline = online ? await checkBackend() : false
      const currentUser = getCurrentUser(config, activeMirrorId || deviceId || '')
      setState({
        backendOnline,
        currentUser: currentUser || 'Not paired',
        deviceId: deviceId || 'Unknown',
        pairingStatus: credential && activeMirrorId ? 'Paired' : 'Not paired',
        pendingUploads: pending.length,
        checks: hardware.checks,
      })
    } finally {
      setLoading(false)
    }
  }

  async function resetPairing() {
    await clearDeviceCredential({ preserveBootstrap: true })
    await AsyncStorage.multiRemove([
      ACTIVE_MIRROR_ID_STORAGE_KEY,
      ACTIVE_NURSE_ID_STORAGE_KEY,
      ACTIVE_PATIENT_ID_STORAGE_KEY,
      MIRROR_LANGUAGE_STORAGE_KEY,
      MIRROR_TIMEZONE_STORAGE_KEY,
      NURSE_PATIENT_CONFIG_STORAGE_KEY,
    ])
    await clearPendingConversations()
    router.replace('/')
  }

  function confirmResetPairing() {
    Alert.alert('Reset pairing?', 'This removes the local patient link and any pending local uploads.', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Reset', style: 'destructive', onPress: () => void resetPairing() },
    ])
  }

  async function uploadLogs() {
    const result = await flushPendingConversations()
    setState((current) => ({ ...current, pendingUploads: result.remaining }))
    Alert.alert('Upload complete', `Synced ${result.synced}. Remaining ${result.remaining}.`)
  }

  async function changeLanguage(value: string) {
    setLanguage(value)
    await AsyncStorage.setItem(MIRROR_LANGUAGE_STORAGE_KEY, value)
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <View>
            <Text style={styles.eyebrow}>RESTRICTED DEVICE AREA</Text>
            <Text style={styles.title}>Admin diagnostics</Text>
          </View>
          <Pressable accessibilityLabel="Close admin diagnostics" onPress={() => router.replace('/conversation')} style={styles.closeButton}>
            <Ionicons name="close" size={24} color={c.linen} />
          </Pressable>
        </View>

        <View style={styles.statusCard}>
          {rows.map(([label, value, healthy]) => (
            <View key={label} style={styles.row}>
              <View style={[styles.statusDot, healthy ? styles.statusGood : styles.statusIssue]} />
              <Text style={styles.label}>{label}</Text>
              <Text numberOfLines={2} selectable style={styles.value}>{value}</Text>
            </View>
          ))}
          {loading ? <View style={styles.loading}><ActivityIndicator color={c.bronze} /><Text style={styles.loadingText}>Running live checks…</Text></View> : null}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Conversation language</Text>
          <View style={styles.languageGrid}>
            {LANGUAGE_OPTIONS.map((option) => (
              <Pressable
                key={option.value}
                onPress={() => void changeLanguage(option.value)}
                style={[styles.languageChip, language === option.value && styles.languageChipActive]}
              >
                <Text style={[styles.languageText, language === option.value && styles.languageTextActive]}>{option.label}</Text>
              </Pressable>
            ))}
          </View>
          <Text style={styles.note}>Applied to the next conversation.</Text>
        </View>

        <View style={styles.actions}>
          <AdminAction icon="chatbubble-ellipses-outline" label="Start today’s check-in" onPress={() => router.replace({ pathname: '/conversation', params: { start: 'screening' } })} />
          <AdminAction icon="refresh-outline" label="Run checks again" onPress={() => void loadAdminState()} />
          <AdminAction icon="cloud-upload-outline" label="Sync pending uploads" onPress={() => void uploadLogs()} />
          <AdminAction icon="pulse-outline" label="Open detailed hardware report" onPress={() => router.push('/hardware-check')} />
          <AdminAction icon="reload-outline" label="Restart mirror app" onPress={() => router.replace('/')} />
          <AdminAction danger icon="unlink-outline" label="Reset pairing" onPress={confirmResetPairing} />
        </View>
      </ScrollView>
    </SafeAreaView>
  )
}

function AdminAction({ danger, icon, label, onPress }: { danger?: boolean; icon: keyof typeof Ionicons.glyphMap; label: string; onPress: () => void }) {
  return (
    <Pressable onPress={onPress} style={styles.actionRow}>
      <View style={styles.actionLabel}>
        <Ionicons name={icon} size={20} color={danger ? c.error : c.bronze} />
        <Text style={[styles.actionText, danger && styles.dangerText]}>{label}</Text>
      </View>
      <Ionicons name="chevron-forward" size={18} color={c.linenSoft} />
    </Pressable>
  )
}

async function checkBackend() {
  try {
    const response = await fetch(getApiUrl('/health'))
    return response.ok
  } catch {
    return false
  }
}

function getCurrentUser(config: unknown, mirrorId: string) {
  if (!config || typeof config !== 'object') return ''
  const name = (config as { patient?: { displayName?: unknown } }).patient?.displayName
  if (typeof name === 'string') return name
  const patients = (config as { patients?: Array<{ mirrorId?: unknown; name?: unknown }> }).patients
  const patient = patients?.find((candidate) => mirrorIdToString(candidate.mirrorId) === mirrorId)
  return typeof patient?.name === 'string' ? patient.name : ''
}

function mirrorIdToString(value: unknown) {
  if (typeof value === 'string') return value
  if (value && typeof value === 'object' && '$oid' in value) return (value as { $oid?: string }).$oid || ''
  return ''
}

function compactId(value: string) {
  if (!value || value === 'Unknown') return value
  return value.length > 18 ? `${value.slice(0, 9)}…${value.slice(-5)}` : value
}

const styles = StyleSheet.create({
  safeArea: { backgroundColor: c.cream, flex: 1 },
  content: { gap: 22, paddingBottom: 48, paddingHorizontal: 24, paddingTop: 28 },
  header: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between' },
  eyebrow: { color: c.goldDeep, fontFamily: f.bodyMedium, fontSize: 10, letterSpacing: 1.8 },
  title: { color: c.text, fontFamily: f.display, fontSize: 32, marginTop: 6 },
  closeButton: { alignItems: 'center', backgroundColor: c.white, borderColor: c.lineWarm, borderRadius: 23, borderWidth: 1, height: 46, justifyContent: 'center', width: 46 },
  statusCard: { backgroundColor: 'rgba(255,255,255,0.72)', borderColor: c.lineWarm, borderRadius: 20, borderWidth: 1, overflow: 'hidden' },
  row: { alignItems: 'center', borderBottomColor: c.lineWarm, borderBottomWidth: StyleSheet.hairlineWidth, flexDirection: 'row', gap: 10, minHeight: 54, paddingHorizontal: 15, paddingVertical: 10 },
  statusDot: { borderRadius: 4, height: 7, width: 7 },
  statusGood: { backgroundColor: c.sageDeep },
  statusIssue: { backgroundColor: c.coral },
  label: { color: c.textSecondary, flex: 0.8, fontFamily: f.body, fontSize: 13 },
  value: { color: c.text, flex: 1.2, fontFamily: f.bodyMedium, fontSize: 13, textAlign: 'right' },
  loading: { alignItems: 'center', flexDirection: 'row', gap: 10, padding: 15 },
  loadingText: { color: c.textSecondary, fontFamily: f.body, fontSize: 13 },
  section: { gap: 12 },
  sectionTitle: { color: c.text, fontFamily: f.bodyMedium, fontSize: 17 },
  languageGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  languageChip: { backgroundColor: 'rgba(255,255,255,0.5)', borderColor: c.lineWarm, borderRadius: 18, borderWidth: 1, paddingHorizontal: 13, paddingVertical: 9 },
  languageChipActive: { backgroundColor: c.beige, borderColor: c.goldDeep },
  languageText: { color: c.textSecondary, fontFamily: f.body, fontSize: 13 },
  languageTextActive: { color: c.text, fontFamily: f.bodyMedium },
  note: { color: c.textSecondary, fontFamily: f.body, fontSize: 12, opacity: 0.8 },
  actions: { backgroundColor: 'rgba(255,255,255,0.58)', borderColor: c.lineWarm, borderRadius: 20, borderWidth: 1, overflow: 'hidden' },
  actionRow: { alignItems: 'center', borderBottomColor: c.lineWarm, borderBottomWidth: StyleSheet.hairlineWidth, flexDirection: 'row', justifyContent: 'space-between', minHeight: 58, paddingHorizontal: 16 },
  actionLabel: { alignItems: 'center', flexDirection: 'row', gap: 11 },
  actionText: { color: c.text, fontFamily: f.bodyMedium, fontSize: 15 },
  dangerText: { color: c.coral },
})
