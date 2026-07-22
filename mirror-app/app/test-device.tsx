import { Ionicons } from '@expo/vector-icons'
import { router } from 'expo-router'
import { useEffect, useState } from 'react'
import { Alert, Platform, Pressable, StyleSheet, Text, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'

import {
  clearDeviceCredential,
  getBootstrapCredential,
  getDeviceCredential,
} from '../src/storage/deviceCredentials'

type DeviceStatus = {
  deviceId: string
  provisioned: boolean
  paired: boolean
  patientId: string
  accessExpiresAt: string
}

const EMPTY_STATUS: DeviceStatus = {
  deviceId: 'None',
  provisioned: false,
  paired: false,
  patientId: 'None',
  accessExpiresAt: 'None',
}

/** Development diagnostics only. Device identity is server-provisioned and cannot be typed here. */
export default function TestDeviceScreen() {
  const [status, setStatus] = useState<DeviceStatus>(EMPTY_STATUS)
  const [clearing, setClearing] = useState(false)

  useEffect(() => { void loadStatus() }, [])

  async function loadStatus() {
    const [bootstrap, credential] = await Promise.all([
      getBootstrapCredential(),
      getDeviceCredential(),
    ])
    setStatus({
      deviceId: credential?.deviceId || bootstrap?.deviceId || 'None',
      provisioned: Boolean(bootstrap),
      paired: Boolean(credential),
      patientId: credential?.patientId || 'None',
      accessExpiresAt: credential?.accessTokenExpiresAt || 'None',
    })
  }

  async function restartPairing() {
    setClearing(true)
    try {
      await clearDeviceCredential({ preserveBootstrap: true })
      router.replace('/')
    } finally {
      setClearing(false)
    }
  }

  function confirmRestart() {
    Alert.alert(
      'Restart pairing?',
      'This clears this device’s local access and refresh credentials. The server provisioning identity is preserved.',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Restart', style: 'destructive', onPress: () => void restartPairing() },
      ],
    )
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.stage}>
        <View style={styles.card}>
          <View style={styles.iconWrap}>
            <Ionicons name="hardware-chip-outline" size={38} color={colors.goldDark} />
          </View>
          <Text style={styles.title}>Device identity</Text>
          <Text style={styles.note}>
            Identity is issued by the backend during provisioning. This screen can inspect it, but cannot create or replace it.
          </Text>

          <StatusRow label="Device ID" value={status.deviceId} />
          <StatusRow label="Provisioned" value={status.provisioned ? 'Yes' : 'No'} />
          <StatusRow label="Paired" value={status.paired ? 'Yes' : 'No'} />
          <StatusRow label="Patient ID" value={status.patientId} />
          <StatusRow label="Access expires" value={status.accessExpiresAt} />

          <Pressable
            disabled={clearing || !status.provisioned}
            onPress={confirmRestart}
            style={[styles.primaryButton, (clearing || !status.provisioned) && styles.disabledButton]}
          >
            <Text style={styles.primaryButtonText}>{clearing ? 'Clearing…' : 'Clear pairing and restart'}</Text>
          </Pressable>

          <Pressable onPress={() => router.replace('/')} style={styles.backButton}>
            <Ionicons name="arrow-back" size={18} color={colors.secondary} />
            <Text style={styles.backButtonText}>Back</Text>
          </Pressable>
        </View>
      </View>
    </SafeAreaView>
  )
}

function StatusRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.currentBlock}>
      <Text style={styles.label}>{label}</Text>
      <Text selectable style={styles.currentValue}>{value}</Text>
    </View>
  )
}

const colors = {
  background: '#FFF9F1',
  card: '#FFFBF4',
  line: '#F1E5D2',
  goldDark: '#C89755',
  taupe: '#BBAFA0',
  text: '#282828',
  secondary: '#686868',
}

const styles = StyleSheet.create({
  safeArea: { backgroundColor: colors.background, flex: 1 },
  stage: { alignItems: 'center', backgroundColor: colors.background, flex: 1, justifyContent: 'center', padding: 18 },
  card: {
    alignItems: 'stretch', backgroundColor: colors.card, borderColor: colors.line, borderRadius: 8,
    borderWidth: 1, gap: 14, maxWidth: 430, padding: 28, shadowColor: '#D8C6A8',
    shadowOffset: { height: 10, width: 0 }, shadowOpacity: 0.22, shadowRadius: 22, width: '100%',
  },
  iconWrap: {
    alignItems: 'center', alignSelf: 'center', backgroundColor: '#FFF2DF', borderRadius: 36,
    height: 72, justifyContent: 'center', width: 72,
  },
  title: { color: colors.text, fontSize: 30, fontWeight: '900', textAlign: 'center' },
  note: { color: colors.secondary, fontSize: 16, fontWeight: '700', lineHeight: 23, textAlign: 'center' },
  currentBlock: {
    backgroundColor: '#FFF6EA', borderColor: colors.line, borderRadius: 8, borderWidth: 1, gap: 6, padding: 12,
  },
  label: { color: colors.secondary, fontSize: 12, fontWeight: '900', textTransform: 'uppercase' },
  currentValue: {
    color: colors.text, fontFamily: Platform.select({ web: 'monospace', default: undefined }),
    fontSize: 14, fontWeight: '800',
  },
  primaryButton: {
    alignItems: 'center', backgroundColor: colors.goldDark, borderRadius: 8, justifyContent: 'center', minHeight: 48,
  },
  primaryButtonText: { color: '#FFFFFF', fontSize: 15, fontWeight: '900' },
  disabledButton: { opacity: 0.48 },
  backButton: {
    alignItems: 'center', alignSelf: 'center', flexDirection: 'row', gap: 8, justifyContent: 'center', paddingVertical: 6,
  },
  backButtonText: { color: colors.secondary, fontSize: 15, fontWeight: '800' },
})
