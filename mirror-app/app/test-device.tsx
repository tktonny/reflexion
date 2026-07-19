import AsyncStorage from '@react-native-async-storage/async-storage'
import { Ionicons } from '@expo/vector-icons'
import { router } from 'expo-router'
import { useEffect, useState } from 'react'
import { Alert, Platform, Pressable, StyleSheet, Text, TextInput, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'

import {
  ACTIVE_MIRROR_ID_STORAGE_KEY,
  ACTIVE_NURSE_ID_STORAGE_KEY,
  ACTIVE_PATIENT_ID_STORAGE_KEY,
  DEVICE_AUTH_TOKEN_STORAGE_KEY,
  DEVICE_ID_STORAGE_KEY,
  MIRROR_LANGUAGE_STORAGE_KEY,
  NURSE_PATIENT_CONFIG_STORAGE_KEY,
} from '../src/constants/nursePatientConfig'
import { generateMirrorId } from '../src/utils/id'

export default function TestDeviceScreen() {
  const [deviceId, setDeviceId] = useState('')
  const [currentDeviceId, setCurrentDeviceId] = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    void loadDeviceId()
  }, [])

  async function loadDeviceId() {
    const storedDeviceId = await AsyncStorage.getItem(DEVICE_ID_STORAGE_KEY)
    setCurrentDeviceId(storedDeviceId || 'None')
    setDeviceId(storedDeviceId || '')
  }

  function generateDeviceId() {
    setDeviceId(generateMirrorId())
  }

  async function applyDeviceId() {
    const normalizedDeviceId = deviceId.trim().toLowerCase()
    if (!isValidMongoObjectId(normalizedDeviceId)) {
      Alert.alert('Invalid device ID', 'Use a valid 24 character MongoDB ObjectId.')
      return
    }

    setSaving(true)
    await AsyncStorage.multiRemove([
      ACTIVE_MIRROR_ID_STORAGE_KEY,
      ACTIVE_NURSE_ID_STORAGE_KEY,
      ACTIVE_PATIENT_ID_STORAGE_KEY,
      DEVICE_AUTH_TOKEN_STORAGE_KEY,
      MIRROR_LANGUAGE_STORAGE_KEY,
      NURSE_PATIENT_CONFIG_STORAGE_KEY,
    ])
    await AsyncStorage.setItem(DEVICE_ID_STORAGE_KEY, normalizedDeviceId)
    setSaving(false)
    router.replace('/')
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.stage}>
        <View style={styles.card}>
          <View style={styles.iconWrap}>
            <Ionicons name="hardware-chip-outline" size={38} color={colors.goldDark} />
          </View>
          <Text style={styles.title}>Testing device ID</Text>
          <Text style={styles.note}>
            Temporary web feature for simulating multiple mirror devices. This clears local pairing state, then the normal
            pairing screen will request a code for the selected device.
          </Text>

          <View style={styles.currentBlock}>
            <Text style={styles.label}>Current device ID</Text>
            <Text style={styles.currentValue}>{currentDeviceId}</Text>
          </View>

          <View style={styles.formBlock}>
            <Text style={styles.label}>Set device ID</Text>
            <TextInput
              autoCapitalize="none"
              autoCorrect={false}
              maxLength={24}
              onChangeText={(value) => setDeviceId(value.replace(/[^0-9a-fA-F]/g, '').slice(0, 24))}
              placeholder="24 character MongoDB ObjectId"
              placeholderTextColor={colors.taupe}
              style={styles.input}
              value={deviceId}
            />
          </View>

          <View style={styles.actions}>
            <Pressable disabled={saving} onPress={generateDeviceId} style={[styles.secondaryButton, saving && styles.disabledButton]}>
              <Text style={styles.secondaryButtonText}>Generate ID</Text>
            </Pressable>
            <Pressable disabled={saving} onPress={() => void applyDeviceId()} style={[styles.primaryButton, saving && styles.disabledButton]}>
              <Text style={styles.primaryButtonText}>{saving ? 'Saving...' : 'Use ID'}</Text>
            </Pressable>
          </View>

          <Pressable onPress={() => router.replace('/')} style={styles.backButton}>
            <Ionicons name="arrow-back" size={18} color={colors.secondary} />
            <Text style={styles.backButtonText}>Back to pairing</Text>
          </Pressable>
        </View>
      </View>
    </SafeAreaView>
  )
}

function isValidMongoObjectId(value: string) {
  return /^[0-9a-f]{24}$/i.test(value)
}

const colors = {
  background: '#FFF9F1',
  card: '#FFFBF4',
  line: '#F1E5D2',
  gold: '#E7CFA6',
  goldDark: '#C89755',
  taupe: '#BBAFA0',
  text: '#282828',
  secondary: '#686868',
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
    alignItems: 'stretch',
    backgroundColor: colors.card,
    borderColor: colors.line,
    borderRadius: 8,
    borderWidth: 1,
    gap: 20,
    maxWidth: 430,
    padding: 28,
    shadowColor: '#D8C6A8',
    shadowOffset: { height: 10, width: 0 },
    shadowOpacity: 0.22,
    shadowRadius: 22,
    width: '100%',
  },
  iconWrap: {
    alignItems: 'center',
    alignSelf: 'center',
    backgroundColor: '#FFF2DF',
    borderRadius: 36,
    height: 72,
    justifyContent: 'center',
    width: 72,
  },
  title: {
    color: colors.text,
    fontSize: 30,
    fontWeight: '900',
    textAlign: 'center',
  },
  note: {
    color: colors.secondary,
    fontSize: 16,
    fontWeight: '700',
    lineHeight: 23,
    textAlign: 'center',
  },
  currentBlock: {
    backgroundColor: '#FFF6EA',
    borderColor: colors.line,
    borderRadius: 8,
    borderWidth: 1,
    gap: 8,
    padding: 14,
  },
  formBlock: {
    gap: 8,
  },
  label: {
    color: colors.secondary,
    fontSize: 13,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  currentValue: {
    color: colors.text,
    fontFamily: Platform.select({ web: 'monospace', default: undefined }),
    fontSize: 14,
    fontWeight: '800',
  },
  input: {
    backgroundColor: '#FFFFFF',
    borderColor: colors.line,
    borderRadius: 8,
    borderWidth: 1,
    color: colors.text,
    fontFamily: Platform.select({ web: 'monospace', default: undefined }),
    fontSize: 16,
    fontWeight: '800',
    letterSpacing: 0,
    paddingHorizontal: 14,
    paddingVertical: 13,
  },
  actions: {
    flexDirection: 'row',
    gap: 10,
  },
  primaryButton: {
    alignItems: 'center',
    backgroundColor: colors.goldDark,
    borderRadius: 8,
    flex: 1,
    justifyContent: 'center',
    minHeight: 48,
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '900',
  },
  secondaryButton: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderColor: colors.line,
    borderRadius: 8,
    borderWidth: 1,
    flex: 1,
    justifyContent: 'center',
    minHeight: 48,
  },
  secondaryButtonText: {
    color: colors.text,
    fontSize: 15,
    fontWeight: '900',
  },
  disabledButton: {
    opacity: 0.62,
  },
  backButton: {
    alignItems: 'center',
    alignSelf: 'center',
    flexDirection: 'row',
    gap: 8,
    justifyContent: 'center',
    paddingVertical: 6,
  },
  backButtonText: {
    color: colors.secondary,
    fontSize: 15,
    fontWeight: '800',
  },
})
