import AsyncStorage from '@react-native-async-storage/async-storage'

import {
  ACTIVE_MIRROR_ID_STORAGE_KEY,
  ACTIVE_NURSE_ID_STORAGE_KEY,
  ACTIVE_PATIENT_ID_STORAGE_KEY,
  DEVICE_ID_STORAGE_KEY,
  NURSE_PATIENT_CONFIG_STORAGE_KEY,
  extractNursePatientIds,
} from '../constants/nursePatientConfig'

export async function loadJson<T>(key: string): Promise<T | null> {
  const raw = await AsyncStorage.getItem(key)
  if (!raw) return null

  try {
    return JSON.parse(raw) as T
  } catch {
    await AsyncStorage.removeItem(key)
    return null
  }
}

export async function persistNursePatientIds(config: unknown, mirrorId: string) {
  const { nurseId, patientId } = extractNursePatientIds(config, mirrorId)

  if (nurseId) {
    await AsyncStorage.setItem(ACTIVE_NURSE_ID_STORAGE_KEY, nurseId)
  } else {
    await AsyncStorage.removeItem(ACTIVE_NURSE_ID_STORAGE_KEY)
  }

  if (patientId) {
    await AsyncStorage.setItem(ACTIVE_PATIENT_ID_STORAGE_KEY, patientId)
  } else {
    await AsyncStorage.removeItem(ACTIVE_PATIENT_ID_STORAGE_KEY)
  }

  return { nurseId, patientId }
}

export async function getStoredConversationOwnerIds() {
  const [nurseId, patientId] = await Promise.all([
    AsyncStorage.getItem(ACTIVE_NURSE_ID_STORAGE_KEY),
    AsyncStorage.getItem(ACTIVE_PATIENT_ID_STORAGE_KEY),
  ])

  if (!nurseId || !patientId) return null
  return { nurseId, patientId }
}

export async function getStoredPatientName() {
  const [mirrorId, patientId, config] = await Promise.all([
    AsyncStorage.getItem(ACTIVE_MIRROR_ID_STORAGE_KEY),
    AsyncStorage.getItem(ACTIVE_PATIENT_ID_STORAGE_KEY),
    loadJson<unknown>(NURSE_PATIENT_CONFIG_STORAGE_KEY),
  ])

  if (!mirrorId || !config || typeof config !== 'object') return null
  const patients = (config as { patients?: Array<{ mirrorId?: unknown; name?: unknown }> })
    .patients
  const patient = findStoredPatient(patients, mirrorId, patientId)

  return typeof patient?.name === 'string' && patient.name.trim()
    ? patient.name.trim()
    : null
}

export async function getStoredMirrorProfile() {
  const [mirrorId, patientId, config] = await Promise.all([
    AsyncStorage.getItem(ACTIVE_MIRROR_ID_STORAGE_KEY),
    AsyncStorage.getItem(ACTIVE_PATIENT_ID_STORAGE_KEY),
    loadJson<unknown>(NURSE_PATIENT_CONFIG_STORAGE_KEY),
  ])

  if (!mirrorId || !config || typeof config !== 'object') {
    return { patientName: null, nurseName: null }
  }

  const root = config as {
    nurseName?: unknown
    caregiverName?: unknown
    name?: unknown
    nurse?: { name?: unknown }
    caregiver?: { name?: unknown }
    patients?: Array<{ _id?: unknown; mirrorId?: unknown; name?: unknown }>
  }
  const patient = findStoredPatient(root.patients, mirrorId, patientId)

  const nurseName =
    root.nurseName ?? root.caregiverName ?? root.nurse?.name ?? root.caregiver?.name ?? root.name

  return {
    patientName:
      typeof patient?.name === 'string' && patient.name.trim()
        ? patient.name.trim()
        : null,
    nurseName:
      typeof nurseName === 'string' && nurseName.trim() ? nurseName.trim() : null,
  }
}

export async function getStoredMirrorSessionMetadata() {
  const [deviceId, mirrorId, patientId, config] = await Promise.all([
    AsyncStorage.getItem(DEVICE_ID_STORAGE_KEY),
    AsyncStorage.getItem(ACTIVE_MIRROR_ID_STORAGE_KEY),
    AsyncStorage.getItem(ACTIVE_PATIENT_ID_STORAGE_KEY),
    loadJson<unknown>(NURSE_PATIENT_CONFIG_STORAGE_KEY),
  ])
  const activeMirrorId = mirrorId || deviceId || ''

  if (!activeMirrorId || !config || typeof config !== 'object') {
    return { deviceId: activeMirrorId || null, language: null }
  }

  const patients = (config as { patients?: Array<{ _id?: unknown; mirrorId?: unknown; preferredLanguage?: unknown }> })
    .patients
  const patient = findStoredPatient(patients, activeMirrorId, patientId)

  return {
    deviceId: activeMirrorId,
    language:
      typeof patient?.preferredLanguage === 'string' && patient.preferredLanguage.trim()
        ? patient.preferredLanguage.trim()
        : null,
  }
}

function objectIdLikeToString(value: unknown) {
  if (typeof value === 'string') return value
  if (value && typeof value === 'object' && '$oid' in value) {
    return (value as { $oid?: string }).$oid || ''
  }
  return ''
}

function findStoredPatient<T extends { _id?: unknown; mirrorId?: unknown }>(
  patients: T[] | undefined,
  mirrorId: string,
  patientId: string | null,
) {
  if (!patients) return null
  if (patientId) {
    const patient = patients.find((candidate) => objectIdLikeToString(candidate._id) === patientId)
    if (patient) return patient
  }

  return (
    patients.find((candidate) => objectIdLikeToString(candidate.mirrorId) === mirrorId) ?? null
  )
}
