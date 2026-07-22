import AsyncStorage from '@react-native-async-storage/async-storage'
import * as SecureStore from 'expo-secure-store'
import { Platform } from 'react-native'

import { getApiUrl } from '../config/apiUrl'
import { validateBootstrapCredential } from '../orchestration/deviceBootstrap'
import {
  ACTIVE_MIRROR_ID_STORAGE_KEY,
  ACTIVE_PATIENT_ID_STORAGE_KEY,
  DEVICE_ACCESS_EXPIRES_AT_STORAGE_KEY,
  DEVICE_AUTH_TOKEN_STORAGE_KEY,
  DEVICE_BOOTSTRAP_TOKEN_STORAGE_KEY,
  DEVICE_CREDENTIAL_ID_STORAGE_KEY,
  DEVICE_ID_STORAGE_KEY,
  DEVICE_REFRESH_EXPIRES_AT_STORAGE_KEY,
} from '../constants/nursePatientConfig'

const SECURE_ACCESS = 'reflexion_device_access_token'
const SECURE_REFRESH = 'reflexion_device_refresh_credential'
const SECURE_BOOTSTRAP = 'reflexion_device_bootstrap_token'

export type StoredDeviceCredential = {
  deviceId: string
  credentialId: string
  patientId: string
  accessToken: string
  accessTokenExpiresAt: string
  refreshCredential: string
  refreshCredentialExpiresAt: string
}

export async function getBootstrapCredential() {
  const stored = await secureGet(SECURE_BOOTSTRAP)
  const configured = process.env.EXPO_PUBLIC_DEVICE_BOOTSTRAP_TOKEN?.trim()
  for (const candidate of [stored, configured]) {
    if (!candidate) continue
    try {
      const { deviceId } = validateBootstrapCredential(candidate)
      if (candidate !== stored) await secureSet(SECURE_BOOTSTRAP, candidate)
      await AsyncStorage.multiSet([
        [DEVICE_ID_STORAGE_KEY, deviceId],
        [DEVICE_BOOTSTRAP_TOKEN_STORAGE_KEY, 'secure-store'],
      ])
      return { token: candidate, deviceId }
    } catch {
      if (candidate === stored) await secureDelete(SECURE_BOOTSTRAP)
    }
  }
  await AsyncStorage.multiRemove([DEVICE_BOOTSTRAP_TOKEN_STORAGE_KEY])
  return null
}

export async function persistBootstrapCredential(token: string) {
  const normalized = token.trim()
  const { deviceId } = validateBootstrapCredential(normalized)
  await clearDeviceCredential({ preserveBootstrap: false })
  await Promise.all([
    secureSet(SECURE_BOOTSTRAP, normalized),
    AsyncStorage.multiSet([
      [DEVICE_ID_STORAGE_KEY, deviceId],
      [DEVICE_BOOTSTRAP_TOKEN_STORAGE_KEY, 'secure-store'],
    ]),
  ])
  return { deviceId }
}

export async function persistDeviceCredential(value: StoredDeviceCredential) {
  await Promise.all([
    secureSet(SECURE_ACCESS, value.accessToken),
    secureSet(SECURE_REFRESH, value.refreshCredential),
    AsyncStorage.multiSet([
      [DEVICE_ID_STORAGE_KEY, value.deviceId],
      [ACTIVE_MIRROR_ID_STORAGE_KEY, value.deviceId],
      [ACTIVE_PATIENT_ID_STORAGE_KEY, value.patientId],
      [DEVICE_CREDENTIAL_ID_STORAGE_KEY, value.credentialId],
      [DEVICE_ACCESS_EXPIRES_AT_STORAGE_KEY, value.accessTokenExpiresAt],
      [DEVICE_REFRESH_EXPIRES_AT_STORAGE_KEY, value.refreshCredentialExpiresAt],
      [DEVICE_AUTH_TOKEN_STORAGE_KEY, 'secure-store'],
    ]),
  ])
}

export async function getDeviceCredential() {
  const [deviceId, patientId, credentialId, accessToken, accessTokenExpiresAt, refreshCredential, refreshCredentialExpiresAt] = await Promise.all([
    AsyncStorage.getItem(DEVICE_ID_STORAGE_KEY),
    AsyncStorage.getItem(ACTIVE_PATIENT_ID_STORAGE_KEY),
    AsyncStorage.getItem(DEVICE_CREDENTIAL_ID_STORAGE_KEY),
    secureGet(SECURE_ACCESS),
    AsyncStorage.getItem(DEVICE_ACCESS_EXPIRES_AT_STORAGE_KEY),
    secureGet(SECURE_REFRESH),
    AsyncStorage.getItem(DEVICE_REFRESH_EXPIRES_AT_STORAGE_KEY),
  ])
  if (!deviceId || !patientId || !credentialId || !accessToken || !accessTokenExpiresAt || !refreshCredential || !refreshCredentialExpiresAt) return null
  return { deviceId, patientId, credentialId, accessToken, accessTokenExpiresAt, refreshCredential, refreshCredentialExpiresAt }
}

export async function ensureDeviceAccessToken() {
  const credential = await getDeviceCredential()
  if (!credential) throw new Error('device_not_paired')
  if (Date.parse(credential.refreshCredentialExpiresAt) <= Date.now()) throw new Error('device_refresh_expired')
  if (Date.parse(credential.accessTokenExpiresAt) > Date.now() + 60_000) return credential.accessToken
  return rotateDeviceCredential(credential)
}

export async function deviceFetch(path: string, init: RequestInit = {}) {
  let token = await ensureDeviceAccessToken()
  let response = await fetch(getApiUrl(path), { ...init, headers: { ...headersObject(init.headers), Authorization: `Bearer ${token}` } })
  if (response.status === 401) {
    const credential = await getDeviceCredential()
    if (!credential) return response
    token = await rotateDeviceCredential(credential)
    response = await fetch(getApiUrl(path), { ...init, headers: { ...headersObject(init.headers), Authorization: `Bearer ${token}` } })
  }
  return response
}

export async function clearDeviceCredential({ preserveBootstrap = true } = {}) {
  await Promise.all([
    secureDelete(SECURE_ACCESS),
    secureDelete(SECURE_REFRESH),
    preserveBootstrap ? Promise.resolve() : secureDelete(SECURE_BOOTSTRAP),
    AsyncStorage.multiRemove([
      ACTIVE_MIRROR_ID_STORAGE_KEY, ACTIVE_PATIENT_ID_STORAGE_KEY, DEVICE_CREDENTIAL_ID_STORAGE_KEY,
      DEVICE_ACCESS_EXPIRES_AT_STORAGE_KEY, DEVICE_REFRESH_EXPIRES_AT_STORAGE_KEY, DEVICE_AUTH_TOKEN_STORAGE_KEY,
    ]),
  ])
}

async function rotateDeviceCredential(credential: StoredDeviceCredential) {
  const response = await fetch(getApiUrl(`/api/v1/devices/${encodeURIComponent(credential.deviceId)}/credential-rotations`), {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'Idempotency-Key': randomIdempotencyKey() },
    body: JSON.stringify({ credentialId: credential.credentialId, refreshCredential: credential.refreshCredential }),
  })
  const payload = await response.json().catch(() => null) as { data?: StoredDeviceCredential; error?: { code?: string } } | null
  if (!response.ok || !payload?.data) throw new Error(payload?.error?.code || 'device_credential_rotation_failed')
  await persistDeviceCredential(payload.data)
  return payload.data.accessToken
}

function headersObject(headers?: HeadersInit) {
  if (!headers) return {}
  if (headers instanceof Headers) return Object.fromEntries(headers.entries())
  if (Array.isArray(headers)) return Object.fromEntries(headers)
  return headers
}

export function randomIdempotencyKey() {
  return `mirror_${Date.now()}_${Math.random().toString(36).slice(2)}_${Math.random().toString(36).slice(2)}`
}

async function secureGet(key: string) { return Platform.OS === 'web' ? AsyncStorage.getItem(`secure:${key}`) : SecureStore.getItemAsync(key) }
async function secureSet(key: string, value: string) { return Platform.OS === 'web' ? AsyncStorage.setItem(`secure:${key}`, value) : SecureStore.setItemAsync(key, value) }
async function secureDelete(key: string) { return Platform.OS === 'web' ? AsyncStorage.removeItem(`secure:${key}`) : SecureStore.deleteItemAsync(key) }
