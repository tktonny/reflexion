import AsyncStorage from '@react-native-async-storage/async-storage'
import { randomUUID } from 'expo-crypto'
import * as SecureStore from 'expo-secure-store'
import { Platform } from 'react-native'

import { getApiUrl } from '../config/apiUrl'
import { recordPairingDiagnostic } from '../lib/pairingDiagnostics'
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
const SECURE_LOCAL_DEVICE_ID = 'reflexion_local_device_id'
const SECURE_LOCAL_DEVICE_SECRET = 'reflexion_local_device_secret'
const SECURE_PENDING_PAIRING = 'reflexion_pending_pairing'
const SECURE_PAIRING_IDEMPOTENCY_KEY = 'reflexion_pairing_idempotency_key'
const SECURE_PENDING_EXCHANGE = 'reflexion_pending_exchange'
const LOCAL_DEVICE_ID_PATTERN = /^dev_[A-Za-z0-9_-]{16,100}$/

let localIdentityPromise: Promise<LocalDeviceIdentity> | null = null

export type StoredDeviceCredential = {
  deviceId: string
  credentialId: string
  patientId: string
  accessToken: string
  accessTokenExpiresAt: string
  refreshCredential: string
  refreshCredentialExpiresAt: string
}

export type LocalDeviceIdentity = {
  deviceId: string
  installSecret: string
}

export type PendingDevicePairing = {
  deviceId: string
  pairingId: string
  pairingCode: string
  pairingToken: string
  expiresAt: string
  idempotencyKey: string
}

export type PendingCredentialExchange = {
  pairingId: string
  exchangeTicket: string
  idempotencyKey: string
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

/**
 * Fresh mirrors do not need a factory-issued secret before they can show a pairing code. The local
 * identity is generated once and kept in SecureStore; the server binds it to an unclaimed device record
 * the first time pairing starts. The install secret is never shown, placed in a QR code, or logged.
 */
export async function getOrCreateLocalDeviceIdentity(): Promise<LocalDeviceIdentity> {
  if (!localIdentityPromise) localIdentityPromise = loadOrCreateLocalDeviceIdentity()
  return localIdentityPromise
}

export async function getDevicePairingIdentity(): Promise<{
  deviceId: string
  headers: Record<string, string>
  source: 'bootstrap' | 'local'
}> {
  const bootstrap = await getBootstrapCredential()
  if (bootstrap) {
    return {
      deviceId: bootstrap.deviceId,
      headers: { 'X-Device-Bootstrap': bootstrap.token },
      source: 'bootstrap' as const,
    }
  }
  const local = await getOrCreateLocalDeviceIdentity()
  return {
    deviceId: local.deviceId,
    headers: { 'X-Device-Id': local.deviceId, 'X-Device-Install-Secret': local.installSecret },
    source: 'local' as const,
  }
}

export async function getPairingIdempotencyKey() {
  const stored = await secureGet(SECURE_PAIRING_IDEMPOTENCY_KEY)
  if (stored) return stored
  const created = randomIdempotencyKey()
  await secureSet(SECURE_PAIRING_IDEMPOTENCY_KEY, created)
  return created
}

export async function savePendingDevicePairing(value: PendingDevicePairing) {
  await secureSet(SECURE_PENDING_PAIRING, JSON.stringify(value))
}

export async function getPendingDevicePairing(): Promise<PendingDevicePairing | null> {
  const raw = await secureGet(SECURE_PENDING_PAIRING)
  if (!raw) return null
  try {
    const value = JSON.parse(raw) as Partial<PendingDevicePairing>
    if (!value.deviceId || !value.pairingId || !value.pairingCode || !value.pairingToken || !value.expiresAt || !value.idempotencyKey) return null
    return value as PendingDevicePairing
  } catch {
    await secureDelete(SECURE_PENDING_PAIRING)
    return null
  }
}

export async function clearPendingDevicePairing() {
  await Promise.all([
    secureDelete(SECURE_PENDING_PAIRING),
    secureDelete(SECURE_PAIRING_IDEMPOTENCY_KEY),
    secureDelete(SECURE_PENDING_EXCHANGE),
  ])
}

export async function savePendingCredentialExchange(value: PendingCredentialExchange) {
  await secureSet(SECURE_PENDING_EXCHANGE, JSON.stringify(value))
}

export async function getPendingCredentialExchange(): Promise<PendingCredentialExchange | null> {
  const raw = await secureGet(SECURE_PENDING_EXCHANGE)
  if (!raw) return null
  try {
    const value = JSON.parse(raw) as Partial<PendingCredentialExchange>
    if (!value.pairingId || !value.exchangeTicket || !value.idempotencyKey) return null
    return value as PendingCredentialExchange
  } catch {
    await secureDelete(SECURE_PENDING_EXCHANGE)
    return null
  }
}

export async function clearPendingCredentialExchange() {
  await secureDelete(SECURE_PENDING_EXCHANGE)
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
  try {
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
  } catch {
    await recordPairingDiagnostic('CREDENTIAL_STORAGE_FAILED', false, 'SECURE_STORAGE_FAILURE')
    throw new Error('CREDENTIAL_STORAGE_FAILED')
  }
  await clearPendingDevicePairing()
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

async function loadOrCreateLocalDeviceIdentity(): Promise<LocalDeviceIdentity> {
  const [storedDeviceId, storedSecret, legacyDeviceId] = await Promise.all([
    secureGet(SECURE_LOCAL_DEVICE_ID),
    secureGet(SECURE_LOCAL_DEVICE_SECRET),
    AsyncStorage.getItem(DEVICE_ID_STORAGE_KEY),
  ])
  if (storedDeviceId && storedSecret) {
    await AsyncStorage.setItem(DEVICE_ID_STORAGE_KEY, storedDeviceId)
    return { deviceId: storedDeviceId, installSecret: storedSecret }
  }

  // Keep a previously known local ID when SecureStore was cleared. The newly generated secret will
  // fail closed against the server and surface the controlled recovery state instead of silently
  // registering a second mirror record for the same physical installation.
  const deviceId = legacyDeviceId && LOCAL_DEVICE_ID_PATTERN.test(legacyDeviceId)
    ? legacyDeviceId
    : `dev_${randomUUID().replaceAll('-', '')}`
  const installSecret = `${randomUUID().replaceAll('-', '')}${randomUUID().replaceAll('-', '')}`
  await Promise.all([
    secureSet(SECURE_LOCAL_DEVICE_ID, deviceId),
    secureSet(SECURE_LOCAL_DEVICE_SECRET, installSecret),
    AsyncStorage.setItem(DEVICE_ID_STORAGE_KEY, deviceId),
  ])
  return { deviceId, installSecret }
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
