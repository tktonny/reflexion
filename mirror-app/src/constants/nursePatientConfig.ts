export const ACTIVE_MIRROR_ID_STORAGE_KEY = 'reflexion:activeMirrorId'
export const NURSE_PATIENT_CONFIG_STORAGE_KEY = 'reflexion:nursePatientConfig'
export const ACTIVE_NURSE_ID_STORAGE_KEY = 'reflexion:activeNurseId'
export const ACTIVE_PATIENT_ID_STORAGE_KEY = 'reflexion:activePatientId'
export const DEVICE_ID_STORAGE_KEY = 'reflexion:deviceId'
export const DEVICE_AUTH_TOKEN_STORAGE_KEY = 'reflexion:deviceAuthToken'
export const DEVICE_CREDENTIAL_ID_STORAGE_KEY = 'reflexion:deviceCredentialId'
export const DEVICE_ACCESS_EXPIRES_AT_STORAGE_KEY = 'reflexion:deviceAccessExpiresAt'
export const DEVICE_REFRESH_EXPIRES_AT_STORAGE_KEY = 'reflexion:deviceRefreshExpiresAt'
export const DEVICE_BOOTSTRAP_TOKEN_STORAGE_KEY = 'reflexion:deviceBootstrapToken'
export const ACTIVE_SESSION_STORAGE_KEY = 'reflexion:activeV1Session'
export const MIRROR_LANGUAGE_STORAGE_KEY = 'reflexion:mirrorLanguage'
export const MIRROR_TIMEZONE_STORAGE_KEY = 'reflexion:mirrorTimezone'

type ObjectIdLike = string | { $oid?: string } | null | undefined

export type StoredPatient = {
  _id?: ObjectIdLike
  mirrorId?: ObjectIdLike
  name?: unknown
  [key: string]: unknown
}

type StoredNursePatientConfig = {
  _id?: ObjectIdLike
  patients?: StoredPatient[]
}

export function objectIdToString(value: ObjectIdLike): string | null {
  if (typeof value === 'string') return value
  if (value && typeof value.$oid === 'string') return value.$oid
  return null
}

export function extractNursePatientIds(
  config: unknown,
  mirrorId: string,
): { nurseId: string | null; patientId: string | null } {
  if (!config || typeof config !== 'object') {
    return { nurseId: null, patientId: null }
  }

  const nursePatientConfig = config as StoredNursePatientConfig
  const patient = nursePatientConfig.patients?.find(
    (candidate) => objectIdToString(candidate.mirrorId) === mirrorId,
  )

  return {
    nurseId: objectIdToString(nursePatientConfig._id),
    patientId: objectIdToString(patient?._id),
  }
}

export function extractPatientForMirror(
  config: unknown,
  mirrorId: string,
): { patient: StoredPatient | null; patientId: string | null } {
  if (!config || typeof config !== 'object') {
    return { patient: null, patientId: null }
  }

  const nursePatientConfig = config as StoredNursePatientConfig
  const patient =
    nursePatientConfig.patients?.find(
      (candidate) => objectIdToString(candidate.mirrorId) === mirrorId,
    ) ?? null

  return {
    patient,
    patientId: objectIdToString(patient?._id),
  }
}
