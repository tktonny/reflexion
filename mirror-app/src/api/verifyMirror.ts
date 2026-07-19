import { apiPost } from './client'

export type VerifyMirrorResponse =
  | { success: true; nursePatientConfig: unknown }
  | { success: false; reason: string }

export function verifyMirror(mirrorId: string) {
  return apiPost<VerifyMirrorResponse>('/api/verify-mirror', { mirrorId })
}
