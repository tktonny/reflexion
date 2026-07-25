import { apiGet } from './apiClient';

/**
 * One day of the patient-trend read model (`GET /api/patient-trend`).
 *
 * Defined here, next to the request that produces it, rather than in the old mockData module — where the
 * app's fixtures and its shared domain types lived together and production screens ended up compiling
 * against a three-state legacy `status` vocabulary that no longer matches the authoritative four-state
 * V1Status (see src/lib/v1Status.ts).
 */
export type TrendDay = {
  date: string;
  duration: number;
  /** Per-day colour as computed by the server. Not the caregiver-facing status — see v1Status.ts. */
  status: 'green' | 'yellow' | 'red';
  missed: boolean;
};

type TrendRange = 7 | 30;

export async function fetchPatientTrend(
  patientId: string,
  days: TrendRange,
) {
  const body = await apiGet<{ trend?: TrendDay[] }>(`/api/patient-trend?id=${encodeURIComponent(patientId)}&days=${days}`);

  const trend = Array.isArray(body?.trend) ? body.trend : [];
  return trend;
}
