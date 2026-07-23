import { apiGet } from './apiClient';
import type { TrendDay } from '../data/mockData';

type TrendRange = 7 | 30;

export async function fetchPatientTrend(
  patientId: string,
  days: TrendRange,
) {
  const body = await apiGet<{ trend?: TrendDay[] }>(`/api/patient-trend?id=${encodeURIComponent(patientId)}&days=${days}`);

  const trend = Array.isArray(body?.trend) ? body.trend : [];
  return trend;
}
