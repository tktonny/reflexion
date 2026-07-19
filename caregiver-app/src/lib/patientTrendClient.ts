import { getApiUrl } from './apiUrl';
import type { TrendDay } from '../data/mockData';

type TrendRange = 7 | 30;
type TrendCache = Record<string, TrendDay[]>;

const trendCache: TrendCache = {};

export async function fetchPatientTrend(
  patientId: string,
  days: TrendRange,
  options: { forceRefresh?: boolean } = {},
) {
  const cacheKey = getTrendCacheKey(patientId, days);
  const cachedTrend = trendCache[cacheKey];
  if (cachedTrend && !options.forceRefresh) {
    return cachedTrend;
  }

  const response = await fetch(getApiUrl(`/api/patient-trend?id=${encodeURIComponent(patientId)}&days=${days}`));
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body?.error || 'Unable to load patient trend.');
  }

  const trend = Array.isArray(body?.trend) ? body.trend : [];
  trendCache[cacheKey] = trend;
  return trend;
}

function getTrendCacheKey(id: string, range: TrendRange) {
  return `${id}:${range}:${getSingaporeDateKey(new Date())}`;
}

function getSingaporeDateKey(date: Date) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    day: '2-digit',
    month: '2-digit',
    timeZone: 'Asia/Singapore',
    year: 'numeric',
  }).formatToParts(date);

  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}`;
}
