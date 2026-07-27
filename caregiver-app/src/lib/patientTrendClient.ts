import { getSessionTrendV1 } from './v1Caregiver';

/**
 * One day of the trend read model (`GET /api/v1/patients/:id/session-trend`).
 *
 * `status` is the colour the finaliser wrote to daily_statuses, passed through untouched. The app never
 * derives it: colouring a bar from a duration would be the client deciding a day looked bad, which is the
 * one thing the status contract forbids. Null means no finalised status for that day yet — today before the
 * evening finalise, or a loved one still establishing a baseline — and the chart uses its neutral fill.
 */
export type TrendDay = {
  date: string;
  duration: number;
  sessionCount: number;
  status: 'green' | 'amber' | 'red' | null;
  missed: boolean;
};

type TrendRange = 7 | 30;

export function fetchPatientTrend(patientId: string, days: TrendRange): Promise<TrendDay[]> {
  return getSessionTrendV1(patientId, days) as Promise<TrendDay[]>;
}
