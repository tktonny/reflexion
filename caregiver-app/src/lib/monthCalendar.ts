export type CalendarCell = {
  date: string;
  day: number;
  inMonth: boolean;
};

export const WEEKDAY_LABELS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] as const;

function pad(value: number): string {
  return String(value).padStart(2, '0');
}

function localDateKey(date: Date): string {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function monthParts(month: string): { year: number; monthIndex: number } {
  const match = /^(\d{4})-(\d{2})$/.exec(month);
  if (!match) throw new Error(`Invalid month: ${month}`);
  const year = Number(match[1]);
  const monthNumber = Number(match[2]);
  if (monthNumber < 1 || monthNumber > 12) throw new Error(`Invalid month: ${month}`);
  return { year, monthIndex: monthNumber - 1 };
}

export function monthKey(date = new Date()): string {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}`;
}

/** Shift a YYYY-MM key using local calendar arithmetic, including year boundaries. */
export function shiftMonth(month: string, delta: number): string {
  const { year, monthIndex } = monthParts(month);
  const shifted = new Date(year, monthIndex + delta, 1);
  return monthKey(shifted);
}

export function monthLabel(month: string, locale = 'en-SG'): string {
  const { year, monthIndex } = monthParts(month);
  return new Intl.DateTimeFormat(locale, { month: 'long', year: 'numeric' }).format(new Date(year, monthIndex, 1));
}

/**
 * Build a Monday-first calendar grid. Leading and trailing dates are present but
 * marked out of month so the grid always keeps correct weekday alignment without
 * horizontal scrolling or fixed-width day cards.
 */
export function buildMonthCalendar(month: string): CalendarCell[] {
  const { year, monthIndex } = monthParts(month);
  const firstDay = new Date(year, monthIndex, 1);
  const mondayFirstOffset = (firstDay.getDay() + 6) % 7;
  const daysInMonth = new Date(year, monthIndex + 1, 0).getDate();
  const cellCount = Math.ceil((mondayFirstOffset + daysInMonth) / 7) * 7;

  return Array.from({ length: cellCount }, (_, index) => {
    const dayOffset = index - mondayFirstOffset + 1;
    const date = new Date(year, monthIndex, dayOffset);
    return {
      date: localDateKey(date),
      day: date.getDate(),
      inMonth: date.getMonth() === monthIndex && date.getFullYear() === year,
    };
  });
}
