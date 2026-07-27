import { useQuery } from '@tanstack/react-query';
import React, { useCallback, useMemo, useState } from 'react';
import {
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import { listSessionDaysV1 } from '../../src/lib/v1Caregiver';
import { EmptyState, ErrorState, LoadingState } from '../../src/components/ScreenState';
import {
  MIN_TOUCH_TARGET, cardShadow, colors, fontFamily, fontSize, radius, scaleSize, spacing,
} from '../../src/theme';

type CalendarDay = {
  date: string;
  day: number;
  count: number;
  completedCount?: number;
  hasCompletedSession?: boolean;
};

const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

// The month arrows stay 32pt round to fit the dense header, so the reachable area is grown instead.
const MONTH_BUTTON_HIT_SLOP = { bottom: 8, left: 8, right: 8, top: 8 };

export default function SessionHistoryScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const [month, setMonth] = useState(getSingaporeMonthKey(new Date()));
  // Compared against the server's own date keys, so "future" is decided in the patient's calendar rather
  // than the phone's timezone.
  const todayKey = getSingaporeDateKey(new Date());
  // Any non-empty id is real. This used to test /^[0-9a-f]{24}$/, which was the legacy nurse/patient
  // ObjectId hex — but v1 mints `pat_…` ids for loved ones created since the migration, and CLAUDE.md is
  // explicit that v1 ids are opaque strings. The old guard silently blanked the screen for them.
  const shouldLoadRealSession = Boolean(id);
  const monthQuery = useQuery({
    enabled: shouldLoadRealSession,
    queryKey: ['sessionCounts', id, month],
    queryFn: async () => {
      return listSessionDaysV1(id, month);
    },
  });
  const { refetch: refetchMonth } = monthQuery;
  useFocusEffect(
    useCallback(() => {
      if (shouldLoadRealSession) {
        void refetchMonth();
      }
    }, [refetchMonth, shouldLoadRealSession]),
  );
  const days = monthQuery.data || [];
  const calendarCells = useMemo(() => buildCalendarCells(month, days), [days, month]);
  const totalSessions = days.reduce((sum, day) => sum + day.count, 0);

  // Usable cached data outranks a stale error: this screen refetches on every focus, and in react-query a
  // failed refetch keeps the data it already had. Gating the grid on isError alone wiped a fully populated
  // month the moment the caregiver came back to it on a flaky connection.
  const hasMonthData = days.length > 0;
  const showCalendar = !monthQuery.isLoading && (hasMonthData || !monthQuery.isError);
  const isEmptyMonth = showCalendar && monthQuery.isSuccess && totalSessions === 0;
  const monthSubtitle = monthQuery.isLoading
    ? 'Loading sessions...'
    : monthQuery.isError && !hasMonthData
      ? 'Not loaded yet'
      : `${totalSessions} sessions this month`;

  if (!shouldLoadRealSession) {
    return (
      <SafeAreaView style={styles.safe}>
        <View style={styles.stateWrap}>
          <EmptyState
            icon="calendar"
            title="Bear with us"
            message="This session history is not ready to show yet."
          />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.card}>
          <View style={styles.monthHeader}>
            <TouchableOpacity
              accessibilityLabel={`Previous month, ${formatMonthTitle(addMonths(month, -1))}`}
              accessibilityRole="button"
              hitSlop={MONTH_BUTTON_HIT_SLOP}
              style={styles.monthButton}
              onPress={() => setMonth(addMonths(month, -1))}
            >
              <Feather
                name="chevron-left"
                size={18}
                color={colors.accent}
                accessibilityElementsHidden={true}
                importantForAccessibility="no"
              />
            </TouchableOpacity>
            <View style={styles.monthTitleWrap}>
              <Text maxFontSizeMultiplier={1.4} style={styles.monthTitle}>{formatMonthTitle(month)}</Text>
              {/* Read out on its own after a month change or a failed load, so the caregiver hears which
                  month they landed on without going looking for it. */}
              <Text accessibilityLiveRegion="polite" style={styles.monthSubtitle}>
                {monthSubtitle}
              </Text>
            </View>
            <TouchableOpacity
              accessibilityLabel={`Next month, ${formatMonthTitle(addMonths(month, 1))}`}
              accessibilityRole="button"
              hitSlop={MONTH_BUTTON_HIT_SLOP}
              style={styles.monthButton}
              onPress={() => setMonth(addMonths(month, 1))}
            >
              <Feather
                name="chevron-right"
                size={18}
                color={colors.accent}
                accessibilityElementsHidden={true}
                importantForAccessibility="no"
              />
            </TouchableOpacity>
          </View>

          {showCalendar ? (
            <>
              <View style={styles.weekdayRow}>
                {WEEKDAYS.map((weekday) => (
                  <Text key={weekday} style={styles.weekdayText}>{weekday}</Text>
                ))}
              </View>

              <View style={styles.calendarGrid}>
                {calendarCells.map((cell, index) => {
                  // Padding cells before the 1st and after the last carry nothing; left visible to a screen
                  // reader they make it count out blank squares before reaching the first real day.
                  if (!cell) {
                    return (
                      <View
                        key={`blank-${index}`}
                        accessibilityElementsHidden={true}
                        importantForAccessibility="no"
                        style={styles.dayCell}
                      />
                    );
                  }

                  const tone = toneForDay(cell, todayKey);
                  return (
                    <TouchableOpacity
                      key={cell.date}
                      // One label for the whole cell — otherwise each day is announced as two loose numbers.
                      accessibilityLabel={`${formatDayLabel(cell.date)}, ${describeDay(cell, tone)}`}
                      accessibilityRole="button"
                      // Seven across leaves each cell ~42dp wide on a 360dp phone (~37dp at 320dp), so the
                      // 44pt minimum is reached horizontally with hitSlop rather than by breaking the grid.
                      hitSlop={{ bottom: 2, left: 4, right: 4, top: 2 }}
                      style={[
                        styles.dayCell,
                        tone === 'checked-in' && styles.dayCellGood,
                        tone === 'unfinished' && styles.dayCellUnfinished,
                        tone === 'future' && styles.dayCellFuture,
                      ]}
                      onPress={() => router.push(`/session-history/${id}/${cell.date}`)}
                      activeOpacity={0.75}
                    >
                      {/* Capped growth: these two numbers sit in a fixed seven-across grid square. */}
                      <Text maxFontSizeMultiplier={1.6} style={styles.dayNumber}>{cell.day}</Text>
                      <Text
                        maxFontSizeMultiplier={1.6}
                        style={[
                          styles.dayCount,
                          tone === 'checked-in' ? styles.dayCountGood : styles.dayCountQuiet,
                        ]}
                      >
                        {tone === 'future' ? '' : cell.count}
                      </Text>
                    </TouchableOpacity>
                  );
                })}
              </View>
            </>
          ) : null}
        </View>

        {monthQuery.isLoading ? <LoadingState message="Loading this month’s check-ins." /> : null}

        {/* Never the server's own error text: this screen's sibling once greeted caregivers with the
            headline "Not found". A failed month is a connection matter, not news about their loved one. */}
        {monthQuery.isError && !hasMonthData ? (
          <ErrorState
            title="We could not load this month just now"
            onRetry={() => void monthQuery.refetch()}
          />
        ) : null}

        {/* Loaded and genuinely quiet — deliberately worded so it cannot be mistaken for the failure above. */}
        {isEmptyMonth ? (
          <EmptyState
            compact
            icon="calendar"
            title="Nothing recorded this month"
            message="Days that have not come round yet stay empty. You can still tap any day to look at it."
          />
        ) : null}

        {showCalendar ? (
          <View style={styles.hintCard}>
            <Feather
              name="mouse-pointer"
              size={14}
              color={colors.accent}
              accessibilityElementsHidden={true}
              importantForAccessibility="no"
            />
            <Text style={styles.hintText}>Tap a day to open that day’s full sessions.</Text>
          </View>
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
}

// Spoken form of a day cell. Stays about the check-in itself, never about the person, and never repeats the
// colour of the square.
type DayTone = 'checked-in' | 'unfinished' | 'quiet' | 'future';

/**
 * A day gets a colour only for something the server actually reported.
 *
 * Previously every cell without a finished session — including days that have not arrived yet — was
 * painted in the needs_attention palette, so opening next month showed a caregiver a whole calendar of
 * alarm-coloured squares. That is the app inventing a status, which it must never do: it renders only what
 * the status engine computed. Days with nothing recorded are now simply quiet, and future days are blank.
 */
function toneForDay(cell: CalendarDay, todayKey: string): DayTone {
  if (cell.date > todayKey) return 'future';
  if (cell.count > 0 && cell.hasCompletedSession) return 'checked-in';
  if (cell.count > 0) return 'unfinished';
  return 'quiet';
}

function describeDay(cell: CalendarDay, tone: DayTone) {
  if (tone === 'checked-in') return cell.count > 1 ? `${cell.count} check-ins` : 'checked in';
  if (tone === 'unfinished') return 'check-in did not finish';
  if (tone === 'future') return 'still to come';
  return 'no check-in recorded';
}

function buildCalendarCells(month: string, days: CalendarDay[]) {
  const byDate = new Map(days.map((day) => [day.date, day]));
  const [year, monthNumber] = month.split('-').map(Number);
  const daysInMonth = new Date(year, monthNumber, 0).getDate();
  const firstDay = new Date(year, monthNumber - 1, 1).getDay();
  const mondayOffset = (firstDay + 6) % 7;
  const cells: Array<CalendarDay | null> = Array.from({ length: mondayOffset }, () => null);

  for (let day = 1; day <= daysInMonth; day++) {
    const date = `${month}-${String(day).padStart(2, '0')}`;
    cells.push(byDate.get(date) || { date, day, count: 0 });
  }

  while (cells.length % 7 !== 0) {
    cells.push(null);
  }

  return cells;
}

function addMonths(monthKey: string, amount: number) {
  const [year, month] = monthKey.split('-').map(Number);
  const date = new Date(year, month - 1 + amount, 1);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
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

function getSingaporeMonthKey(date: Date) {
  return getSingaporeDateKey(date).slice(0, 7);
}

function formatDayLabel(dateKey: string) {
  const [year, month, day] = dateKey.split('-').map(Number);
  return new Date(year, month - 1, day).toLocaleDateString('en-SG', {
    day: 'numeric',
    month: 'long',
  });
}

function formatMonthTitle(monthKey: string) {
  const [year, month] = monthKey.split('-').map(Number);
  return new Date(year, month - 1, 1).toLocaleDateString('en-SG', {
    month: 'long',
    year: 'numeric',
  });
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.surface.page },
  stateWrap: { flex: 1, justifyContent: 'center', paddingHorizontal: spacing.lg },
  content: { paddingBottom: 48, paddingHorizontal: spacing.md, paddingTop: spacing.lg },
  card: {
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: radius.xl,
    borderWidth: 1,
    marginBottom: 14,
    padding: spacing.md,
    ...cardShadow,
  },
  monthHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: spacing.md,
  },
  monthButton: {
    alignItems: 'center',
    backgroundColor: colors.surface.muted,
    borderColor: colors.border.default,
    borderRadius: radius.pill,
    borderWidth: 1,
    height: scaleSize(32),
    justifyContent: 'center',
    width: scaleSize(32),
  },
  monthTitleWrap: { alignItems: 'center', flex: 1 },
  monthTitle: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: scaleSize(18), fontWeight: '500' },
  monthSubtitle: { color: colors.text.tertiary, fontSize: fontSize.body, marginTop: 3 },
  weekdayRow: { flexDirection: 'row', marginBottom: spacing.sm },
  weekdayText: {
    color: colors.text.tertiary,
    flex: 1,
    fontSize: fontSize.caption,
    fontWeight: '700',
    textAlign: 'center',
  },
  calendarGrid: { flexDirection: 'row', flexWrap: 'wrap' },
  dayCell: {
    alignItems: 'center',
    borderColor: colors.border.subtle,
    borderRadius: radius.md,
    borderWidth: 1,
    justifyContent: 'space-between',
    marginBottom: 5,
    marginHorizontal: '0.35%',
    // 44pt floor rather than the old 34–44 aspect-ratio box: each cell is a tap target, and a hard height
    // cap clipped the two numbers as soon as the system font size was turned up.
    minHeight: MIN_TOUCH_TARGET,
    paddingHorizontal: 2,
    paddingVertical: spacing.xs,
    width: '13.58%',
  },
  // Only a day the server reported a finished check-in for gets a colour. A quiet day keeps the plain cell
  // (no alarm palette — the app does not decide that a missed day means anything), an unfinished check-in
  // gets a soft neutral tint, and a day that has not happened yet is faded out.
  // Borders carry the state, so they clear 3:1 against the card behind them (the previous #ABC5A1 was
  // 1.87:1 — a tint an older eye cannot separate from a plain cell). Colour is never the only channel: each
  // cell also shows its count and spells the state out for a screen reader.
  dayCellGood: {
    backgroundColor: '#EEF7EA',
    borderColor: '#79936F',
  },
  dayCellUnfinished: {
    backgroundColor: '#F6EFE5',
    borderColor: '#9C8B70',
  },
  dayCellFuture: { opacity: 0.45 },
  dayNumber: {
    color: colors.text.primary,
    fontSize: fontSize.caption,
    fontWeight: '800',
    textAlign: 'center',
  },
  dayCount: { fontSize: fontSize.body, fontWeight: '900', textAlign: 'center' },
  dayCountGood: { color: '#617A58' },
  dayCountQuiet: { color: colors.text.secondary },
  hintCard: {
    alignItems: 'center',
    backgroundColor: '#F6EFE5',
    borderColor: colors.border.default,
    borderRadius: 12,
    borderWidth: 1,
    flexDirection: 'row',
    gap: spacing.sm,
    padding: 14,
  },
  hintText: { color: colors.text.secondary, flex: 1, fontSize: fontSize.body },
});
