import React, { useEffect, useMemo, useState } from 'react';
import {
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Feather } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { getApiUrl } from '../../src/lib/apiUrl';

type CalendarDay = {
  date: string;
  day: number;
  count: number;
  completedCount?: number;
  hasCompletedSession?: boolean;
};

const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

export default function SessionHistoryScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const [month, setMonth] = useState(getSingaporeMonthKey(new Date()));
  const [days, setDays] = useState<CalendarDay[]>([]);
  const [isLoadingMonth, setIsLoadingMonth] = useState(false);
  const shouldLoadRealSession = Boolean(id && /^[0-9a-f]{24}$/i.test(id));
  const calendarCells = useMemo(() => buildCalendarCells(month, days), [days, month]);
  const totalSessions = days.reduce((sum, day) => sum + day.count, 0);

  useEffect(() => {
    if (!shouldLoadRealSession) return;

    let isMounted = true;
    async function loadMonth() {
      setIsLoadingMonth(true);
      try {
        const response = await fetch(
          getApiUrl(`/api/conversation-session-counts?id=${encodeURIComponent(id)}&month=${encodeURIComponent(month)}`),
        );
        const body = await response.json();
        if (!response.ok) {
          throw new Error(body?.error || 'Unable to load session history.');
        }
        if (isMounted) {
          setDays(Array.isArray(body?.days) ? body.days : []);
        }
      } catch (err) {
        console.error('[SessionHistoryScreen] load month failed', err);
        if (isMounted) {
          setDays([]);
        }
      } finally {
        if (isMounted) {
          setIsLoadingMonth(false);
        }
      }
    }

    void loadMonth();
    return () => {
      isMounted = false;
    };
  }, [id, month, shouldLoadRealSession]);

  if (!shouldLoadRealSession) {
    return (
      <SafeAreaView style={styles.safe}>
        <View style={styles.placeholder}>
          <Feather name="calendar" size={28} color="#87566A" />
          <Text style={styles.placeholderTitle}>Bear with us</Text>
          <Text style={styles.placeholderText}>This session history is not ready to show yet.</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.card}>
          <View style={styles.monthHeader}>
            <TouchableOpacity style={styles.monthButton} onPress={() => setMonth(addMonths(month, -1))}>
              <Feather name="chevron-left" size={18} color="#87566A" />
            </TouchableOpacity>
            <View style={styles.monthTitleWrap}>
              <Text style={styles.monthTitle}>{formatMonthTitle(month)}</Text>
              <Text style={styles.monthSubtitle}>
                {isLoadingMonth ? 'Loading sessions...' : `${totalSessions} sessions this month`}
              </Text>
            </View>
            <TouchableOpacity style={styles.monthButton} onPress={() => setMonth(addMonths(month, 1))}>
              <Feather name="chevron-right" size={18} color="#87566A" />
            </TouchableOpacity>
          </View>

          <View style={styles.weekdayRow}>
            {WEEKDAYS.map((weekday) => (
              <Text key={weekday} style={styles.weekdayText}>{weekday}</Text>
            ))}
          </View>

          <View style={styles.calendarGrid}>
            {calendarCells.map((cell, index) => {
              if (!cell) return <View key={`blank-${index}`} style={styles.dayCell} />;

              const isGoodDay = cell.count > 0 && Boolean(cell.hasCompletedSession);
              return (
                <TouchableOpacity
                  key={cell.date}
                  style={[
                    styles.dayCell,
                    isGoodDay ? styles.dayCellGood : styles.dayCellNeedsAttention,
                  ]}
                  onPress={() => router.push(`/session-history/${id}/${cell.date}`)}
                  activeOpacity={0.75}
                >
                  <Text style={styles.dayNumber}>{cell.day}</Text>
                  <Text style={[styles.dayCount, isGoodDay ? styles.dayCountGood : styles.dayCountNeedsAttention]}>
                    {cell.count}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>
        </View>

        <View style={styles.hintCard}>
          <Feather name="mouse-pointer" size={14} color="#87566A" />
          <Text style={styles.hintText}>Tap a day to open that day’s full sessions.</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
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

function formatMonthTitle(monthKey: string) {
  const [year, month] = monthKey.split('-').map(Number);
  return new Date(year, month - 1, 1).toLocaleDateString('en-SG', {
    month: 'long',
    year: 'numeric',
  });
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F8F3EC' },
  placeholder: {
    alignItems: 'center',
    flex: 1,
    gap: 8,
    justifyContent: 'center',
    paddingHorizontal: 28,
  },
  placeholderTitle: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 24, fontWeight: '500' },
  placeholderText: { color: '#756C64', fontSize: 15, lineHeight: 22, textAlign: 'center' },
  content: { paddingBottom: 48, paddingHorizontal: 20, paddingTop: 16 },
  card: {
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 16,
    borderWidth: 1,
    elevation: 2,
    marginBottom: 14,
    padding: 18,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.035,
    shadowRadius: 10,
  },
  monthHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  monthButton: {
    alignItems: 'center',
    backgroundColor: '#F4F0EA',
    borderColor: '#E7DED2',
    borderRadius: 999,
    borderWidth: 1,
    height: 36,
    justifyContent: 'center',
    width: 36,
  },
  monthTitleWrap: { alignItems: 'center', flex: 1 },
  monthTitle: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 20, fontWeight: '500' },
  monthSubtitle: { color: '#A69C92', fontSize: 12, marginTop: 3 },
  weekdayRow: { flexDirection: 'row', marginBottom: 8 },
  weekdayText: {
    color: '#A69C92',
    flex: 1,
    fontSize: 11,
    fontWeight: '700',
    textAlign: 'center',
  },
  calendarGrid: { flexDirection: 'row', flexWrap: 'wrap' },
  dayCell: {
    alignItems: 'center',
    aspectRatio: 1,
    borderColor: '#F3EDE6',
    borderRadius: 14,
    borderWidth: 1,
    justifyContent: 'space-between',
    marginBottom: 6,
    marginHorizontal: '0.7%',
    maxHeight: 92,
    minHeight: 44,
    paddingHorizontal: 3,
    paddingVertical: 5,
    width: '12.88%',
  },
  dayCellGood: {
    backgroundColor: '#EEF7EA',
    borderColor: '#ABC5A1',
  },
  dayCellNeedsAttention: {
    backgroundColor: '#FBEDEA',
    borderColor: '#E7B8B1',
  },
  dayNumber: { color: '#2B2522', fontSize: 13, fontWeight: '800', textAlign: 'center' },
  dayCount: { fontSize: 16, fontWeight: '900', lineHeight: 19, textAlign: 'center' },
  dayCountGood: { color: '#617A58' },
  dayCountNeedsAttention: { color: '#B45F56' },
  hintCard: {
    alignItems: 'center',
    backgroundColor: '#F6EFE5',
    borderColor: '#E7DED2',
    borderRadius: 12,
    borderWidth: 1,
    flexDirection: 'row',
    gap: 8,
    padding: 14,
  },
  hintText: { color: '#756C64', flex: 1, fontSize: 13 },
});
