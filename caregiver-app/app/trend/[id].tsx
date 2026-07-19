import React, { useEffect, useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams } from 'expo-router';
import { Feather } from '@expo/vector-icons';
import type { TrendDay } from '../../src/data/mockData';
import { fetchPatientTrend } from '../../src/lib/patientTrendClient';

const SCREEN_WIDTH = Dimensions.get('window').width;

type Range = 7 | 30 | 90;

export default function TrendScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const [range, setRange] = useState<Range>(30);
  const [realTrend, setRealTrend] = useState<TrendDay[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const shouldLoadRealTrend = Boolean(id && /^[0-9a-f]{24}$/i.test(id));
  const realRangeImplemented = range === 7 || range === 30;

  useEffect(() => {
    if (!shouldLoadRealTrend || !realRangeImplemented) {
      setRealTrend([]);
      setIsLoading(false);
      return;
    }

    let isMounted = true;
    const loadTrend = async () => {
      setIsLoading(true);
      try {
        const trend = await fetchPatientTrend(id, range);
        if (isMounted) {
          setRealTrend(trend);
        }
      } catch (err) {
        console.error('[TrendScreen] load patient trend failed', err);
        if (isMounted) {
          setRealTrend([]);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    void loadTrend();
    return () => {
      isMounted = false;
    };
  }, [id, range, realRangeImplemented, shouldLoadRealTrend]);

  if (!shouldLoadRealTrend) {
    return (
      <SafeAreaView style={styles.safe}>
        <View style={styles.placeholder}>
          <Feather name="bar-chart-2" size={28} color="#87566A" />
          <Text style={styles.placeholderTitle}>Bear with us</Text>
          <Text style={styles.placeholderText}>This trend is not ready to show yet.</Text>
        </View>
      </SafeAreaView>
    );
  }

  const trend = realTrend;

  const maxDuration = Math.max(...trend.map(d => d.duration), 1);
  const talkedDays = trend.filter(d => !d.missed).length;
  const avgDuration = talkedDays
    ? Math.round(trend.filter(d => !d.missed).reduce((s, d) => s + d.duration, 0) / talkedDays)
    : 0;

  const summaryText = (() => {
    if (shouldLoadRealTrend && range === 90) {
      return '3-month trend is not available yet.';
    }
    if (isLoading) {
      return 'Loading trend...';
    }
    if (talkedDays >= range * 0.85)
      return `They have been consistently engaged over the past ${range} days. No significant changes detected.`;
    if (talkedDays >= range * 0.6)
      return `They have been mostly engaged with a few quieter days.`;
    return `They have missed several sessions recently. Consider checking in.`;
  })();

  const notable: { date: string; note: string }[] = [];
  let missStreak = 0;
  for (const d of trend) {
    if (d.missed) {
      missStreak++;
      if (missStreak === 2) notable.push({ date: d.date, note: `Missed ${missStreak} sessions in a row` });
    } else {
      if (d.status === 'yellow') notable.push({ date: d.date, note: 'Worth checking - quieter than usual' });
      missStreak = 0;
    }
  }

  const CHART_HEIGHT = 120;
  const barW = Math.max(3, (SCREEN_WIDTH - 56) / Math.max(trend.length, 1) - 2);

  function barColor(d: TrendDay): string {
    if (d.missed) return '#E8E0D6';
    if (d.status === 'yellow') return '#C5AA80';
    if (d.status === 'red') return '#C09898';
    return '#B9AA99';
  }

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.rangeRow}>
          {([7, 30, 90] as Range[]).map(r => (
            <TouchableOpacity
              key={r}
              style={[styles.rangePill, range === r && styles.rangePillActive]}
              onPress={() => setRange(r)}
            >
              <Text style={[styles.rangePillText, range === r && styles.rangePillTextActive]}>
                {r === 90 ? '3 months' : `${r} days`}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <View style={styles.card}>
          <Text style={styles.summaryText}>{summaryText}</Text>
          {range === 90 && shouldLoadRealTrend ? (
            <Text style={styles.summaryStats}>Coming soon</Text>
          ) : (
            <Text style={styles.summaryStats}>
              Talked {talkedDays} of {range} days · Avg {Math.floor(avgDuration / 60)}m {avgDuration % 60}s
            </Text>
          )}
        </View>

        <View style={styles.card}>
          {range === 90 && shouldLoadRealTrend ? (
            <View style={[styles.emptyChart, { height: CHART_HEIGHT }]}>
              <Text style={styles.emptyChartText}>3-month view coming soon</Text>
            </View>
          ) : (
            <>
              <View style={[styles.chart, { height: CHART_HEIGHT }]}>
                {trend.map((d, i) => {
                  const h = d.missed ? 3 : Math.max(6, (d.duration / maxDuration) * CHART_HEIGHT);
                  return (
                    <View key={i} style={[styles.barWrap, { height: CHART_HEIGHT }]}>
                      <View style={[styles.bar, { height: h, width: barW, backgroundColor: barColor(d) }]} />
                    </View>
                  );
                })}
              </View>
              <View style={styles.chartLabels}>
                <Text style={styles.chartLabel}>{trend[0]?.date?.slice(5)}</Text>
                <Text style={styles.chartLabel}>{trend[Math.floor(trend.length / 2)]?.date?.slice(5)}</Text>
                <Text style={styles.chartLabel}>{trend[trend.length - 1]?.date?.slice(5)}</Text>
              </View>
            </>
          )}
          <View style={styles.legend}>
            <LegendDot color="#B9AA99" label="Doing well" />
            <LegendDot color="#C5AA80" label="Worth checking" />
            <LegendDot color="#E8E0D6" label="Missed" />
          </View>
        </View>

        {notable.length > 0 && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Notable events</Text>
            {notable.map((n, i) => (
              <View key={i} style={[styles.notableRow, i < notable.length - 1 && styles.notableRowBorder]}>
                <Text style={styles.notableDate}>{n.date.slice(5)}</Text>
                <Text style={styles.notableNote}>{n.note}</Text>
              </View>
            ))}
          </View>
        )}

        <View style={styles.v2Note}>
          <Feather name="info" size={14} color="#B2844B" />
          <Text style={styles.v2NoteText}>
            Cognitive Stability Score with trend arrow is coming in a future update after user validation.
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <View style={styles.legendItem}>
      <View style={[styles.legendDot, { backgroundColor: color }]} />
      <Text style={styles.legendLabel}>{label}</Text>
    </View>
  );
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
  content: { paddingHorizontal: 20, paddingBottom: 48, paddingTop: 16 },
  rangeRow: { flexDirection: 'row', gap: 8, marginBottom: 16 },
  rangePill: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 999,
    backgroundColor: '#F4F0EA',
    borderWidth: 1,
    borderColor: '#E7DED2',
  },
  rangePillActive: { backgroundColor: '#87566A', borderColor: '#87566A' },
  rangePillText: { fontSize: 13, color: '#756C64', fontWeight: '600' },
  rangePillTextActive: { color: '#FFFFFF' },
  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#E7DED2',
    padding: 18,
    marginBottom: 14,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.035,
    shadowRadius: 10,
    elevation: 2,
  },
  summaryText: { fontSize: 14, color: '#756C64', lineHeight: 21 },
  summaryStats: { fontSize: 13, color: '#A69C92', marginTop: 8 },
  chart: { flexDirection: 'row', alignItems: 'flex-end', gap: 1 },
  emptyChart: { alignItems: 'center', justifyContent: 'center' },
  emptyChartText: { color: '#A69C92', fontSize: 13, fontWeight: '600' },
  barWrap: { justifyContent: 'flex-end' },
  bar: { borderRadius: 3 },
  chartLabels: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 8 },
  chartLabel: { fontSize: 11, color: '#A69C92' },
  legend: { flexDirection: 'row', gap: 16, marginTop: 14, flexWrap: 'wrap' },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  legendDot: { width: 10, height: 10, borderRadius: 999 },
  legendLabel: { fontSize: 12, color: '#756C64' },
  cardTitle: {
    fontSize: 12,
    fontWeight: '600',
    color: '#A69C92',
    textTransform: 'uppercase',
    letterSpacing: 0.6,
    marginBottom: 10,
  },
  notableRow: { flexDirection: 'row', gap: 14, paddingVertical: 8 },
  notableRowBorder: { borderBottomWidth: 1, borderBottomColor: '#F3EDE6' },
  notableDate: { fontSize: 13, color: '#A69C92', width: 44 },
  notableNote: { fontSize: 13, color: '#756C64', flex: 1 },
  v2Note: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    backgroundColor: '#F6EFE5',
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: '#E7DED2',
  },
  v2NoteText: { fontSize: 13, color: '#B2844B', lineHeight: 18, flex: 1 },
});
