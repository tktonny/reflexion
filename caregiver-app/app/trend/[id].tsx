import { useQuery } from '@tanstack/react-query';
import React, { useCallback, useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity, Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useLocalSearchParams } from 'expo-router';
import { Feather } from '@expo/vector-icons';
import { EmptyState, ErrorState, LoadingState } from '../../src/components/ScreenState';
import type { TrendDay } from '../../src/lib/patientTrendClient';
import { fetchPatientTrend } from '../../src/lib/patientTrendClient';
import { colors, spacing, radius, fontSize, cardShadow, MIN_TOUCH_TARGET } from '../../src/theme';

const SCREEN_WIDTH = Dimensions.get('window').width;

// Chart fills — a data-visualisation language of its own, deliberately outside the theme, which covers UI
// chrome. Not the status palette either: src/lib/v1Status.ts owns that.
const CHART_FILL = {
  doingWell: '#B9AA99',
  worthChecking: '#C5AA80',
  attention: '#C09898',
  missed: '#E8E0D6',
} as const;

type Range = 7 | 30 | 90;
type ImplementedRange = 7 | 30;

export default function TrendScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const [range, setRange] = useState<Range>(30);

  const shouldLoadRealTrend = Boolean(id && /^[0-9a-f]{24}$/i.test(id));
  const realRangeImplemented = range === 7 || range === 30;
  const trendQuery = useQuery({
    enabled: shouldLoadRealTrend && realRangeImplemented,
    queryKey: ['patientTrend', id, range],
    queryFn: () => fetchPatientTrend(id, range as ImplementedRange),
  });
  const { refetch: refetchTrend } = trendQuery;

  useFocusEffect(
    useCallback(() => {
      if (shouldLoadRealTrend && realRangeImplemented) {
        void refetchTrend();
      }
    }, [realRangeImplemented, refetchTrend, shouldLoadRealTrend]),
  );

  if (!shouldLoadRealTrend) {
    return (
      <SafeAreaView style={styles.safe}>
        <View style={styles.placeholder}>
          <EmptyState
            icon="bar-chart-2"
            title="Bear with us"
            message="This trend is not ready to show yet."
          />
        </View>
      </SafeAreaView>
    );
  }

  const trend = trendQuery.data || [];

  const maxDuration = Math.max(...trend.map(d => d.duration), 1);
  const talkedDays = trend.filter(d => !d.missed).length;
  const avgDuration = talkedDays
    ? Math.round(trend.filter(d => !d.missed).reduce((s, d) => s + d.duration, 0) / talkedDays)
    : 0;

  const summaryText = (() => {
    if (shouldLoadRealTrend && range === 90) {
      return '3-month trend is not available yet.';
    }
    if (trendQuery.isLoading || trendQuery.isFetching) {
      return 'Loading trend...';
    }
    // Deliberately a count, not a verdict. This used to grade the period locally — "No significant changes
    // detected" / "They have missed several sessions recently. Consider checking in." — which is the app
    // computing a status and handing out advice, both of which belong to the server's status engine. It also
    // read as alarming for a mirror paired yesterday, whose every earlier day is legitimately blank.
    return `Check-ins on ${talkedDays} of the last ${range} days.`;
  })();

  // Factual observations only, no advice: the day-level signal comes from the server, and this list just
  // describes it.
  const notable: { date: string; note: string }[] = [];
  let missStreak = 0;
  for (const d of trend) {
    if (d.missed) {
      missStreak++;
      if (missStreak === 2) notable.push({ date: d.date, note: 'No check-in two days running' });
    } else {
      if (d.status === 'yellow') notable.push({ date: d.date, note: 'A shorter check-in than usual' });
      missStreak = 0;
    }
  }

  const CHART_HEIGHT = 120;
  const barW = Math.max(3, (SCREEN_WIDTH - 56) / Math.max(trend.length, 1) - 2);

  function barColor(d: TrendDay): string {
    if (d.missed) return CHART_FILL.missed;
    if (d.status === 'yellow') return CHART_FILL.worthChecking;
    if (d.status === 'red') return CHART_FILL.attention;
    return CHART_FILL.doingWell;
  }

  // The 3-month range has no endpoint yet, so its query is deliberately disabled — it must not be mistaken
  // for a range that came back empty.
  const comingSoon = range === 90;
  const hasTrend = trend.length > 0;
  const showPlaceholder = !comingSoon && !hasTrend;

  // One label for the whole chart. Each bar is a bare View, but grouping them also stops a reader from
  // walking 30 unlabelled stops before reaching the legend.
  const firstDayLabel = formatDayLabel(trend[0]?.date);
  const lastDayLabel = formatDayLabel(trend[trend.length - 1]?.date);
  const chartAccessibilityLabel = [
    `Check-in chart, ${trend.length} days, oldest first.`,
    firstDayLabel && lastDayLabel ? `${firstDayLabel} to ${lastDayLabel}.` : '',
    `${talkedDays} of those days had a check-in.`,
  ].filter(Boolean).join(' ');

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.content}>
        {/* Segmented control: without a selected state all three pills read the same to a screen reader. */}
        <View accessibilityRole="tablist" style={styles.rangeRow}>
          {([7, 30, 90] as Range[]).map(r => (
            <TouchableOpacity
              key={r}
              accessibilityLabel={r === 90 ? 'Last 3 months' : `Last ${r} days`}
              accessibilityRole="tab"
              accessibilityState={{ selected: range === r }}
              activeOpacity={0.82}
              style={[styles.rangePill, range === r && styles.rangePillActive]}
              onPress={() => setRange(r)}
            >
              <Text style={[styles.rangePillText, range === r && styles.rangePillTextActive]}>
                {r === 90 ? '3 months' : `${r} days`}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {showPlaceholder ? (
          // Stands in for the summary and chart cards, because those two are derived locally from the rows.
          // With zero rows the summary line reads "They have missed several sessions recently" — an alarming
          // claim about a person generated by a request that simply did not come back.
          <View accessibilityLiveRegion="polite">
            <TrendPlaceholder
              hasError={Boolean(trendQuery.error)}
              isLoading={trendQuery.isLoading || trendQuery.isFetching}
              range={range}
              onRetry={() => void refetchTrend()}
            />
          </View>
        ) : (
          <>
            <View style={styles.card}>
              {/* Announce the new summary when the caregiver switches range, rather than leaving them to
                  hunt for what changed. */}
              <Text accessibilityLiveRegion="polite" style={styles.summaryText}>{summaryText}</Text>
              {comingSoon ? (
                <Text style={styles.summaryStats}>Coming soon</Text>
              ) : (
                <Text style={styles.summaryStats}>
                  Talked {talkedDays} of {range} days · Avg {Math.floor(avgDuration / 60)}m {avgDuration % 60}s
                </Text>
              )}
            </View>

            <View style={styles.card}>
              {comingSoon ? (
                <View style={[styles.emptyChart, { minHeight: CHART_HEIGHT }]}>
                  <Text style={styles.emptyChartText}>3-month view coming soon</Text>
                </View>
              ) : (
                <>
                  <View
                    accessible
                    accessibilityLabel={chartAccessibilityLabel}
                    accessibilityRole="image"
                    style={[styles.chart, { height: CHART_HEIGHT }]}
                  >
                    {trend.map((d, i) => {
                      const h = d.missed ? 3 : Math.max(6, (d.duration / maxDuration) * CHART_HEIGHT);
                      return (
                        <View key={i} style={[styles.barWrap, { height: CHART_HEIGHT }]}>
                          <View style={[styles.bar, { height: h, width: barW, backgroundColor: barColor(d) }]} />
                        </View>
                      );
                    })}
                  </View>
                  {/* Axis dates are a visual scale only: the grouped chart label above already says the span,
                      and left announced they arrive as three "06-09" fragments. */}
                  <View
                    accessibilityElementsHidden
                    importantForAccessibility="no-hide-descendants"
                    style={styles.chartLabels}
                  >
                    <Text maxFontSizeMultiplier={1.6} style={styles.chartLabel}>{trend[0]?.date?.slice(5)}</Text>
                    <Text maxFontSizeMultiplier={1.6} style={styles.chartLabel}>
                      {trend[Math.floor(trend.length / 2)]?.date?.slice(5)}
                    </Text>
                    <Text maxFontSizeMultiplier={1.6} style={styles.chartLabel}>
                      {trend[trend.length - 1]?.date?.slice(5)}
                    </Text>
                  </View>
                </>
              )}
              <View style={styles.legend}>
                <LegendDot color={CHART_FILL.doingWell} label="Doing well" />
                <LegendDot color={CHART_FILL.worthChecking} label="Worth checking" />
                <LegendDot color={CHART_FILL.missed} label="Missed" />
              </View>
            </View>
          </>
        )}

        {notable.length > 0 && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Notable events</Text>
            {notable.map((n, i) => (
              // Grouped: date and note are one thought, and read apart the date is just loose digits.
              <View
                key={i}
                accessible
                accessibilityLabel={`${formatDayLabel(n.date) || n.date}. ${n.note}`}
                style={[styles.notableRow, i < notable.length - 1 && styles.notableRowBorder]}
              >
                <Text style={styles.notableDate}>{n.date.slice(5)}</Text>
                <Text style={styles.notableNote}>{n.note}</Text>
              </View>
            ))}
          </View>
        )}

        <View accessible style={styles.v2Note}>
          <Feather
            accessibilityElementsHidden
            importantForAccessibility="no"
            name="info"
            size={14}
            color="#8B673A"
          />
          <Text style={styles.v2NoteText}>
            A longer-term wellbeing overview is coming in a future update, once we have validated it with families.
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

/**
 * Loading / failed / genuinely-empty are three different situations and the caregiver deserves to be told
 * which — until now all three rendered the same undifferentiated box. A failure NEVER shows the server's
 * error text: that is how a sibling screen came to greet people with the headline "Not found".
 */
function TrendPlaceholder({
  hasError,
  isLoading,
  range,
  onRetry,
}: {
  hasError: boolean;
  isLoading: boolean;
  range: Range;
  onRetry: () => void;
}) {
  // Loading wins over a stale error so a retry visibly does something.
  if (isLoading) {
    return <LoadingState message={`Fetching the last ${range} days.`} />;
  }

  if (hasError) {
    return <ErrorState onRetry={onRetry} />;
  }

  return (
    <EmptyState
      icon="bar-chart-2"
      title="No days to show yet"
      message={`Once a few check-ins have happened, the last ${range} days will appear here.`}
    />
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    // Grouped so the reader says "Doing well" once instead of stopping on the swatch first.
    <View accessible accessibilityLabel={label} style={styles.legendItem}>
      <View
        accessibilityElementsHidden
        importantForAccessibility="no"
        style={[styles.legendDot, { backgroundColor: color }]}
      />
      <Text style={styles.legendLabel}>{label}</Text>
    </View>
  );
}

/** "9 Jun" rather than the raw "2026-06-09", which a screen reader spells out digit by digit. */
function formatDayLabel(value?: string) {
  if (!value) {
    return '';
  }

  // Parsed as a LOCAL date, matching app/session-history/[id].tsx. `new Date('2026-06-09')` is UTC
  // midnight, and Intl then renders it in the device timezone — so on any device behind UTC the label and
  // the screen-reader announcement both slid to the previous day.
  const [year, month, day] = value.split('-').map(Number);
  if (!year || !month || !day) {
    return '';
  }

  return new Intl.DateTimeFormat('en-SG', { day: 'numeric', month: 'short' }).format(new Date(year, month - 1, day));
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.surface.page },
  placeholder: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: spacing.xl,
  },
  content: { paddingHorizontal: spacing.xl, paddingBottom: 48, paddingTop: spacing.lg },
  // Wraps rather than squeezing the pills off-screen at large system font sizes.
  rangeRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, marginBottom: spacing.lg },
  rangePill: {
    alignItems: 'center',
    justifyContent: 'center',
    // 44pt: these three pills were ~34pt tall, small for a thumb on a phone held in one hand.
    minHeight: MIN_TOUCH_TARGET,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: radius.pill,
    backgroundColor: colors.surface.muted,
    borderWidth: 1,
    borderColor: colors.border.default,
  },
  rangePillActive: { backgroundColor: colors.accent, borderColor: colors.accent },
  rangePillText: { fontSize: fontSize.body, color: colors.text.secondary, fontWeight: '600' },
  rangePillTextActive: { color: colors.text.onAccent },
  card: {
    backgroundColor: colors.surface.card,
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border.default,
    padding: 18,
    marginBottom: 14,
    ...cardShadow,
  },
  summaryText: { fontSize: fontSize.bodyLarge, color: colors.text.secondary, lineHeight: 21 },
  summaryStats: { fontSize: fontSize.body, color: colors.text.tertiary, marginTop: spacing.sm },
  chart: { flexDirection: 'row', alignItems: 'flex-end', gap: 1 },
  emptyChart: { alignItems: 'center', justifyContent: 'center' },
  emptyChartText: { color: colors.text.tertiary, fontSize: fontSize.body, fontWeight: '600' },
  barWrap: { justifyContent: 'flex-end' },
  bar: { borderRadius: 3 },
  chartLabels: { flexDirection: 'row', justifyContent: 'space-between', marginTop: spacing.sm },
  chartLabel: { fontSize: fontSize.caption, color: colors.text.tertiary },
  legend: { flexDirection: 'row', gap: spacing.lg, marginTop: 14, flexWrap: 'wrap' },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  legendDot: { width: 10, height: 10, borderRadius: radius.pill },
  legendLabel: { fontSize: fontSize.caption, color: colors.text.secondary },
  cardTitle: {
    fontSize: fontSize.caption,
    fontWeight: '600',
    color: colors.text.tertiary,
    textTransform: 'uppercase',
    letterSpacing: 0.6,
    marginBottom: 10,
  },
  notableRow: { flexDirection: 'row', gap: 14, paddingVertical: spacing.sm },
  notableRowBorder: { borderBottomWidth: 1, borderBottomColor: colors.border.subtle },
  // minWidth, not width: at large font sizes a fixed 44 clipped the date it is meant to show.
  notableDate: { fontSize: fontSize.body, color: colors.text.tertiary, minWidth: 44 },
  notableNote: { fontSize: fontSize.body, color: colors.text.secondary, flex: 1 },
  v2Note: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.sm,
    backgroundColor: '#F6EFE5',
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: colors.border.default,
  },
  v2NoteText: { fontSize: fontSize.body, color: '#8B673A', lineHeight: 19, flex: 1 },
});
