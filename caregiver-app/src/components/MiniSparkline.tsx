import React from 'react';
import { View, StyleSheet } from 'react-native';
import type { TrendDay } from '../lib/patientTrendClient';
import { scaleSize } from '../theme';

interface Props {
  data: TrendDay[];
  days?: number;
  height?: number;
}

// Sparkline fills — data-visualisation colour, a separate language from the theme's UI chrome, and not the
// status palette either (src/lib/v1Status.ts owns that).
const SPARKLINE_FILL = {
  talked: '#B9AA99',
  missed: '#E8E0D6',
} as const;

// A bar chart carries nothing for a screen reader, so the whole strip is exposed as a single readable
// sentence and the individual bars are hidden. Wording stays plain and non-clinical: it reports whether a
// check-in happened, never how "good" it was.
function describeWeek(slice: TrendDay[]): string {
  if (!slice.length) return 'No check-ins recorded yet.';
  const talked = slice.filter((day) => !day.missed).length;
  const missed = slice.length - talked;
  const missedPart = missed ? ` ${missed} ${missed === 1 ? 'day' : 'days'} without one.` : '';
  return `Check-ins on ${talked} of the last ${slice.length} ${slice.length === 1 ? 'day' : 'days'}.${missedPart}`;
}

export default function MiniSparkline({ data, days = 7, height = scaleSize(32) }: Props) {
  const slice = data.slice(-days);
  const maxDuration = Math.max(...slice.map(d => d.duration), 1);

  return (
    <View
      accessible
      accessibilityRole="image"
      accessibilityLabel={describeWeek(slice)}
      style={[styles.container, { height }]}
    >
      {slice.map((day, i) => {
        const barH = day.missed ? 3 : Math.max(4, (day.duration / maxDuration) * height);
        return (
          <View key={i} importantForAccessibility="no" style={styles.barWrapper}>
            <View
              style={[
                styles.bar,
                { height: barH, backgroundColor: day.missed ? SPARKLINE_FILL.missed : SPARKLINE_FILL.talked },
              ]}
            />
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: scaleSize(3),
  },
  barWrapper: { flex: 1, justifyContent: 'flex-end' },
  bar: { borderRadius: 3, width: '100%' },
});
