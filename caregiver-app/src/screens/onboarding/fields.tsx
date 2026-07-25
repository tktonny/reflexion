import React from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { colors, fontSize, MIN_TOUCH_TARGET, radius, spacing } from '../../theme';

export function Label({ children }: { children: React.ReactNode }) {
  return <Text style={fieldStyles.label}>{children}</Text>;
}

// Chips only look selected (filled pill). Screen readers need accessibilityState.selected to say so, and
// `groupLabel` exists because several grids on the same step offer a chip literally labelled "Other" —
// without it a caregiver hears the same word three times with no way to tell which question it answers.
export function OptionGrid<T extends string | boolean>({
  groupLabel,
  options,
  selected,
  onSelect,
}: {
  groupLabel?: string;
  options: { value: T; label: string }[];
  selected: T;
  onSelect: (value: T) => void;
}) {
  return (
    <View style={fieldStyles.pillRow}>
      {options.map((option) => {
        const isSelected = option.value === selected;
        return (
          <TouchableOpacity
            accessibilityLabel={groupLabel ? `${groupLabel}: ${option.label}` : option.label}
            accessibilityRole="button"
            accessibilityState={{ selected: isSelected }}
            key={String(option.value)}
            onPress={() => onSelect(option.value)}
            style={[fieldStyles.pill, isSelected && fieldStyles.pillActive]}
          >
            <Text style={[fieldStyles.pillText, isSelected && fieldStyles.pillTextActive]}>
              {option.label}
            </Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
}

export function MultiOptionGrid<T extends string>({
  groupLabel,
  options,
  selected,
  onToggle,
}: {
  groupLabel?: string;
  options: { value: T; label: string }[];
  selected: T[];
  onToggle: (value: T) => void;
}) {
  return (
    <View style={fieldStyles.pillRow}>
      {options.map((option) => {
        const isSelected = selected.includes(option.value);
        return (
          <TouchableOpacity
            accessibilityLabel={groupLabel ? `${groupLabel}: ${option.label}` : option.label}
            accessibilityRole="button"
            accessibilityState={{ selected: isSelected }}
            key={option.value}
            onPress={() => onToggle(option.value)}
            style={[fieldStyles.pill, isSelected && fieldStyles.pillActive]}
          >
            <Text style={[fieldStyles.pillText, isSelected && fieldStyles.pillTextActive]}>
              {option.label}
            </Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
}

// Exported because every step renders the same text inputs; the label and pill styles are only read by
// the components above.
export const fieldStyles = StyleSheet.create({
  label: {
    color: colors.text.primary,
    fontSize: fontSize.body,
    fontWeight: '800',
    marginBottom: 7,
    marginTop: spacing.lg,
  },
  input: {
    backgroundColor: colors.surface.card,
    borderColor: colors.border.strong,
    borderRadius: radius.sm,
    borderWidth: 1,
    color: colors.text.primary,
    fontSize: fontSize.subheading,
    minHeight: 48,
    paddingHorizontal: 14,
    paddingVertical: spacing.md,
  },
  textArea: {
    minHeight: 84,
    textAlignVertical: 'top',
  },
  pillRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  pill: {
    backgroundColor: colors.surface.card,
    borderColor: colors.border.strong,
    borderRadius: radius.sm,
    borderWidth: 1,
    justifyContent: 'center',
    // Padding alone left these ~38pt tall; every chip on this form is a real answer being tapped once.
    minHeight: MIN_TOUCH_TARGET,
    paddingHorizontal: 13,
    paddingVertical: 10,
  },
  pillActive: {
    backgroundColor: colors.accent,
    borderColor: colors.accent,
  },
  pillText: {
    color: colors.text.secondary,
    fontSize: fontSize.body,
    fontWeight: '800',
  },
  pillTextActive: { color: colors.text.onAccent },
});
