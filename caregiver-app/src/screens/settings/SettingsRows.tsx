import { Feather } from '@expo/vector-icons';
import React from 'react';
import { Image, StyleSheet, Switch, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { colors, fontFamily, fontSize, radius, spacing } from '../../theme';
import { getInitials } from './helpers';
import { PILL_HIT_SLOP, pillStyles } from './optionPills';

export function SectionHeader({ title }: { title: string }) {
  return <Text accessibilityRole="header" style={styles.sectionHeader}>{title}</Text>;
}

export function SettingRow({ label, value }: { label: string; value: string }) {
  return (
    // One announcement ("Email, mum@example.com") instead of two disconnected fragments.
    <View accessible accessibilityLabel={`${label}, ${value}`} style={styles.row}>
      <Text style={styles.rowLabel}>{label}</Text>
      {/* No numberOfLines: at large system font sizes a truncated email address is unusable. */}
      <Text style={styles.rowValue}>{value}</Text>
    </View>
  );
}

export function InputRow({
  keyboardType = 'default',
  label,
  onChangeText,
  value,
}: {
  keyboardType?: 'default' | 'phone-pad' | 'numeric';
  label: string;
  onChangeText: (value: string) => void;
  value: string;
}) {
  return (
    <View style={styles.inputRow}>
      <Text style={styles.rowLabel}>{label}</Text>
      <TextInput
        accessibilityLabel={label}
        keyboardType={keyboardType}
        onChangeText={onChangeText}
        style={styles.inlineInput}
        value={value}
      />
    </View>
  );
}

export function ActionRow({
  fallbackName,
  imageUrl,
  label,
  value,
  onPress,
}: {
  fallbackName?: string;
  imageUrl?: string;
  label: string;
  value?: string;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity
      // One label for the whole row, so a loved one reads as "Mum, English" rather than three fragments.
      accessibilityLabel={value ? `${label}, ${value}` : label}
      accessibilityRole="button"
      style={styles.row}
      onPress={onPress}
      activeOpacity={0.7}
    >
      <View style={styles.rowLeft}>
        {fallbackName ? (
          // Avatar/initials only repeat the name already in the row label.
          <View accessibilityElementsHidden importantForAccessibility="no" style={styles.patientAvatar}>
            {imageUrl ? (
              <Image source={{ uri: imageUrl }} style={styles.patientAvatarImage} />
            ) : (
              // Fixed 34pt circle: initials cannot grow without spilling out of it.
              <Text maxFontSizeMultiplier={1.6} style={styles.patientAvatarText}>{getInitials(fallbackName)}</Text>
            )}
          </View>
        ) : null}
        <Text style={styles.rowLabel}>{label}</Text>
      </View>
      <View style={styles.rowRight}>
        {value ? <Text style={styles.rowValue}>{value}</Text> : null}
        <Feather
          accessibilityElementsHidden
          importantForAccessibility="no"
          name="chevron-right"
          size={16}
          color={colors.textDecorative}
        />
      </View>
    </TouchableOpacity>
  );
}

export function SwitchRow({
  disabled = false,
  label,
  value,
  onChange,
}: {
  disabled?: boolean;
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <View style={styles.row}>
      <Text style={styles.rowLabel}>{label}</Text>
      <Switch
        accessibilityLabel={label}
        disabled={disabled}
        value={value}
        onValueChange={onChange}
        trackColor={{ false: colors.border.strong, true: colors.accent }}
        thumbColor={colors.surface.card}
      />
    </View>
  );
}

export function PickerRow({ label, options, selected, onSelect }: {
  label: string;
  options: { value: string; label: string }[];
  selected: string;
  onSelect: (v: string) => void;
}) {
  return (
    <View style={styles.pickerBlock}>
      <Text style={styles.rowLabel}>{label}</Text>
      <View style={pillStyles.pickerOptions}>
        {options.map(o => (
          <TouchableOpacity
            key={o.value}
            // The group name is in the label so "Evening (7pm)" is not announced without saying of what.
            accessibilityLabel={`${label}, ${o.label}`}
            accessibilityRole="button"
            accessibilityState={{ selected: selected === o.value }}
            hitSlop={PILL_HIT_SLOP}
            style={[pillStyles.pill, selected === o.value && pillStyles.pillActive]}
            onPress={() => onSelect(o.value)}
          >
            <Text style={[pillStyles.pillText, selected === o.value && pillStyles.pillTextActive]}>{o.label}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  sectionHeader: {
    fontSize: fontSize.caption,
    fontWeight: '600',
    color: colors.text.tertiary,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    paddingHorizontal: spacing.xl,
    paddingTop: spacing.xxl,
    paddingBottom: spacing.sm,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: colors.surface.card,
    paddingHorizontal: spacing.xl,
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.subtle,
  },
  inputRow: {
    backgroundColor: colors.surface.card,
    borderBottomColor: colors.border.subtle,
    borderBottomWidth: 1,
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.md,
  },
  // flexShrink 1 (was 0): at large system font sizes a long label has to wrap instead of shoving the value
  // off the right edge of the row.
  rowLabel: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.subheading, fontWeight: '500' },
  rowValue: {
    color: colors.text.tertiary,
    flex: 1,
    fontSize: fontSize.subheading,
    lineHeight: 20,
    marginLeft: spacing.md,
    minWidth: 0,
    textAlign: 'right',
  },
  inlineInput: {
    borderColor: colors.border.default,
    borderRadius: radius.md,
    borderWidth: 1,
    color: colors.text.primary,
    fontSize: fontSize.subheading,
    marginTop: spacing.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: 9,
  },
  rowLeft: { alignItems: 'center', flex: 1, flexDirection: 'row', gap: 10, minWidth: 0 },
  rowRight: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  patientAvatar: {
    alignItems: 'center',
    backgroundColor: '#EEE7DE',
    borderRadius: 17,
    height: 34,
    justifyContent: 'center',
    overflow: 'hidden',
    width: 34,
  },
  patientAvatarImage: { height: '100%', width: '100%' },
  patientAvatarText: { color: colors.accent, fontFamily: fontFamily.display, fontSize: fontSize.bodyLarge, fontWeight: '600' },
  pickerBlock: {
    backgroundColor: colors.surface.card,
    paddingHorizontal: spacing.xl,
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.subtle,
  },
});
