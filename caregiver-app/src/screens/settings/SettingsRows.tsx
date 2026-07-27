import { Feather } from '@expo/vector-icons';
import React from 'react';
import { Image, StyleSheet, Switch, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { colors, fontFamily, fontSize, radius, scaleSize, spacing } from '../../theme';
import { getInitials } from './helpers';
import { PILL_HIT_SLOP, pillStyles } from './optionPills';

export function SectionHeader({ title }: { title: string }) {
  return <Text accessibilityRole="header" style={styles.sectionHeader}>{title}</Text>;
}

/**
 * Wraps a run of settings rows into one rounded, inset card.
 *
 * The rows were full-bleed white bands with square corners — the only place in the app built that way. Every
 * other surface is a rounded card (the home dashboard, the alert cards, the chat bubbles, the sign-in panel),
 * so settings read as a different app: two visual languages, one of them accidental.
 *
 * The group owns the shape and the rows keep only their hairline, which is why the card clips: `overflow`
 * hidden makes the first and last row take the container's corners without either of them needing to know it
 * is first or last. The final hairline is dropped the same way — by the container, not by asking each row.
 *
 * The radius is the same token the home cards use, so the two match by construction rather than by
 * coincidence.
 */
export function SettingsGroup({ children }: { children: React.ReactNode }) {
  return <View style={styles.group}>{children}</View>;
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
      <View style={styles.rowMain}>
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
        <View style={styles.rowText}>
          <Text style={styles.rowLabel}>{label}</Text>
          {/* Full width to itself, so a long summary neither truncates nor pushes the label around. The row's
              accessibilityLabel already reads the two together as one phrase. */}
          {value ? <Text style={styles.rowValue}>{value}</Text> : null}
        </View>
      </View>
      <View style={styles.rowRight}>
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
  group: {
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: radius.lg,
    borderWidth: 1,
    marginHorizontal: spacing.xl,
    // What makes the first and last row take the container's corners without either knowing its position,
    // and what hides the last row's hairline against the border.
    overflow: 'hidden',
  },
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
    paddingHorizontal: spacing.lg,
    paddingVertical: scaleSize(15),
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
  rowLabel: { color: colors.text.primary, fontSize: fontSize.subheading, fontWeight: '500' },
  // Left-aligned under the label, not right-aligned across from it. The right alignment was left over from
  // when the two sat side by side; under the label it just looked like a second column that had collapsed.
  rowValue: {
    color: colors.text.tertiary,
    fontSize: fontSize.body,
    minWidth: 0,
  },
  inlineInput: {
    borderColor: colors.border.default,
    borderRadius: radius.md,
    borderWidth: 1,
    color: colors.text.primary,
    fontSize: fontSize.subheading,
    marginTop: spacing.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: scaleSize(9),
  },
  rowMain: { alignItems: 'center', flex: 1, flexDirection: 'row', gap: scaleSize(10), minWidth: 0 },
  // The label and its summary as one block, so the chevron sits beside the pair rather than between them.
  rowText: { flex: 1, gap: scaleSize(2), minWidth: 0 },
  rowRight: { alignItems: 'center', flexDirection: 'row' },
  patientAvatar: {
    alignItems: 'center',
    backgroundColor: '#EEE7DE',
    borderRadius: scaleSize(17),
    height: scaleSize(34),
    justifyContent: 'center',
    overflow: 'hidden',
    width: scaleSize(34),
  },
  patientAvatarImage: { height: '100%', width: '100%' },
  patientAvatarText: { color: colors.accent, fontFamily: fontFamily.display, fontSize: fontSize.bodyLarge, fontWeight: '600' },
  pickerBlock: {
    backgroundColor: colors.surface.card,
    paddingHorizontal: spacing.xl,
    paddingVertical: scaleSize(15),
    borderBottomWidth: 1,
    borderBottomColor: colors.border.subtle,
  },
});
