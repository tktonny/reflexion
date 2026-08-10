import { Feather } from '@expo/vector-icons';
import React from 'react';
import {
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  type TextInputProps,
  View,
} from 'react-native';

import { colors, fontSize, MIN_TOUCH_TARGET, radius, spacing } from '../theme';
import { MotionPressable } from './Motion';

export function Field({ label, secure, error, helperText, ...props }: TextInputProps & { label: string; secure?: boolean; error?: string; helperText?: string }) {
  return (
    <View style={styles.wrap}>
      <Text style={styles.label}>{label}</Text>
      <TextInput
        accessibilityLabel={label}
        autoCapitalize="none"
        placeholderTextColor={colors.placeholder}
        secureTextEntry={secure}
        style={[styles.input, props.multiline && styles.multiline, error && styles.inputError]}
        {...props}
      />
      {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : helperText ? <Text style={styles.helper}>{helperText}</Text> : null}
    </View>
  );
}

export type PhoneFieldProps = {
  label: string;
  countryCode: string;
  phoneNumber: string;
  onCountryCodeChange: (value: string) => void;
  onPhoneNumberChange: (value: string) => void;
  error?: string;
  helperText?: string;
  disabled?: boolean;
};

const COUNTRY_CODES = ['+65', '+1', '+44', '+61', '+81', '+86'] as const;

/** The shared phone contract: country code and national number are separate values everywhere. */
export function PhoneField({ label, countryCode, phoneNumber, onCountryCodeChange, onPhoneNumberChange, error, helperText, disabled = false }: PhoneFieldProps) {
  const [open, setOpen] = React.useState(false);
  return (
    <View style={styles.wrap}>
      <Text style={styles.label}>{label}</Text>
      <View style={[styles.phoneRow, error && styles.inputError]}>
        <MotionPressable
          accessibilityLabel={`Country code, ${countryCode}`}
          accessibilityRole="button"
          disabled={disabled}
          onPress={() => setOpen(true)}
          style={styles.countryButton}
        >
          <Text style={styles.countryText}>{countryCode}</Text>
          <Text style={styles.countryChevron}>⌄</Text>
        </MotionPressable>
        <TextInput
          accessibilityLabel={`${label} number`}
          autoCapitalize="none"
          autoComplete="tel"
          editable={!disabled}
          keyboardType="phone-pad"
          onChangeText={(value) => onPhoneNumberChange(value.replace(/[^0-9\s().-]/g, ''))}
          placeholder="9000 1234"
          placeholderTextColor={colors.placeholder}
          style={styles.phoneInput}
          value={phoneNumber}
        />
      </View>
      {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : helperText ? <Text style={styles.helper}>{helperText}</Text> : null}
      <Modal accessibilityViewIsModal animationType="fade" onRequestClose={() => setOpen(false)} transparent visible={open}>
        <Pressable accessibilityLabel="Close country code selector" onPress={() => setOpen(false)} style={styles.modalBackdrop}>
          <Pressable onPress={(event) => event.stopPropagation()} style={styles.modalCard}>
            <Text accessibilityRole="header" style={styles.modalTitle}>Choose country code</Text>
            <ScrollView contentContainerStyle={styles.codeList}>
              {COUNTRY_CODES.map((code) => (
                <MotionPressable
                  accessibilityRole="button"
                  accessibilityState={{ selected: code === countryCode }}
                  feedback="card"
                  haptic="selection"
                  key={code}
                  onPress={() => { onCountryCodeChange(code); setOpen(false); }}
                  style={[styles.codeOption, code === countryCode && styles.codeOptionSelected]}
                >
                  <Text style={styles.codeOptionText}>{code}</Text>
                </MotionPressable>
              ))}
            </ScrollView>
            <MotionPressable accessibilityRole="button" onPress={() => setOpen(false)} style={styles.modalCancel}>
              <Text style={styles.modalCancelText}>Cancel</Text>
            </MotionPressable>
          </Pressable>
        </Pressable>
      </Modal>
    </View>
  );
}

export type SelectFieldOption = { value: string; label: string };

/** A normal form field that opens a selection sheet. The arrow describes the field action, not a page navigation. */
export function SelectField({ label, value, options, onChange, placeholder = 'Choose an option', disabled = false, helperText }: {
  label: string;
  value: string;
  options: SelectFieldOption[];
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  helperText?: string;
}) {
  const [open, setOpen] = React.useState(false);
  const selected = options.find((option) => option.value === value);
  return (
    <View style={styles.wrap}>
      <Text style={styles.label}>{label}</Text>
      <MotionPressable accessibilityLabel={label} accessibilityRole="button" accessibilityState={{ disabled }} disabled={disabled} onPress={() => setOpen(true)} style={styles.selectButton}>
        <Text style={[styles.selectText, !selected && styles.selectPlaceholder]}>{selected?.label || placeholder}</Text>
        <Feather color={colors.text.secondary} name="chevron-down" size={21} />
      </MotionPressable>
      {helperText ? <Text style={styles.helper}>{helperText}</Text> : null}
      <Modal accessibilityViewIsModal animationType="slide" onRequestClose={() => setOpen(false)} transparent visible={open}>
        <Pressable accessibilityLabel={`Close ${label} selector`} onPress={() => setOpen(false)} style={styles.modalBackdrop}>
          <Pressable onPress={(event) => event.stopPropagation()} style={styles.modalCard}>
            <Text accessibilityRole="header" style={styles.modalTitle}>Choose {label.toLowerCase()}</Text>
            <ScrollView contentContainerStyle={styles.codeList} keyboardShouldPersistTaps="handled">
              {options.map((option) => {
                const isSelected = option.value === value;
                return <MotionPressable accessibilityLabel={option.label} accessibilityRole="button" accessibilityState={{ selected: isSelected }} feedback="card" haptic="selection" key={option.value} onPress={() => { onChange(option.value); setOpen(false); }} style={[styles.codeOption, isSelected && styles.codeOptionSelected]}><Text style={[styles.codeOptionText, isSelected && styles.codeOptionTextSelected]}>{option.label}</Text>{isSelected ? <Feather color={colors.accent} name="check" size={21} /> : null}</MotionPressable>;
              })}
            </ScrollView>
            <MotionPressable accessibilityRole="button" onPress={() => setOpen(false)} style={styles.modalCancel}><Text style={styles.modalCancelText}>Cancel</Text></MotionPressable>
          </Pressable>
        </Pressable>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { gap: spacing.sm, minWidth: 0, width: '100%' },
  label: { color: colors.text.primary, fontSize: fontSize.body, fontWeight: '700', lineHeight: 20, minWidth: 0 },
  input: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.md, borderWidth: 1, color: colors.text.primary, fontSize: fontSize.bodyLarge, minHeight: 54, minWidth: 0, paddingHorizontal: spacing.lg, paddingVertical: spacing.md },
  selectButton: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.md, borderWidth: 1, flexDirection: 'row', justifyContent: 'space-between', minHeight: 54, minWidth: 0, paddingHorizontal: spacing.lg },
  selectText: { color: colors.text.primary, flex: 1, flexShrink: 1, fontSize: fontSize.bodyLarge, lineHeight: 22, minWidth: 0 },
  selectPlaceholder: { color: colors.placeholder },
  multiline: { minHeight: 120, paddingTop: spacing.md, textAlignVertical: 'top' },
  inputError: { borderColor: colors.error.border, borderWidth: 1.5 },
  error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 21 },
  helper: { color: colors.text.secondary, fontSize: fontSize.caption, lineHeight: 18 },
  phoneRow: { alignItems: 'stretch', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.md, borderWidth: 1, flexDirection: 'row', minHeight: 54, minWidth: 0, overflow: 'hidden' },
  countryButton: { alignItems: 'center', borderRightColor: colors.border.default, borderRightWidth: 1, flexDirection: 'row', gap: spacing.xs, justifyContent: 'center', minHeight: MIN_TOUCH_TARGET, paddingHorizontal: spacing.md },
  countryText: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '600' },
  countryChevron: { color: colors.text.secondary, fontSize: fontSize.bodyLarge },
  phoneInput: { color: colors.text.primary, flex: 1, fontSize: fontSize.bodyLarge, minWidth: 0, paddingHorizontal: spacing.lg, paddingVertical: spacing.md },
  modalBackdrop: { alignItems: 'center', backgroundColor: 'rgba(22,50,74,0.24)', flex: 1, justifyContent: 'center', padding: spacing.screen },
  modalCard: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, maxHeight: '80%', padding: spacing.xl, width: '100%' },
  modalTitle: { color: colors.text.primary, fontSize: fontSize.heading, fontWeight: '700', lineHeight: 26 },
  codeList: { gap: spacing.sm, paddingVertical: spacing.lg },
  codeOption: { alignItems: 'center', borderColor: colors.border.default, borderRadius: radius.md, borderWidth: 1, flexDirection: 'row', justifyContent: 'space-between', minHeight: MIN_TOUCH_TARGET, minWidth: 0, paddingHorizontal: spacing.lg },
  codeOptionSelected: { backgroundColor: '#E7F3F0', borderColor: colors.accent },
  codeOptionText: { color: colors.text.primary, flex: 1, flexShrink: 1, fontSize: fontSize.bodyLarge, lineHeight: 22, minWidth: 0 },
  codeOptionTextSelected: { color: colors.accent, fontWeight: '700' },
  modalCancel: { alignItems: 'center', justifyContent: 'center', minHeight: MIN_TOUCH_TARGET },
  modalCancelText: { color: colors.accent, fontSize: fontSize.bodyLarge, fontWeight: '700' },
});
