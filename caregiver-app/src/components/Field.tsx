import React from 'react';
import {
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  type TextInputProps,
  TouchableOpacity,
  View,
} from 'react-native';

import { colors, fontSize, MIN_TOUCH_TARGET, radius, spacing } from '../theme';

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
        <TouchableOpacity
          accessibilityLabel={`Country code, ${countryCode}`}
          accessibilityRole="button"
          disabled={disabled}
          onPress={() => setOpen(true)}
          style={styles.countryButton}
        >
          <Text style={styles.countryText}>{countryCode}</Text>
          <Text style={styles.countryChevron}>⌄</Text>
        </TouchableOpacity>
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
                <TouchableOpacity
                  accessibilityRole="button"
                  accessibilityState={{ selected: code === countryCode }}
                  key={code}
                  onPress={() => { onCountryCodeChange(code); setOpen(false); }}
                  style={[styles.codeOption, code === countryCode && styles.codeOptionSelected]}
                >
                  <Text style={styles.codeOptionText}>{code}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
            <TouchableOpacity accessibilityRole="button" onPress={() => setOpen(false)} style={styles.modalCancel}>
              <Text style={styles.modalCancelText}>Cancel</Text>
            </TouchableOpacity>
          </Pressable>
        </Pressable>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { gap: spacing.sm, width: '100%' },
  label: { color: colors.text.primary, fontSize: fontSize.body, fontWeight: '700', lineHeight: 20 },
  input: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.md, borderWidth: 1, color: colors.text.primary, fontSize: fontSize.bodyLarge, minHeight: 54, paddingHorizontal: spacing.lg, paddingVertical: spacing.md },
  multiline: { minHeight: 120, paddingTop: spacing.md, textAlignVertical: 'top' },
  inputError: { borderColor: colors.error.border, borderWidth: 1.5 },
  error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 21 },
  helper: { color: colors.text.secondary, fontSize: fontSize.caption, lineHeight: 18 },
  phoneRow: { alignItems: 'stretch', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.md, borderWidth: 1, flexDirection: 'row', minHeight: 54, overflow: 'hidden' },
  countryButton: { alignItems: 'center', borderRightColor: colors.border.default, borderRightWidth: 1, flexDirection: 'row', gap: spacing.xs, justifyContent: 'center', minHeight: MIN_TOUCH_TARGET, paddingHorizontal: spacing.md },
  countryText: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '600' },
  countryChevron: { color: colors.text.secondary, fontSize: fontSize.bodyLarge },
  phoneInput: { color: colors.text.primary, flex: 1, fontSize: fontSize.bodyLarge, minWidth: 0, paddingHorizontal: spacing.lg, paddingVertical: spacing.md },
  modalBackdrop: { alignItems: 'center', backgroundColor: 'rgba(22,50,74,0.24)', flex: 1, justifyContent: 'center', padding: spacing.screen },
  modalCard: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, maxHeight: '80%', padding: spacing.xl, width: '100%' },
  modalTitle: { color: colors.text.primary, fontSize: fontSize.heading, fontWeight: '700', lineHeight: 26 },
  codeList: { gap: spacing.sm, paddingVertical: spacing.lg },
  codeOption: { borderColor: colors.border.default, borderRadius: radius.md, borderWidth: 1, justifyContent: 'center', minHeight: MIN_TOUCH_TARGET, paddingHorizontal: spacing.lg },
  codeOptionSelected: { backgroundColor: '#E7F3F0', borderColor: colors.accent },
  codeOptionText: { color: colors.text.primary, fontSize: fontSize.bodyLarge },
  modalCancel: { alignItems: 'center', justifyContent: 'center', minHeight: MIN_TOUCH_TARGET },
  modalCancelText: { color: colors.accent, fontSize: fontSize.bodyLarge, fontWeight: '700' },
});
