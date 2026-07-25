import React from 'react';
import { Alert, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { colors, fontSize, MIN_TOUCH_TARGET, radius, spacing } from '../../theme';
import { fieldStyles, Label } from './fields';
import { formatPairingInput } from './helpers';
import type { PatientForm } from './types';

export function MirrorStep({
  patients,
  updatePatient,
}: {
  patients: PatientForm[];
  updatePatient: (index: number, updates: Partial<PatientForm>) => void;
}) {
  return (
    <View>
      <View style={styles.infoBox}>
        <Text accessibilityRole="header" style={styles.infoTitle}>Mirror pairing</Text>
        <Text style={styles.infoText}>
          On the mirror, open setup and enter the 6-digit pairing code shown there. You can leave this blank and pair the mirror later from settings.
        </Text>
      </View>

      {patients.map((patient, index) => {
        // With two or more mirrors on one screen, every field and button here would otherwise announce
        // identically ("Mirror name", "Mirror name"). The person's name disambiguates each block.
        const heading = patient.name.trim() || `Person ${index + 1}`;
        return (
          <View key={index} style={styles.mirrorBlock}>
            <Text accessibilityRole="header" style={styles.mirrorHeading}>{heading}</Text>
            <Label>Mirror name</Label>
            <TextInput
              accessibilityLabel={`Mirror name for ${heading}`}
              onChangeText={(mirrorName) => updatePatient(index, { mirrorName })}
              placeholder={`Mirror ${index + 1} - Toa Payoh home`}
              placeholderTextColor={colors.placeholder}
              style={fieldStyles.input}
              value={patient.mirrorName}
            />
            <Label>Mirror pairing code</Label>
            <TextInput
              accessibilityLabel={`Six-digit pairing code shown on the mirror for ${heading}, optional`}
              keyboardType="number-pad"
              maxLength={7}
              onChangeText={(mirrorPairingCode) => updatePatient(index, { mirrorPairingCode })}
              placeholder="482 913"
              placeholderTextColor={colors.placeholder}
              style={fieldStyles.input}
              value={formatPairingInput(patient.mirrorPairingCode)}
            />
            <Label>Mirror timezone</Label>
            <TextInput
              accessibilityLabel={`Mirror timezone for ${heading}`}
              autoCapitalize="none"
              onChangeText={(timezone) => updatePatient(index, { timezone })}
              placeholder="Asia/Singapore"
              placeholderTextColor={colors.placeholder}
              style={fieldStyles.input}
              value={patient.timezone}
            />
            <TouchableOpacity
              accessibilityLabel={`How pairing works for ${heading}'s mirror`}
              accessibilityRole="button"
              onPress={() => Alert.alert('Pairing instructions', 'Enter the code displayed on the mirror, or scan the mirror QR in the caregiver app once scanner support is enabled.')}
              style={styles.testBtn}
            >
              <Text style={styles.testBtnText}>How pairing works</Text>
            </TouchableOpacity>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  infoBox: {
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: radius.sm,
    borderWidth: 1,
    padding: 14,
  },
  infoTitle: {
    color: colors.text.primary,
    fontSize: fontSize.bodyLarge,
    fontWeight: '800',
    marginBottom: spacing.xs,
  },
  infoText: {
    color: colors.text.secondary,
    fontSize: fontSize.body,
    lineHeight: 19,
  },
  mirrorBlock: {
    borderBottomColor: colors.border.default,
    borderBottomWidth: 1,
    paddingBottom: 18,
    paddingTop: 18,
  },
  mirrorHeading: {
    color: colors.text.primary,
    fontSize: 16,
    fontWeight: '800',
  },
  testBtn: {
    alignItems: 'center',
    backgroundColor: colors.text.primary,
    borderRadius: radius.sm,
    marginTop: spacing.md,
    minHeight: MIN_TOUCH_TARGET,
    justifyContent: 'center',
  },
  testBtnText: {
    color: colors.text.onAccent,
    fontSize: fontSize.bodyLarge,
    fontWeight: '800',
  },
});
