import { Feather } from '@expo/vector-icons';
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { PASSWORD_REQUIREMENTS, passwordRequirementState } from '../lib/authValidation';
import { colors, fontSize, spacing } from '../theme';

export function PasswordRequirements({ password, repeatPassword }: { password: string; repeatPassword?: string }) {
  const state = passwordRequirementState(password);
  const hasRepeat = repeatPassword !== undefined;
  const matches = hasRepeat && password.length > 0 && password === repeatPassword;
  return (
    <View accessibilityLabel="Password requirements" style={styles.wrap}>
      {PASSWORD_REQUIREMENTS.map((requirement) => {
        const satisfied = state[requirement.key];
        return (
          <View key={requirement.key} style={styles.row}>
            <Feather color={satisfied ? colors.accent : colors.text.secondary} name={satisfied ? 'check-circle' : 'circle'} size={16} />
            <Text style={[styles.text, satisfied && styles.satisfied]}>{requirement.label}</Text>
          </View>
        );
      })}
      {hasRepeat ? (
        <View style={styles.row}>
          <Feather color={matches ? colors.accent : colors.text.secondary} name={matches ? 'check-circle' : 'circle'} size={16} />
          <Text style={[styles.text, matches && styles.satisfied]}>{matches ? 'Passwords match' : 'Passwords must match'}</Text>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { gap: spacing.xs, minWidth: 0 },
  row: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, minWidth: 0 },
  text: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.caption, lineHeight: 18, minWidth: 0 },
  satisfied: { color: colors.accent, fontWeight: '600' },
});
