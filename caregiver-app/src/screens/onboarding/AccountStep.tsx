import React from 'react';
import { StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { colors, fontSize, MIN_TOUCH_TARGET } from '../../theme';
import { fieldStyles, Label, OptionGrid } from './fields';
import type { AccountForm, Relationship } from './types';

const RELATIONSHIP_OPTIONS: { value: Relationship; label: string }[] = [
  { value: 'parent', label: 'Parent' },
  { value: 'sibling', label: 'Sibling' },
  { value: 'spouse', label: 'Spouse' },
  { value: 'inlaw', label: 'In-law' },
  { value: 'grandpa', label: 'Grandpa' },
  { value: 'grandma', label: 'Grandma' },
  { value: 'other', label: 'Other' },
];

export function AccountStep({
  account,
  onSignIn,
  setAccount,
}: {
  account: AccountForm;
  onSignIn: () => void;
  setAccount: React.Dispatch<React.SetStateAction<AccountForm>>;
}) {
  return (
    <View>
      <Label>Name</Label>
      <TextInput
        accessibilityLabel="Your name"
        onChangeText={(name) => setAccount((current) => ({ ...current, name }))}
        placeholder="e.g. Sarah Lim"
        placeholderTextColor={colors.placeholder}
        style={fieldStyles.input}
        value={account.name}
      />

      <Label>Email</Label>
      <TextInput
        accessibilityLabel="Email"
        autoCapitalize="none"
        autoComplete="email"
        keyboardType="email-address"
        onChangeText={(email) => setAccount((current) => ({ ...current, email }))}
        placeholder="you@email.com"
        placeholderTextColor={colors.placeholder}
        style={fieldStyles.input}
        value={account.email}
      />

      <Label>Password</Label>
      <TextInput
        accessibilityLabel="Password, at least 8 characters"
        autoCapitalize="none"
        onChangeText={(password) => setAccount((current) => ({ ...current, password }))}
        placeholder="Create a password"
        placeholderTextColor={colors.placeholder}
        secureTextEntry
        style={fieldStyles.input}
        value={account.password}
      />

      <Label>Phone number</Label>
      <TextInput
        accessibilityLabel="Your phone number"
        keyboardType="phone-pad"
        onChangeText={(phoneNumber) => setAccount((current) => ({ ...current, phoneNumber }))}
        placeholder="+65 9123 4567"
        placeholderTextColor={colors.placeholder}
        style={fieldStyles.input}
        value={account.phoneNumber}
      />

      <Label>I am caring for...</Label>
      <OptionGrid
        groupLabel="I am caring for"
        options={RELATIONSHIP_OPTIONS}
        selected={account.relationshipToElderly}
        onSelect={(relationshipToElderly) =>
          setAccount((current) => ({ ...current, relationshipToElderly }))
        }
      />

      <TouchableOpacity accessibilityRole="button" onPress={onSignIn} style={styles.signInLink}>
        <Text style={styles.signInLinkText}>Have an account? Sign in!</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  signInLink: {
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 22,
    minHeight: MIN_TOUCH_TARGET,
  },
  signInLinkText: {
    color: colors.accent,
    fontSize: fontSize.bodyLarge,
    fontWeight: '800',
  },
});
