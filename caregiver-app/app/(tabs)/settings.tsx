import { useRouter } from 'expo-router';
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { ScreenLayout, SettingsRow } from '../../src/components/AppUI';
import { useTabBarClearance } from '../../src/lib/useTabBarClearance';
import { colors, contentColumn, fontFamily, fontSize, radius, spacing } from '../../src/theme';

const rows = [
  ['user', 'Account', 'Profile, verified email and sign-in', '/settings/account'],
  ['bell', 'App notifications', 'Messages, check-ins and device updates', '/settings/notifications'],
  ['globe', 'App language', 'English', '/settings/language'],
  ['users', 'Loved Ones', 'Profiles and personal details', '/settings/household'],
  ['clock', 'Routines', 'Daily prompts and caregiver notification choices', '/settings/routines'],
  ['monitor', 'Connected Devices', 'Pair and manage Reflexion Mirrors', '/settings/devices'],
  ['shield', 'Consent & control', 'Review consent and pause participation', '/settings/consent'],
  ['users', 'Care Circle', 'Invite caregivers and manage permissions', '/settings/care-circle'],
  ['lock', 'Privacy & data', 'Retention, consent history and deletion', '/settings/privacy'],
  ['book-open', 'Research participation', 'Optional and separate from care', '/settings/research'],
  ['message-square', 'Pilot Feedback', 'Share feedback or report an issue', '/settings/feedback'],
  ['help-circle', 'Help & Support', 'Contact Reflexion support', '/settings/help'],
] as const;

export default function SettingsScreen() { const router = useRouter(); const clearance = useTabBarClearance(); return <ScreenLayout bottomInset={clearance} contentContainerStyle={styles.content}><Text accessibilityRole="header" style={styles.title}>Settings</Text><Text style={styles.subtitle}>Manage your account and the parts of care that are connected today.</Text><View style={styles.group}>{rows.map(([icon, label, value, route]) => <SettingsRow key={label} icon={icon} label={label} value={value} onPress={() => router.push(route)} />)}<SettingsRow icon="credit-card" label="Subscription" value="Not available in this pilot" disabled /></View></ScreenLayout>; }
const styles = StyleSheet.create({ content: { gap: spacing.xl, paddingTop: spacing.xl }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500' }, subtitle: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22, marginTop: -spacing.md }, group: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, overflow: 'hidden' } });
