import { Feather } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { useCaregiver } from '../../src/architecture/CaregiverContext';
import { type SetupCategory } from '../../src/architecture/models';
import { AppHeader, InfoCard, PrimaryButton, ScreenLayout, TertiaryButton } from '../../src/components/AppUI';
import { colors, contentColumn, fontFamily, fontSize, spacing } from '../../src/theme';

type Detail = { icon: keyof typeof Feather.glyphMap; title: string; description: string };
const DETAILS: Record<SetupCategory, { title: string; subtitle: string; action: string; items: Detail[] }> = {
  household: { title: 'Your household', subtitle: 'Add each loved one with the details that help you stay connected.', action: 'Open household', items: [{ icon: 'user', title: 'Loved-one profile', description: 'Preferred name, age or date of birth, relationship and contact details.' }, { icon: 'users', title: 'Add another loved one', description: 'Keep each person’s care information separate.' }, { icon: 'phone', title: 'Emergency contact', description: 'Add a contact and living arrangement for each person.' }] },
  'pair-device': { title: 'Pair a device', subtitle: 'Choose the loved one and a Reflexion device to connect.', action: 'Pair device', items: [{ icon: 'monitor', title: 'Mirror', description: 'Pair with a QR code or six-digit code.' }, { icon: 'heart', title: 'Bear', description: 'Bear pairing is outside this pilot and is shown as unavailable.' }, { icon: 'smartphone', title: 'App', description: 'The Reflexion mobile app is not a caregiver device in this pilot.' }, { icon: 'tablet', title: 'Other supported device', description: 'Only the Reflexion Mirror pairing contract is enabled today.' }] },
  'language-accessibility': { title: 'Language & accessibility', subtitle: 'Set up a familiar and comfortable Reflexion experience.', action: 'Open preferences', items: [{ icon: 'globe', title: 'Language and time zone', description: 'Country, time zone, preferred and secondary spoken language.' }, { icon: 'type', title: 'Text and captions', description: 'Text size, captions, high contrast and simplified interface.' }, { icon: 'volume-2', title: 'Voice and hearing support', description: 'Assistant voice, pace, volume, hearing support and voice preview.' }] },
  routines: { title: 'Routines', subtitle: 'Create gentle prompts that fit Mum’s day.', action: 'Open routines', items: [{ icon: 'box', title: 'Medication', description: 'Use familiar wording without treating a report as verification.' }, { icon: 'coffee', title: 'Meals, hydration and exercise', description: 'Support everyday routines and movement.' }, { icon: 'calendar', title: 'Appointments and family events', description: 'Keep plans and important moments in view.' }, { icon: 'bell', title: 'Caregiver notification choice', description: 'Do not notify me · Notify me after one missed or unclear response · Include it in my daily summary.' }, { icon: 'plus-circle', title: 'Custom / Other', description: 'Create a routine in words that feel familiar.' }] },
  notifications: { title: 'Notification preferences', subtitle: 'Choose the updates you would like Reflexion to send.', action: 'Open preferences', items: [{ icon: 'message-circle', title: 'Conversation session summaries', description: 'Immediately after each session, daily, weekly or off.' }, { icon: 'clock', title: 'Interaction updates', description: 'No interaction yet, repeated missed interactions or shorter than usual.' }, { icon: 'wifi-off', title: 'Device and routine updates', description: 'Device may be offline, or a reminder is unclear.' }, { icon: 'bell', title: 'Family messages & weekly summaries', description: 'A message is delivered to a paired device, plus the weekly summary.' }] },
  'consent-control': { title: 'Older-Adult Consent & Control', subtitle: 'Explain Reflexion in plain language and record Mum’s choices.', action: 'Review consent', items: [{ icon: 'shield', title: 'Consent status', description: 'Pending, Accepted, Declined or Withdrawn.' }, { icon: 'pause-circle', title: 'Pause or stop', description: 'Pause conversations, stop a current conversation or pause sharing.' }, { icon: 'help-circle', title: 'Help with consent', description: 'Request help with the explanation. Research remains optional and separate.' }] },
  'care-circle': { title: 'Care Circle', subtitle: 'Invite the trusted people who help care for Mum.', action: 'Open Care Circle', items: [{ icon: 'mail', title: 'Invite a caregiver', description: 'Invite by email or phone, then choose their relationship.' }, { icon: 'users', title: 'Role and access', description: 'Full access, Standard access, View only or Custom access.' }, { icon: 'sliders', title: 'Permissions', description: 'Choose loved ones, notifications, routines, devices and caregiver-management access.' }] },
  'research-participation': { title: 'Research participation', subtitle: 'Research is optional and separate from everyday Reflexion care.', action: 'Review research choice', items: [{ icon: 'book-open', title: 'What participation means', description: 'De-identified product data may support research when you choose it.' }, { icon: 'x-circle', title: 'You can decline or withdraw', description: 'Your care experience continues and your choice can change later.' }, { icon: 'shield', title: 'No identifying information', description: 'Names and contact details are not shared with researchers.' }] },
};

function isCategory(value: string): value is SetupCategory { return value in DETAILS; }

export default function SetupCategoryScreen() {
  const router = useRouter();
  const { category } = useLocalSearchParams<{ category?: string }>();
  const candidate = Array.isArray(category) ? category[0] : category;
  const key: SetupCategory = isCategory(candidate || '') ? candidate : 'household';
  const detail = DETAILS[key];
  const { setSetupStatus } = useCaregiver();
  const continueSetup = () => {
    setSetupStatus(key, 'in-progress');
    const route: Record<SetupCategory, string> = { household: '/settings/household', 'pair-device': '/settings/devices', 'language-accessibility': '/settings/language', notifications: '/settings/notifications', routines: '/settings/routines', 'consent-control': '/settings/consent', 'care-circle': '/settings/care-circle', 'research-participation': '/settings/research' };
    router.push(route[key]);
  };
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Setup" onBack={() => router.back()} /><Text style={styles.step}>Setup category</Text><Text accessibilityRole="header" style={styles.title}>{detail.title}</Text><Text style={styles.subtitle}>{detail.subtitle}</Text><View style={styles.cards}>{detail.items.map((item) => <InfoCard key={item.title} {...item} />)}</View><View style={styles.actions}><PrimaryButton label={detail.action} onPress={continueSetup} /><TertiaryButton label="Set up later" onPress={() => { setSetupStatus(key, 'skipped'); router.replace('/setup'); }} /></View></ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, step: { color: colors.accent, fontSize: fontSize.caption, fontWeight: '700', marginTop: spacing.xl }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 34, marginTop: spacing.xs }, subtitle: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 25, marginTop: spacing.md }, cards: { gap: spacing.md, marginTop: spacing.xxl }, actions: { gap: spacing.sm, marginTop: spacing.xxl } });
