import { Feather } from '@expo/vector-icons';
import * as Linking from 'expo-linking';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { useCaregiver } from '../../src/architecture/CaregiverContext';
import { ConfigurationBanner, PrimaryButton, ScreenLayout, StatusPill } from '../../src/components/AppUI';
import { MirrorIllustration } from '../../src/components/Illustrations';
import { LovedAvatar, LovedBotanical, LovedBrandLockup, LovedIconCircle } from '../../src/components/LovedOneVisuals';
import { MotionFadeIn, MotionPressable } from '../../src/components/Motion';
import { loadCaregiverHome, type CaregiverHome, type CaregiverHomePatient } from '../../src/lib/v1Caregiver';
import { usePatientStatusesV1 } from '../../src/lib/v1Client';
import { formatLastInteraction, getConversationsTodayText, getObjectiveInteractionState, getTechnicalNote } from '../../src/lib/v1Status';
import { useTabBarClearance } from '../../src/lib/useTabBarClearance';
import { cardShadow, colors, fontFamily, fontSize, radius, spacing, typography } from '../../src/theme';

function CallButton({ person }: { person: CaregiverHomePatient }) {
  if (!person.profile.phoneNumber) return null;
  return <View style={styles.callWrap}><PrimaryButton icon="phone" label="Call" onPress={() => { void Linking.openURL(`tel:${person.profile.phoneNumber}`).catch(() => undefined); }} /></View>;
}

export default function HomeScreen() {
  const router = useRouter();
  const clearance = useTabBarClearance();
  const { setup, notificationsEnabled, loadSetupProgress, setNotificationsEnabled } = useCaregiver();
  const [home, setHome] = useState<CaregiverHome | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const patientIds = home?.patients.map((person) => person.patientId) ?? [];
  const statusSlots = usePatientStatusesV1(patientIds);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const nextHome = await loadCaregiverHome();
      setHome(nextHome);
      setNotificationsEnabled(nextHome.caregiver.notificationPreferences.pushNotificationsEnabled);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'We could not load your household yet.');
    } finally {
      setLoading(false);
    }
  }, [setNotificationsEnabled]);
  useFocusEffect(useCallback(() => { void refresh(); void loadSetupProgress(); }, [loadSetupProgress, refresh]));

  const incomplete = Object.values(setup).some((state) => state === 'not-started' || state === 'in-progress');
  const caregiverName = home?.caregiver.name || 'Chloe';
  const today = new Intl.DateTimeFormat('en-SG', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' }).format(new Date());

  return <ScreenLayout bottomInset={clearance} contentContainerStyle={styles.content}>
    <View style={styles.hero}><LovedBrandLockup compact /><MotionPressable accessibilityLabel="Open setup" accessibilityRole="button" onPress={() => router.push('/setup')} style={styles.settings}><Feather color={colors.accent} name="sliders" size={20} /></MotionPressable><LovedBotanical style={styles.botanical} /></View>
    <View style={styles.greeting}><Text accessibilityRole="header" style={styles.greetingTitle}>Good morning, {caregiverName}</Text><Text style={styles.greetingDate}>{today}</Text></View>
    {incomplete ? <ConfigurationBanner title="Complete your Reflexion setup" detail={`${Object.values(setup).filter((state) => state === 'complete').length} of 7 sections complete`} action="Continue setup" onPress={() => router.push('/setup')} /> : null}
    {!notificationsEnabled ? <ConfigurationBanner title="Turn on notifications" detail="Allow Reflexion to tell you when an update may need your attention." action="Enable notifications" onPress={() => router.push('/settings/notifications')} /> : null}
    {loading ? <View style={styles.center}><ActivityIndicator color={colors.accent} /><Text style={styles.emptyText}>Loading your connected household…</Text></View> : null}
    {error ? <View style={styles.empty}><Text style={styles.emptyTitle}>Updates are unavailable</Text><Text accessibilityRole="alert" style={styles.emptyText}>{error}</Text><PrimaryButton label="Try again" onPress={() => void refresh()} /></View> : null}
    {!loading && !error && !home?.patients.length ? <View style={styles.empty}><MirrorIllustration size={112} /><Text style={styles.emptyTitle}>Add a loved one to begin</Text><Text style={styles.emptyText}>Once you add someone and pair their device, factual interaction updates will appear here.</Text><PrimaryButton label="Add a loved one" onPress={() => router.push('/setup/household')} /></View> : null}
    <View style={styles.cards}>{home?.patients.map((person, index) => {
      const status = statusSlots[index];
      const hasDevice = Boolean(person.deviceId);
      const interaction = status?.data;
      const interactionState = person.deviceTechnicalState === 'possible_issue' ? 'device-may-be-offline' : getObjectiveInteractionState(interaction, hasDevice);
      const conversations = getConversationsTodayText(interaction);
      const lastInteraction = interaction ? formatLastInteraction(interaction.lastInteractionAt) : 'No check-in yet.';
      const technicalNote = person.deviceTechnicalState === 'possible_issue' ? getTechnicalNote('possible_issue') : interaction ? getTechnicalNote(interaction.technicalState) : null;
      return <MotionFadeIn key={person.patientId} delay={Math.min(index, 4) * 18} playKey={person.patientId}><View style={styles.card}>
        <MotionPressable accessibilityLabel={`${person.displayName}. Open details`} accessibilityRole="button" feedback="card" onPress={() => router.push(`/loved-one/${person.patientId}`)} style={styles.top}>
          <LovedAvatar name={person.displayName} photoUrl={person.profile.photoUrl} size={92} />
          <Text accessibilityRole="header" style={styles.name}>{person.displayName}</Text>
          <Feather color={colors.textDecorative} name="chevron-right" size={24} />
        </MotionPressable>
        <View style={styles.statusRow}><StatusPill state={interactionState} /></View>
        <View style={styles.factBlock}><View style={styles.factRow}><LovedIconCircle icon="calendar" size={34} /><Text style={styles.fact}>Last interaction: {lastInteraction}</Text></View><View style={styles.factRow}><LovedIconCircle icon="clock" size={34} /><Text style={styles.fact}>{conversations || 'No conversations yet.'}</Text></View></View>
        {technicalNote ? <Text style={styles.technicalNote}>{technicalNote}</Text> : null}
        <View style={styles.actionRow}><View style={styles.primaryWrap}><PrimaryButton icon={hasDevice ? 'message-circle' : 'monitor'} label={hasDevice ? 'Leave a message' : 'Pair device'} onPress={() => router.push(hasDevice ? `/chat/${person.patientId}/compose` : `/device/${person.patientId}/pairing`)} /></View><CallButton person={person} /></View>
        <View style={styles.deviceRow}><Feather color={person.deviceTechnicalState === 'ok' ? colors.status.green : colors.status.grey} name="monitor" size={18} /><Text style={styles.deviceText}>Device status · {hasDevice ? person.deviceTechnicalState === 'ok' ? `Online${person.lastHeartbeatAt ? ` · ${formatLastInteraction(person.lastHeartbeatAt)}` : ''}` : person.deviceTechnicalState === 'possible_issue' ? 'May be offline' : 'Unknown' : 'Not paired'}</Text></View>
      </View></MotionFadeIn>;
    })}</View>
  </ScreenLayout>;
}

const styles = StyleSheet.create({
  content: { gap: spacing.xl, minWidth: 0, paddingTop: spacing.lg },
  hero: { minHeight: 84, minWidth: 0, position: 'relative' },
  botanical: { height: 220, right: -spacing.xl, top: -spacing.xxl, width: 154 },
  settings: { alignItems: 'center', justifyContent: 'center', minHeight: 44, width: 44 },
  greeting: { gap: spacing.xs, marginTop: -spacing.sm, minWidth: 0 },
  greetingTitle: { ...typography.display, color: colors.text.primary, fontFamily: fontFamily.ui, fontSize: fontSize.title, fontWeight: '500', lineHeight: 37 },
  greetingDate: { ...typography.bodyLarge, color: colors.text.secondary },
  cards: { gap: spacing.lg, minWidth: 0 },
  card: { ...cardShadow, backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: 22, borderWidth: 1, minWidth: 0, padding: spacing.lg },
  top: { alignItems: 'center', flexDirection: 'row', gap: spacing.md, minWidth: 0 },
  name: { color: colors.text.primary, flex: 1, flexShrink: 1, fontFamily: fontFamily.display, fontSize: 27, fontStyle: 'italic', fontWeight: '400', lineHeight: 34, minWidth: 0 },
  statusRow: { alignItems: 'flex-start', marginTop: spacing.md, minWidth: 0 },
  factBlock: { gap: spacing.sm, marginTop: spacing.md, minWidth: 0 },
  factRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.md, minWidth: 0 },
  fact: { color: colors.text.secondary, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 23, minWidth: 0 },
  technicalNote: { color: colors.text.secondary, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.caption, lineHeight: 19, marginTop: spacing.md, minWidth: 0 },
  deviceRow: { alignItems: 'center', backgroundColor: '#FBFAF7', borderColor: colors.border.subtle, borderRadius: radius.lg, borderWidth: 1, flexDirection: 'row', gap: spacing.sm, marginTop: spacing.lg, minWidth: 0, paddingHorizontal: spacing.md, paddingVertical: spacing.md },
  deviceText: { color: colors.text.secondary, flex: 1, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 21, minWidth: 0 },
  actionRow: { alignItems: 'stretch', flexDirection: 'row', flexWrap: 'wrap', gap: spacing.md, marginTop: spacing.lg, minWidth: 0 },
  primaryWrap: { flexBasis: 190, flexGrow: 1, minWidth: 0 },
  callWrap: { flexBasis: 94, flexGrow: 0, flexShrink: 1, minWidth: 88 },
  center: { alignItems: 'center', gap: spacing.md, paddingVertical: spacing.xxl },
  empty: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, gap: spacing.md, minWidth: 0, padding: spacing.xl },
  emptyTitle: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.heading, fontWeight: '500', textAlign: 'center' },
  emptyText: { color: colors.text.secondary, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 22, textAlign: 'center' },
});
