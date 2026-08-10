import { Feather } from '@expo/vector-icons';
import * as Linking from 'expo-linking';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, Image, StyleSheet, Text, View } from 'react-native';

import { useCaregiver } from '../../src/architecture/CaregiverContext';
import { ConfigurationBanner, PrimaryButton, ScreenLayout, SecondaryButton, StatusPill } from '../../src/components/AppUI';
import { BrandLockup } from '../../src/components/BrandLockup';
import { MirrorIllustration } from '../../src/components/Illustrations';
import { MotionFadeIn, MotionPressable } from '../../src/components/Motion';
import { loadCaregiverHome, type CaregiverHome, type CaregiverHomePatient } from '../../src/lib/v1Caregiver';
import { usePatientStatusesV1 } from '../../src/lib/v1Client';
import { formatLastInteraction, getConversationsTodayText, getObjectiveInteractionState, getTechnicalNote } from '../../src/lib/v1Status';
import { useTabBarClearance } from '../../src/lib/useTabBarClearance';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';

function CallButton({ person }: { person: CaregiverHomePatient }) {
  if (!person.profile.phoneNumber) return null;
  return <View style={styles.callWrap}><SecondaryButton label="Call" onPress={() => { void Linking.openURL(`tel:${person.profile.phoneNumber}`).catch(() => undefined); }} /></View>;
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
  const firstPerson = home?.patients[0];
  const firstHasDevice = Boolean(firstPerson?.deviceId);

  return <ScreenLayout bottomInset={clearance} contentContainerStyle={styles.content}>
    <View style={styles.brand}><BrandLockup compact /><MotionPressable accessibilityLabel="Open setup" accessibilityRole="button" onPress={() => router.push('/setup')} style={styles.settings}><Feather color={colors.accent} name="sliders" size={20} /></MotionPressable></View>
    {incomplete ? <ConfigurationBanner title="Complete your Reflexion setup" detail={`${Object.values(setup).filter((state) => state === 'complete').length} of 7 sections complete`} action="Continue setup" onPress={() => router.push('/setup')} /> : null}
    {!notificationsEnabled ? <ConfigurationBanner title="Turn on notifications" detail="Allow Reflexion to tell you when an update may need your attention." action="Enable notifications" onPress={() => router.push('/settings/notifications')} /> : null}
    {loading ? <View style={styles.center}><ActivityIndicator color={colors.accent} /><Text style={styles.emptyText}>Loading your connected household…</Text></View> : null}
    {error ? <View style={styles.empty}><Text style={styles.emptyTitle}>Updates are unavailable</Text><Text accessibilityRole="alert" style={styles.emptyText}>{error}</Text><PrimaryButton label="Try again" onPress={() => void refresh()} /></View> : null}
    {!loading && !error && !home?.patients.length ? <View style={styles.empty}><MirrorIllustration size={112} /><Text style={styles.emptyTitle}>Add a loved one to begin</Text><Text style={styles.emptyText}>Once you add someone and pair their device, factual interaction updates will appear here.</Text><PrimaryButton label="Add a loved one" onPress={() => router.push('/setup/household')} /></View> : null}
    {!loading && !error && firstPerson ? <MotionFadeIn playKey={firstPerson.patientId}><View style={styles.deviceCard}>
      <View style={styles.deviceHeading}><Feather color={colors.text.primary} name="monitor" size={24} /><Text style={styles.deviceHeadingText}>Device status · {firstHasDevice ? firstPerson.deviceTechnicalState === 'ok' ? 'Online' : firstPerson.deviceTechnicalState === 'possible_issue' ? 'May be offline' : 'Unknown' : 'Not paired'}</Text></View>
      <View style={styles.deviceActions}><View style={styles.primaryWrap}><PrimaryButton icon={firstHasDevice ? 'message-circle' : 'monitor'} label={firstHasDevice ? 'Leave a message' : 'Pair device'} onPress={() => router.push(firstHasDevice ? `/chat/${firstPerson.patientId}/compose` : `/device/${firstPerson.patientId}/pairing`)} /></View><CallButton person={firstPerson} /></View>
    </View></MotionFadeIn> : null}
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
          <View style={styles.avatar}>{person.profile.photoUrl ? <Image accessibilityLabel={`${person.displayName} photo`} source={{ uri: person.profile.photoUrl }} style={styles.avatarImage} /> : <Text style={styles.avatarText}>{person.displayName.slice(0, 1).toUpperCase()}</Text>}</View>
          <Text accessibilityRole="header" style={styles.name}>{person.displayName}</Text>
          <Feather color={colors.textDecorative} name="chevron-right" size={24} />
        </MotionPressable>
        <View style={styles.statusRow}><StatusPill state={interactionState} /></View>
        <View style={styles.divider} />
        <View style={styles.factBlock}><Text style={styles.fact}>{conversations || 'No conversations yet.'}</Text><Text style={styles.fact}>Last interaction: {lastInteraction}</Text></View>
        {technicalNote ? <Text style={styles.technicalNote}>{technicalNote}</Text> : null}
        <View style={styles.deviceRow}><Feather color={person.deviceTechnicalState === 'ok' ? colors.status.green : colors.status.grey} name="monitor" size={18} /><Text style={styles.deviceText}>Device status · {hasDevice ? person.deviceTechnicalState === 'ok' ? `Online${person.lastHeartbeatAt ? ` · ${formatLastInteraction(person.lastHeartbeatAt)}` : ''}` : person.deviceTechnicalState === 'possible_issue' ? 'May be offline' : 'Unknown' : 'Not paired'}</Text></View>
        <View style={styles.actionRow}><View style={styles.primaryWrap}><PrimaryButton icon={hasDevice ? 'message-circle' : 'monitor'} label={hasDevice ? 'Leave a message' : 'Pair device'} onPress={() => router.push(hasDevice ? `/chat/${person.patientId}/compose` : `/device/${person.patientId}/pairing`)} /></View><CallButton person={person} /></View>
      </View></MotionFadeIn>;
    })}</View>
  </ScreenLayout>;
}

const styles = StyleSheet.create({
  content: { gap: spacing.xl, minWidth: 0, paddingTop: spacing.lg },
  brand: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between', minWidth: 0 },
  settings: { alignItems: 'center', justifyContent: 'center', minHeight: 44, width: 44 },
  deviceCard: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.lg, minWidth: 0, padding: spacing.lg },
  deviceHeading: { alignItems: 'center', flexDirection: 'row', gap: spacing.md, minWidth: 0 },
  deviceHeadingText: { color: colors.text.primary, flex: 1, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.bodyLarge, fontWeight: '600', lineHeight: 23, minWidth: 0 },
  deviceActions: { alignItems: 'stretch', flexDirection: 'row', flexWrap: 'wrap', gap: spacing.md, minWidth: 0 },
  cards: { gap: spacing.lg, minWidth: 0 },
  card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: 22, borderWidth: 1, minWidth: 0, padding: spacing.lg, shadowColor: colors.shadow, shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.055, shadowRadius: 20 },
  top: { alignItems: 'center', flexDirection: 'row', gap: spacing.md, minWidth: 0 },
  avatar: { alignItems: 'center', backgroundColor: '#EADFCF', borderRadius: 999, flexShrink: 0, height: 64, justifyContent: 'center', overflow: 'hidden', width: 64 },
  avatarImage: { height: '100%', width: '100%' },
  avatarText: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: 26, fontWeight: '500' },
  name: { color: colors.text.primary, flex: 1, flexShrink: 1, fontFamily: fontFamily.display, fontSize: 26, fontWeight: '500', lineHeight: 32, minWidth: 0 },
  statusRow: { alignItems: 'flex-start', marginTop: spacing.md, minWidth: 0 },
  divider: { backgroundColor: colors.border.subtle, height: 1, marginVertical: spacing.lg },
  factBlock: { gap: spacing.sm, minWidth: 0 },
  fact: { color: colors.text.secondary, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 23, minWidth: 0 },
  technicalNote: { color: colors.text.secondary, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.caption, lineHeight: 19, marginTop: spacing.md, minWidth: 0 },
  deviceRow: { alignItems: 'flex-start', borderTopColor: colors.border.subtle, borderTopWidth: 1, flexDirection: 'row', gap: spacing.sm, marginTop: spacing.lg, minWidth: 0, paddingTop: spacing.lg },
  deviceText: { color: colors.text.secondary, flex: 1, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 21, minWidth: 0 },
  actionRow: { alignItems: 'stretch', flexDirection: 'row', flexWrap: 'wrap', gap: spacing.md, marginTop: spacing.lg, minWidth: 0 },
  primaryWrap: { flexBasis: 190, flexGrow: 1, minWidth: 0 },
  callWrap: { flexBasis: 94, flexGrow: 0, flexShrink: 1, minWidth: 88 },
  center: { alignItems: 'center', gap: spacing.md, paddingVertical: spacing.xxl },
  empty: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, gap: spacing.md, minWidth: 0, padding: spacing.xl },
  emptyTitle: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.heading, fontWeight: '500', textAlign: 'center' },
  emptyText: { color: colors.text.secondary, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 22, textAlign: 'center' },
});
