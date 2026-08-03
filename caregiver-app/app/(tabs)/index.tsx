import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

import { useCaregiver } from '../../src/architecture/CaregiverContext';
import { ConfigurationBanner, PrimaryButton, ScreenLayout } from '../../src/components/AppUI';
import { BrandLockup } from '../../src/components/BrandLockup';
import { loadCaregiverHome, type CaregiverHome } from '../../src/lib/v1Caregiver';
import { usePatientStatusesV1 } from '../../src/lib/v1Client';
import { formatLastInteraction, getConversationsTodayText } from '../../src/lib/v1Status';
import { useTabBarClearance } from '../../src/lib/useTabBarClearance';
import { colors, contentColumn, fontFamily, fontSize, radius, spacing } from '../../src/theme';

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
    setLoading(true); setError('');
    try {
      const nextHome = await loadCaregiverHome();
      setHome(nextHome);
      setNotificationsEnabled(nextHome.caregiver.notificationPreferences.pushNotificationsEnabled);
    }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'We could not load your household yet.'); }
    finally { setLoading(false); }
  }, []);
  useFocusEffect(useCallback(() => { void refresh(); void loadSetupProgress(); }, [loadSetupProgress, refresh]));

  const incomplete = Object.values(setup).some((state) => state === 'not-started' || state === 'in-progress');
  return <ScreenLayout bottomInset={clearance} contentContainerStyle={styles.content}>
    <View style={styles.brand}><BrandLockup compact /><TouchableOpacity accessibilityLabel="Open setup" accessibilityRole="button" onPress={() => router.push('/setup')} style={styles.settings}><Feather color={colors.accent} name="sliders" size={20} /></TouchableOpacity></View>
    <Text accessibilityRole="header" style={styles.greeting}>Good morning</Text>
    <Text style={styles.date}>Here are your family’s connected updates.</Text>
    {incomplete ? <ConfigurationBanner title="Complete your Reflexion setup" detail={`${Object.values(setup).filter((state) => state === 'complete').length} of 8 sections complete`} action="Continue setup" onPress={() => router.push('/setup')} /> : null}
    {!notificationsEnabled ? <ConfigurationBanner title="Turn on notifications" detail="Allow Reflexion to tell you when an update may need your attention." action="Enable notifications" onPress={() => router.push('/settings/notifications')} /> : null}
    {loading ? <View style={styles.center}><ActivityIndicator color={colors.accent} /><Text style={styles.emptyText}>Loading your connected household…</Text></View> : null}
    {error ? <View style={styles.empty}><Text style={styles.emptyTitle}>Updates are unavailable</Text><Text style={styles.emptyText}>{error}</Text><PrimaryButton label="Try again" onPress={() => void refresh()} /></View> : null}
    {!loading && !error && !home?.patients.length ? <View style={styles.empty}><Text style={styles.emptyTitle}>Add a loved one to begin</Text><Text style={styles.emptyText}>Once you add someone and pair their device, their factual interaction updates will appear here.</Text><PrimaryButton label="Add a loved one" onPress={() => router.push('/setup/household')} /></View> : null}
    <View style={styles.cards}>{home?.patients.map((person, index) => {
      const status = statusSlots[index];
      const hasDevice = Boolean(person.deviceId);
      const interaction = status?.data;
      const headline = !hasDevice ? 'No device paired' : !interaction ? 'Waiting for the first interaction' : interaction.completedToday ? 'Interaction recorded today' : 'No interaction recorded today';
      const detail = !hasDevice ? 'Pair a Reflexion Mirror before interaction updates can be shown.'
        : !interaction ? 'The device is paired. Updates will appear after a real interaction is recorded.'
        : `${getConversationsTodayText(interaction) || 'No conversation count is available yet.'} Last interaction · ${formatLastInteraction(interaction.lastInteractionAt)}.`;
      const action = !hasDevice ? 'Pair device' : 'Leave a message';
      return <View key={person.patientId} style={styles.card}>
        <TouchableOpacity accessibilityLabel={`${person.displayName}. ${headline}. Open details`} accessibilityRole="button" activeOpacity={0.86} onPress={() => router.push(`/loved-one/${person.patientId}`)} style={styles.top}><View style={styles.avatar}><Text style={styles.avatarText}>{person.displayName.slice(0, 1)}</Text></View><View style={styles.nameBlock}><Text style={styles.name}>{person.displayName}</Text><Text style={styles.headline}>{headline}</Text></View><Feather color={colors.textDecorative} name="chevron-right" size={24} /></TouchableOpacity>
        <View style={styles.divider} /><Text style={styles.fact}>{detail}</Text>
        <View style={styles.actionRow}><View style={styles.primaryWrap}><PrimaryButton label={action} onPress={() => router.push(hasDevice ? `/chat/${person.patientId}/compose` : `/device/${person.patientId}/pairing`)} /></View></View>
      </View>;
    })}</View>
  </ScreenLayout>;
}

const styles = StyleSheet.create({
  content: { gap: spacing.xl, paddingTop: spacing.lg }, brand: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between' }, settings: { alignItems: 'center', justifyContent: 'center', minHeight: 44, width: 44 }, greeting: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.display, fontWeight: '500', lineHeight: 41, marginTop: spacing.md }, date: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 25, marginTop: -spacing.md }, cards: { gap: spacing.lg }, card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: 22, borderWidth: 1, padding: spacing.xl, shadowColor: colors.shadow, shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.055, shadowRadius: 20 }, top: { alignItems: 'center', flexDirection: 'row', gap: spacing.md }, avatar: { alignItems: 'center', backgroundColor: '#EADFCF', borderRadius: 999, flexShrink: 0, height: 76, justifyContent: 'center', width: 76 }, avatarText: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: 28, fontWeight: '500' }, nameBlock: { flex: 1, gap: spacing.sm }, name: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.display, fontSize: 26, fontWeight: '500', lineHeight: 32 }, headline: { color: colors.accent, fontSize: fontSize.caption, fontWeight: '700' }, divider: { backgroundColor: colors.border.subtle, height: 1, marginVertical: spacing.lg }, fact: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 23 }, actionRow: { flexDirection: 'row', gap: spacing.md, marginTop: spacing.xl }, primaryWrap: { flex: 1 }, center: { alignItems: 'center', gap: spacing.md, paddingVertical: spacing.xxl }, empty: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, gap: spacing.md, padding: spacing.xl }, emptyTitle: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.heading, fontWeight: '500' }, emptyText: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 },
});
