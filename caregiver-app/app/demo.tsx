import { useRouter } from 'expo-router';
import React, { useEffect, useMemo } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { useQueryClient } from '@tanstack/react-query';

import { AppHeader, PrimaryButton, ScreenLayout, SecondaryButton, SelectionButton, TertiaryButton } from '../src/components/AppUI';
import { isDemoFeatureEnabled, enterDemoMode, exitDemoMode, useDemoMode } from '../src/demo/demoMode';
import { DEFAULT_DEMO_SCENARIO, getDemoScenario, resetDemoData, setDemoScenario, subscribeDemoRepository, type DemoScenario } from '../src/demo/demoRepository';
import { colors, fontFamily, fontSize, radius, spacing } from '../src/theme';

const routeLinks = [
  ['Home', '/(tabs)'],
  ['Welcome', '/welcome'],
  ['Setup overview', '/setup'],
  ['Setup household', '/setup/household'],
  ['Household review', '/setup/household-review'],
  ['Select device', '/device/select'],
  ['Setup review', '/setup/review'],
  ['Setup complete', '/setup/complete'],
  ['Loved one profile', '/loved-one/demo-margaret'],
  ['Sessions', '/loved-one/demo-margaret/sessions'],
  ['Session detail', '/loved-one/demo-margaret/sessions/demo-session-complete'],
  ['Trends', '/loved-one/demo-margaret/trends'],
  ['History', '/loved-one/demo-margaret/history'],
  ['Weekly summary', '/loved-one/demo-margaret/weekly-summary'],
  ['Export summaries', '/loved-one/demo-margaret/export'],
  ['Activity', '/(tabs)/activity'],
  ['Family messages', '/(tabs)/chat'],
  ['Chat thread', '/chat/demo-margaret'],
  ['Text composer', '/chat/demo-margaret/compose'],
  ['Photo composer', '/chat/demo-margaret/photo'],
  ['Voice composer', '/chat/demo-margaret/voice'],
  ['Routines', '/settings/routines'],
  ['Notifications', '/settings/notifications'],
  ['Consent & control', '/settings/consent'],
  ['Research participation', '/research/overview'],
  ['Away mode', '/settings/away'],
  ['Device health', '/settings/devices'],
  ['Device detail', '/device/demo-margaret/detail'],
  ['Device troubleshooting', '/device/demo-margaret/troubleshooting'],
  ['Care Circle', '/settings/care-circle'],
  ['Privacy & data', '/settings/privacy'],
  ['Loved Ones', '/settings/household'],
  ['Language & accessibility', '/settings/language'],
  ['Account', '/settings/account'],
  ['Personal details', '/settings/account/personal'],
  ['Sign-in methods', '/settings/account/sign-in-methods'],
  ['Help Centre', '/settings/help'],
  ['Contact Support', '/settings/contact-support'],
  ['Subscription', '/settings/subscription'],
  ['Payment method', '/settings/payment-method'],
  ['About Reflexion', '/settings/about'],
] as const;

type Option<T extends string> = { value: T; label: string };

function OptionGroup<T extends string>({ title, value, options, onChange }: { title: string; value: T; options: Option<T>[]; onChange: (value: T) => void }) {
  return <View style={styles.group}>
    <Text style={styles.groupTitle}>{title}</Text>
    <View style={styles.options}>{options.map((option) => <View key={option.value} style={styles.option}><SelectionButton label={option.label} selected={value === option.value} onPress={() => onChange(option.value)} /></View>)}</View>
  </View>;
}

export default function DemoScreen() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const active = useDemoMode();
  const [scenario, setScenario] = React.useState<DemoScenario>(() => getDemoScenario());

  useEffect(() => {
    if (!isDemoFeatureEnabled()) return;
    void enterDemoMode();
    return subscribeDemoRepository(() => setScenario(getDemoScenario()));
  }, []);

  const change = <K extends keyof DemoScenario>(key: K, value: DemoScenario[K]) => {
    setDemoScenario({ [key]: value } as Partial<DemoScenario>);
    queryClient.clear();
  };

  const currentPatientRoute = useMemo(() => scenario.lovedOne === 'second' ? 'demo-james' : 'demo-margaret', [scenario.lovedOne]);
  if (!isDemoFeatureEnabled()) return <ScreenLayout><Text style={styles.title}>Demo mode is not available in this build.</Text><SecondaryButton label="Back to sign in" onPress={() => router.replace('/sign-in')} /></ScreenLayout>;

  return <ScreenLayout contentContainerStyle={styles.content}>
    <AppHeader title="Demo controls" onBack={() => router.back()} />
    <Text accessibilityRole="header" style={styles.title}>Local caregiver demo</Text>
    <Text style={styles.subtitle}>Use the real caregiver navigation with local fixtures. Changes here stay on this device and are never synced.</Text>
    <View style={styles.notice}><Text style={styles.noticeTitle}>Demo only — not synced</Text><Text style={styles.noticeCopy}>No production credentials, profiles, messages, routines, consent or Mirror data are written in this mode.</Text></View>

    <OptionGroup title="Loved one" value={scenario.lovedOne} options={[{ value: 'margaret', label: 'Margaret' }, { value: 'second', label: 'Second loved one' }]} onChange={(value) => change('lovedOne', value)} />
    <OptionGroup title="Device" value={scenario.device} options={[{ value: 'online', label: 'Online' }, { value: 'offline', label: 'Offline' }]} onChange={(value) => change('device', value)} />
    <OptionGroup title="Setup" value={scenario.setup} options={[{ value: 'complete', label: 'Complete' }, { value: 'incomplete', label: 'Incomplete' }]} onChange={(value) => change('setup', value)} />
    <OptionGroup title="Session" value={scenario.session} options={[{ value: 'none', label: 'None' }, { value: 'completed', label: 'Completed' }, { value: 'partial', label: 'Partial' }, { value: 'processing', label: 'Processing' }]} onChange={(value) => change('session', value)} />
    <OptionGroup title="Routine response" value={scenario.routineResponse} options={[{ value: 'presented', label: 'Presented' }, { value: 'reported-complete', label: 'Reported complete' }, { value: 'deferred', label: 'Deferred' }, { value: 'declined', label: 'Declined' }, { value: 'no-response', label: 'No response' }]} onChange={(value) => change('routineResponse', value)} />
    <OptionGroup title="Message type" value={scenario.messageType} options={[{ value: 'text', label: 'Text' }, { value: 'photo', label: 'Photo' }, { value: 'voice', label: 'Voice' }]} onChange={(value) => change('messageType', value)} />
    <OptionGroup title="Message delivery" value={scenario.messageDelivery} options={[{ value: 'queued', label: 'Queued' }, { value: 'delivered', label: 'Delivered' }, { value: 'failed', label: 'Failed' }]} onChange={(value) => change('messageDelivery', value)} />
    <OptionGroup title="Message interaction" value={scenario.messageInteraction} options={[{ value: 'none', label: 'Not opened' }, { value: 'viewed', label: 'Viewed' }, { value: 'played', label: 'Played' }, { value: 'replayed', label: 'Replayed' }]} onChange={(value) => change('messageInteraction', value)} />
    <OptionGroup title="Product consent" value={scenario.consent} options={[{ value: 'pending', label: 'Pending' }, { value: 'accepted', label: 'Accepted' }, { value: 'declined', label: 'Declined' }, { value: 'withdrawn', label: 'Withdrawn' }]} onChange={(value) => change('consent', value)} />
    <OptionGroup title="Product control" value={scenario.control} options={[{ value: 'active', label: 'Active' }, { value: 'paused', label: 'Paused' }]} onChange={(value) => change('control', value)} />
    <OptionGroup title="Away mode" value={scenario.away} options={[{ value: 'off', label: 'Off' }, { value: 'on', label: 'On' }]} onChange={(value) => change('away', value)} />
    <OptionGroup title="Research" value={scenario.research} options={[{ value: 'invited', label: 'Invited' }, { value: 'not-invited', label: 'Not invited' }]} onChange={(value) => change('research', value)} />

    <View style={styles.actions}><PrimaryButton label="Open caregiver app" onPress={() => router.replace('/(tabs)')} /><SecondaryButton label="Reset demo data" onPress={() => { resetDemoData(); queryClient.clear(); }} /><TertiaryButton label="Exit Demo Mode" onPress={() => { void exitDemoMode().then(() => router.replace('/sign-in')); }} /></View>
    <Text style={styles.sectionTitle}>Explore final caregiver screens</Text>
    <Text style={styles.helper}>These links open the existing routes and production screen components with the current fixture state.</Text>
    <View style={styles.links}>{routeLinks.map(([label, route]) => <SecondaryButton key={label} label={label} onPress={() => router.push(route.replace('demo-margaret', currentPatientRoute) as never)} />)}</View>
    {!active ? <Text style={styles.helper}>Starting demo mode…</Text> : null}
  </ScreenLayout>;
}

const styles = StyleSheet.create({
  content: { gap: spacing.lg, minWidth: 0 },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 38, marginTop: spacing.lg, minWidth: 0 },
  subtitle: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 25, minWidth: 0 },
  notice: { backgroundColor: '#EEF7F0', borderColor: '#CDE2D6', borderRadius: radius.xl, borderWidth: 1, gap: spacing.xs, minWidth: 0, padding: spacing.lg },
  noticeTitle: { color: colors.accent, fontSize: fontSize.bodyLarge, fontWeight: '800' },
  noticeCopy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 21 },
  group: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, minWidth: 0, padding: spacing.lg },
  groupTitle: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '800' },
  options: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, minWidth: 0 },
  option: { flexGrow: 1, flexShrink: 1, minWidth: 112 },
  actions: { gap: spacing.md, minWidth: 0 },
  sectionTitle: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.heading, fontWeight: '500', marginTop: spacing.md },
  helper: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 21 },
  links: { gap: spacing.sm, minWidth: 0 },
});
