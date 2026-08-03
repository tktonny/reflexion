import { Feather } from '@expo/vector-icons';
import React from 'react';
import { KeyboardAvoidingView, Platform, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import type { ReactNode } from 'react';
import type { StyleProp, ViewStyle } from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';

import type { InteractionState, SetupStatus } from '../architecture/models';
import { colors, contentColumn, fontFamily, fontSize, layout, MIN_TOUCH_TARGET, radius, spacing } from '../theme';

type IconName = keyof typeof Feather.glyphMap;

/**
 * Shared screen chrome: safe areas, one content boundary, keyboard avoidance and overflow scrolling.
 * Individual routes may add spacing, but should not replace this wrapper with ad-hoc geometry.
 */
export function ScreenLayout({
  children,
  contentContainerStyle,
  keyboardAware = true,
  scroll = true,
  bottomInset = 0,
  footer,
}: {
  children: ReactNode;
  contentContainerStyle?: StyleProp<ViewStyle>;
  keyboardAware?: boolean;
  scroll?: boolean;
  bottomInset?: number;
  footer?: ReactNode;
}) {
  const insets = useSafeAreaInsets();
  const contentStyle = [
    styles.screenContent,
    contentColumn,
    { paddingBottom: layout.bottomPadding + bottomInset + insets.bottom },
    contentContainerStyle,
  ];
  const body = scroll
    ? <ScrollView style={styles.flex} contentContainerStyle={contentStyle} keyboardShouldPersistTaps="handled" keyboardDismissMode={Platform.OS === 'ios' ? 'interactive' : 'on-drag'}>{children}</ScrollView>
    : <View style={[styles.flex, contentStyle]}>{children}</View>;
  const bodyWithFooter = <>{body}{footer}</>;
  const wrapped = keyboardAware
    ? <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} keyboardVerticalOffset={layout.keyboardOffset} style={styles.flex}>{bodyWithFooter}</KeyboardAvoidingView>
    : bodyWithFooter;
  return <SafeAreaView edges={['top', 'bottom']} style={styles.safe}>{wrapped}</SafeAreaView>;
}

export function AppHeader({ title, onBack }: { title?: string; onBack?: () => void }) {
  return (
    <View style={styles.header}>
      {onBack ? <TouchableOpacity accessibilityLabel="Go back" accessibilityRole="button" onPress={onBack} style={styles.back}><Feather color={colors.text.primary} name="chevron-left" size={24} /></TouchableOpacity> : <View style={styles.back} />}
      {title ? <Text accessibilityRole="header" style={styles.headerTitle}>{title}</Text> : <View style={styles.headerSpacer} />}
      <View style={styles.back} />
    </View>
  );
}

export function PrimaryButton({ label, onPress, disabled = false }: { label: string; onPress: () => void; disabled?: boolean }) {
  return <TouchableOpacity accessibilityRole="button" accessibilityState={{ disabled }} activeOpacity={0.82} disabled={disabled} onPress={onPress} style={[styles.primary, disabled && styles.disabled]}><Text style={styles.primaryText}>{label}</Text></TouchableOpacity>;
}

export function SecondaryButton({ label, onPress, accessibilityLabel }: { label: string; onPress: () => void; accessibilityLabel?: string }) {
  return <TouchableOpacity accessibilityLabel={accessibilityLabel || label} accessibilityRole="button" activeOpacity={0.82} onPress={onPress} style={styles.secondary}><Text style={styles.secondaryText}>{label}</Text></TouchableOpacity>;
}

export function TertiaryButton({ label, onPress, disabled = false }: { label: string; onPress: () => void; disabled?: boolean }) {
  return <TouchableOpacity accessibilityRole="button" accessibilityState={{ disabled }} disabled={disabled} onPress={onPress} style={[styles.tertiary, disabled && styles.disabled]}><Text style={styles.tertiaryText}>{label}</Text></TouchableOpacity>;
}

export function ChoiceCard({ icon, title, description, selected = false, onPress }: { icon: IconName; title: string; description: string; selected?: boolean; onPress: () => void }) {
  return (
    <TouchableOpacity accessibilityLabel={`${title}. ${description}`} accessibilityRole="button" accessibilityState={{ selected }} onPress={onPress} style={[styles.choice, selected && styles.choiceSelected]}>
      <View style={styles.choiceIcon}><Feather color={selected ? colors.accent : colors.textDecorative} name={selected ? 'check-circle' : icon} size={23} /></View>
      <View style={styles.choiceCopy}><Text style={styles.choiceTitle}>{title}</Text><Text style={styles.choiceDescription}>{description}</Text></View>
      <Feather color={colors.textDecorative} name="chevron-right" size={20} />
    </TouchableOpacity>
  );
}

/** Informational card for architecture items that are described but not selectable on this screen. */
export function InfoCard({ icon, title, description }: { icon: IconName; title: string; description: string }) {
  return <View style={styles.choice}><View style={styles.choiceIcon}><Feather color={colors.textDecorative} name={icon} size={23} /></View><View style={styles.choiceCopy}><Text style={styles.choiceTitle}>{title}</Text><Text style={styles.choiceDescription}>{description}</Text></View></View>;
}

const INTERACTION_META: Record<InteractionState, { label: string; icon: IconName; tone: 'green' | 'amber' | 'grey' | 'red' }> = {
  'interaction-recorded-today': { label: 'Interaction recorded today', icon: 'check-circle', tone: 'green' },
  'no-interaction-yet-today': { label: 'No interaction yet today', icon: 'clock', tone: 'amber' },
  'recent-interaction-shorter-than-usual': { label: 'Recent interaction shorter than usual', icon: 'bar-chart-2', tone: 'amber' },
  'device-may-be-offline': { label: 'Device may be offline', icon: 'wifi-off', tone: 'grey' },
  'needs-your-attention': { label: 'Needs your attention', icon: 'alert-circle', tone: 'red' },
};

export function StatusPill({ state }: { state: InteractionState }) {
  const meta = INTERACTION_META[state];
  const theme = {
    green: { bg: colors.status.greenBg, color: colors.status.green },
    amber: { bg: colors.status.amberBg, color: colors.status.amber },
    grey: { bg: colors.status.greyBg, color: colors.status.grey },
    red: { bg: colors.status.redBg, color: colors.status.red },
  }[meta.tone];
  return <View accessible accessibilityLabel={meta.label} style={[styles.pill, { backgroundColor: theme.bg }]}><Feather color={theme.color} name={meta.icon} size={16} /><Text style={[styles.pillText, { color: theme.color }]}>{meta.label}</Text></View>;
}

export function SetupProgressCard({ title, description, status, onPress }: { title: string; description: string; status: SetupStatus; onPress: () => void }) {
  const label = status === 'not-started' ? 'Not started' : status === 'in-progress' ? 'In progress' : status === 'complete' ? 'Complete' : 'Skipped';
  return <ChoiceCard icon={status === 'complete' ? 'check-circle' : 'circle'} title={title} description={`${label} · ${description}`} onPress={onPress} selected={status === 'complete'} />;
}

export function ConfigurationBanner({ title, detail, action, onPress }: { title: string; detail: string; action: string; onPress: () => void }) {
  return <View style={styles.banner}><View style={styles.bannerIcon}><Feather color={colors.status.amber} name="info" size={20} /></View><View style={styles.bannerCopy}><Text style={styles.bannerTitle}>{title}</Text><Text style={styles.bannerDetail}>{detail}</Text><TertiaryButton label={action} onPress={onPress} /></View></View>;
}

export function ProvenanceSection({ label, children }: { label: string; children: React.ReactNode }) {
  return <View style={styles.provenance}><Text style={styles.provenanceLabel}>{label}</Text><Text style={styles.provenanceValue}>{children}</Text></View>;
}

export function SettingsRow({ icon, label, value, onPress, disabled = false }: { icon: IconName; label: string; value?: string; onPress?: () => void; disabled?: boolean }) {
  const inactive = disabled || !onPress;
  return <TouchableOpacity accessibilityLabel={value ? `${label}. ${value}` : label} accessibilityRole={inactive ? 'text' : 'button'} accessibilityState={{ disabled: inactive }} disabled={inactive} onPress={onPress} style={[styles.settingsRow, inactive && styles.disabled]}><View style={styles.settingsIcon}><Feather color={colors.accent} name={icon} size={19} /></View><View style={styles.settingsCopy}><Text style={styles.settingsLabel}>{label}</Text>{value ? <Text style={styles.settingsValue}>{value}</Text> : null}</View>{inactive ? null : <Feather color={colors.textDecorative} name="chevron-right" size={20} />}</TouchableOpacity>;
}

const styles = StyleSheet.create({
  safe: { backgroundColor: colors.surface.page, flex: 1 },
  flex: { flex: 1 },
  screenContent: { gap: spacing.lg, paddingHorizontal: layout.horizontalPadding, paddingTop: layout.verticalPadding },
  header: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between', minHeight: 52, paddingVertical: spacing.xs, width: '100%' },
  back: { alignItems: 'center', justifyContent: 'center', minHeight: MIN_TOUCH_TARGET, width: MIN_TOUCH_TARGET },
  headerTitle: { color: colors.text.primary, flex: 1, flexShrink: 1, fontSize: fontSize.bodyLarge, fontWeight: '600', lineHeight: 22, textAlign: 'center' },
  headerSpacer: { flex: 1 },
  primary: { alignItems: 'center', backgroundColor: colors.accent, borderRadius: radius.lg, justifyContent: 'center', minHeight: 54, paddingHorizontal: spacing.xl },
  primaryText: { color: colors.text.onAccent, flexShrink: 1, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 22, textAlign: 'center' },
  secondary: { alignItems: 'center', borderColor: colors.accent, borderRadius: radius.lg, borderWidth: 1.5, justifyContent: 'center', minHeight: 52, paddingHorizontal: spacing.xl },
  secondaryText: { color: colors.accent, flexShrink: 1, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 22, textAlign: 'center' },
  tertiary: { alignItems: 'center', alignSelf: 'flex-start', justifyContent: 'center', minHeight: MIN_TOUCH_TARGET, paddingRight: spacing.md },
  tertiaryText: { color: colors.accent, flexShrink: 1, fontSize: fontSize.body, fontWeight: '700', lineHeight: 20, textAlign: 'center' },
  disabled: { opacity: 0.45 },
  choice: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.lg, borderWidth: 1, flexDirection: 'row', gap: spacing.md, minHeight: 76, padding: spacing.lg },
  choiceSelected: { backgroundColor: '#F2F8F6', borderColor: colors.accent, borderWidth: 1.5 },
  choiceIcon: { alignItems: 'center', backgroundColor: '#EEF3E9', borderRadius: radius.pill, height: 42, justifyContent: 'center', width: 42 },
  choiceCopy: { flex: 1 },
  choiceTitle: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 22 },
  choiceDescription: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.caption, lineHeight: 18, marginTop: 2 },
  pill: { alignItems: 'center', alignSelf: 'flex-start', borderRadius: radius.pill, flexDirection: 'row', gap: 7, minHeight: 34, paddingHorizontal: spacing.md, paddingVertical: spacing.xs },
  pillText: { fontSize: fontSize.body, fontWeight: '700' },
  banner: { backgroundColor: colors.status.amberBg, borderColor: '#EBCF9F', borderRadius: radius.lg, borderWidth: 1, flexDirection: 'row', gap: spacing.md, padding: spacing.lg },
  bannerIcon: { paddingTop: 2 }, bannerCopy: { flex: 1, flexShrink: 1 }, bannerTitle: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 22 }, bannerDetail: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 21, marginTop: 4 },
  provenance: { borderBottomColor: colors.border.subtle, borderBottomWidth: 1, gap: 4, paddingVertical: spacing.lg },
  provenanceLabel: { color: colors.text.secondary, fontSize: fontSize.caption, fontWeight: '700' }, provenanceValue: { color: colors.text.primary, fontSize: fontSize.body, lineHeight: 23 },
  settingsRow: { alignItems: 'center', backgroundColor: colors.surface.card, borderBottomColor: colors.border.subtle, borderBottomWidth: 1, flexDirection: 'row', gap: spacing.md, minHeight: 70, paddingHorizontal: spacing.lg, paddingVertical: spacing.md },
  settingsIcon: { alignItems: 'center', backgroundColor: '#EEF3E9', borderRadius: radius.pill, flexShrink: 0, height: 38, justifyContent: 'center', width: 38 }, settingsCopy: { flex: 1, flexShrink: 1 }, settingsLabel: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, fontWeight: '600', lineHeight: 22 }, settingsValue: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.caption, lineHeight: 18, marginTop: 2 },
});
