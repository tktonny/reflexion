import { Feather } from '@expo/vector-icons';
import React from 'react';
import { KeyboardAvoidingView, Platform, ScrollView, StyleSheet, Text, View } from 'react-native';
import type { ReactNode } from 'react';
import type { StyleProp, ViewStyle } from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';

import type { InteractionState, SetupStatus } from '../architecture/models';
import { MotionCheckmark, MotionPressable } from './Motion';
import { cardShadow, colors, contentColumn, fontFamily, fontSize, layout, MIN_TOUCH_TARGET, radius, spacing, typography } from '../theme';

export type IconName = keyof typeof Feather.glyphMap;

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
  scrollRef,
}: {
  children: ReactNode;
  contentContainerStyle?: StyleProp<ViewStyle>;
  keyboardAware?: boolean;
  scroll?: boolean;
  bottomInset?: number;
  footer?: ReactNode;
  scrollRef?: React.RefObject<ScrollView | null>;
}) {
  const insets = useSafeAreaInsets();
  const contentStyle = [
    styles.screenContent,
    contentColumn,
    { paddingBottom: layout.bottomPadding + bottomInset + insets.bottom },
    contentContainerStyle,
  ];
  const body = scroll
    ? <ScrollView ref={scrollRef} style={styles.flex} contentContainerStyle={contentStyle} keyboardShouldPersistTaps="handled" keyboardDismissMode={Platform.OS === 'ios' ? 'interactive' : 'on-drag'}>{children}</ScrollView>
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
      {onBack ? <MotionPressable accessibilityLabel="Go back" accessibilityRole="button" onPress={onBack} style={styles.back}><Feather color={colors.text.primary} name="chevron-left" size={24} /></MotionPressable> : <View style={styles.back} />}
      {title ? <Text accessibilityRole="header" style={styles.headerTitle}>{title}</Text> : <View style={styles.headerSpacer} />}
      <View style={styles.back} />
    </View>
  );
}

export function PrimaryButton({ label, onPress, disabled = false, icon }: { label: string; onPress: () => void; disabled?: boolean; icon?: IconName }) {
  return <MotionPressable accessibilityRole="button" accessibilityState={{ disabled }} disabled={disabled} onPress={onPress} style={[styles.primary, disabled && styles.disabled]}><View style={styles.buttonContent}>{icon ? <Feather color={colors.text.onAccent} name={icon} size={21} /> : null}<Text style={styles.primaryText}>{label}</Text></View></MotionPressable>;
}

export function SecondaryButton({ label, onPress, accessibilityLabel, icon, disabled = false }: { label: string; onPress: () => void; accessibilityLabel?: string; icon?: IconName; disabled?: boolean }) {
  return <MotionPressable accessibilityLabel={accessibilityLabel || label} accessibilityRole="button" accessibilityState={{ disabled }} disabled={disabled} onPress={onPress} style={[styles.secondary, disabled && styles.disabled]}><View style={styles.buttonContent}>{icon ? <Feather color={colors.accent} name={icon} size={21} /> : null}<Text style={styles.secondaryText}>{label}</Text></View></MotionPressable>;
}

export function TertiaryButton({ label, onPress, disabled = false }: { label: string; onPress: () => void; disabled?: boolean }) {
  return <MotionPressable accessibilityRole="button" accessibilityState={{ disabled }} disabled={disabled} onPress={onPress} style={[styles.tertiary, disabled && styles.disabled]}><Text style={styles.tertiaryText}>{label}</Text></MotionPressable>;
}

export function SurfaceCard({ children, style }: { children: ReactNode; style?: StyleProp<ViewStyle> }) {
  return <View style={[styles.surfaceCard, style]}>{children}</View>;
}

export function ChoiceCard({ icon, title, description, selected = false, onPress, navigates = false }: { icon: IconName; title: string; description: string; selected?: boolean; onPress: () => void; navigates?: boolean }) {
  return (
    <MotionPressable accessibilityLabel={`${title}. ${description}`} accessibilityRole="button" accessibilityState={{ selected }} feedback="card" haptic={navigates ? undefined : 'selection'} onPress={onPress} style={[styles.choice, selected && styles.choiceSelected]}>
      <View style={styles.choiceIcon}><Feather color={selected ? colors.accent : colors.textDecorative} name={selected ? 'check-circle' : icon} size={23} /></View>
      <View style={styles.choiceCopy}><Text style={styles.choiceTitle}>{title}</Text><Text style={styles.choiceDescription}>{description}</Text></View>
      {navigates ? <Feather color={colors.textDecorative} name="chevron-right" size={20} /> : null}
    </MotionPressable>
  );
}

/** A compact selection control. It deliberately has no chevron because it does not navigate. */
export function SelectionButton({ label, selected = false, onPress }: { label: string; selected?: boolean; onPress: () => void }) {
  return <MotionPressable accessibilityLabel={label} accessibilityRole="button" accessibilityState={{ selected }} haptic="selection" onPress={onPress} style={[styles.selectionButton, selected && styles.selectionButtonSelected]}><Text style={[styles.selectionButtonText, selected && styles.selectionButtonTextSelected]}>{label}</Text><MotionCheckmark visible={selected} /></MotionPressable>;
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
  const label = status === 'not-started'
    ? 'Not started'
    : status === 'in-progress'
      ? 'In progress'
      : status === 'complete'
        ? 'Complete'
        : status === 'not-applicable'
          ? 'Not applicable'
          : 'Skipped';
  const satisfied = status === 'complete' || status === 'not-applicable';
  return <ChoiceCard icon={satisfied ? 'check-circle' : 'circle'} title={title} description={`${label} · ${description}`} navigates onPress={onPress} selected={satisfied} />;
}

export function ConfigurationBanner({ title, detail, action, onPress }: { title: string; detail: string; action: string; onPress: () => void }) {
  return <View style={styles.banner}><View style={styles.bannerIcon}><Feather color={colors.status.amber} name="info" size={20} /></View><View style={styles.bannerCopy}><Text style={styles.bannerTitle}>{title}</Text><Text style={styles.bannerDetail}>{detail}</Text><TertiaryButton label={action} onPress={onPress} /></View></View>;
}

export function ProvenanceSection({ label, children }: { label: string; children: React.ReactNode }) {
  return <View style={styles.provenance}><Text style={styles.provenanceLabel}>{label}</Text><Text style={styles.provenanceValue}>{children}</Text></View>;
}

export function SettingsRow({ icon, label, value, onPress, disabled = false }: { icon: IconName; label: string; value?: string; onPress?: () => void; disabled?: boolean }) {
  const inactive = disabled || !onPress;
  return <MotionPressable accessibilityLabel={value ? `${label}. ${value}` : label} accessibilityRole={inactive ? 'text' : 'button'} accessibilityState={{ disabled: inactive }} disabled={inactive} feedback="card" onPress={onPress} style={[styles.settingsRow, inactive && styles.disabled]}><View style={styles.settingsIcon}><Feather color={colors.accent} name={icon} size={19} /></View><View style={styles.settingsCopy}><Text style={styles.settingsLabel}>{label}</Text>{value ? <Text style={styles.settingsValue}>{value}</Text> : null}</View>{inactive ? null : <Feather color={colors.textDecorative} name="chevron-right" size={20} />}</MotionPressable>;
}

const styles = StyleSheet.create({
  safe: { backgroundColor: colors.surface.page, flex: 1, minWidth: 0 },
  flex: { flex: 1 },
  screenContent: { gap: spacing.lg, minWidth: 0, paddingHorizontal: layout.horizontalPadding, paddingTop: layout.verticalPadding },
  header: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between', minHeight: 52, minWidth: 0, paddingVertical: spacing.xs, width: '100%' },
  back: { alignItems: 'center', justifyContent: 'center', minHeight: MIN_TOUCH_TARGET, width: MIN_TOUCH_TARGET },
  headerTitle: { ...typography.label, color: colors.text.primary, flex: 1, flexShrink: 1, minWidth: 0, textAlign: 'center' },
  headerSpacer: { flex: 1 },
  primary: { alignItems: 'center', alignSelf: 'stretch', backgroundColor: colors.accent, borderRadius: radius.lg, justifyContent: 'center', minHeight: 54, minWidth: 0, paddingHorizontal: spacing.lg, paddingVertical: spacing.md },
  primaryText: { ...typography.button, color: colors.text.onAccent, flexShrink: 1, minWidth: 0, textAlign: 'center' },
  secondary: { alignItems: 'center', alignSelf: 'stretch', borderColor: colors.accent, borderRadius: radius.lg, borderWidth: 1.5, justifyContent: 'center', minHeight: 52, minWidth: 0, paddingHorizontal: spacing.lg, paddingVertical: spacing.md },
  secondaryText: { ...typography.button, color: colors.accent, flexShrink: 1, minWidth: 0, textAlign: 'center' },
  buttonContent: { alignItems: 'center', flexDirection: 'row', flexShrink: 1, gap: spacing.md, justifyContent: 'center', minWidth: 0 },
  tertiary: { alignItems: 'center', alignSelf: 'flex-start', justifyContent: 'center', minHeight: MIN_TOUCH_TARGET, paddingHorizontal: spacing.sm, paddingVertical: spacing.xs },
  tertiaryText: { ...typography.body, color: colors.accent, flexShrink: 1, fontWeight: '700', minWidth: 0, textAlign: 'center' },
  disabled: { opacity: 0.45 },
  surfaceCard: { ...cardShadow, backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, minWidth: 0, padding: spacing.lg },
  choice: { ...cardShadow, alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.lg, borderWidth: 1, flexDirection: 'row', gap: spacing.md, minHeight: 76, minWidth: 0, padding: spacing.lg },
  choiceSelected: { backgroundColor: '#F2F8F6', borderColor: colors.accent, borderWidth: 1.5 },
  choiceIcon: { alignItems: 'center', backgroundColor: '#EEF3E9', borderRadius: radius.pill, height: 42, justifyContent: 'center', width: 42 },
  choiceCopy: { flex: 1, minWidth: 0 },
  choiceTitle: { ...typography.label, color: colors.text.primary, flexShrink: 1, minWidth: 0 },
  choiceDescription: { ...typography.caption, color: colors.text.secondary, flexShrink: 1, marginTop: 2, minWidth: 0 },
  selectionButton: { alignItems: 'center', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.md, borderWidth: 1, flexDirection: 'row', justifyContent: 'space-between', minHeight: 52, minWidth: 0, paddingHorizontal: spacing.lg, paddingVertical: spacing.sm },
  selectionButtonSelected: { backgroundColor: '#E7F3F0', borderColor: colors.accent, borderWidth: 1.5 },
  selectionButtonText: { ...typography.label, color: colors.text.primary, flex: 1, flexShrink: 1, minWidth: 0 },
  selectionButtonTextSelected: { color: colors.accent, fontWeight: '700' },
  pill: { alignItems: 'center', alignSelf: 'flex-start', borderRadius: radius.pill, flexDirection: 'row', gap: 7, maxWidth: '100%', minHeight: 34, minWidth: 0, paddingHorizontal: spacing.md, paddingVertical: spacing.sm },
  pillText: { ...typography.body, flexShrink: 1, fontWeight: '700', minWidth: 0 },
  banner: { backgroundColor: colors.status.amberBg, borderColor: '#EBCF9F', borderRadius: radius.lg, borderWidth: 1, flexDirection: 'row', gap: spacing.md, minWidth: 0, padding: spacing.lg },
  bannerIcon: { flexShrink: 0, paddingTop: 2 }, bannerCopy: { flex: 1, flexShrink: 1, minWidth: 0 }, bannerTitle: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 22, minWidth: 0 }, bannerDetail: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 21, marginTop: 4, minWidth: 0 },
  provenance: { borderBottomColor: colors.border.subtle, borderBottomWidth: 1, gap: 4, minWidth: 0, paddingVertical: spacing.lg },
  provenanceLabel: { ...typography.caption, color: colors.text.secondary, fontWeight: '700' }, provenanceValue: { ...typography.body, color: colors.text.primary, flexShrink: 1, minWidth: 0 },
  settingsRow: { alignItems: 'center', backgroundColor: colors.surface.card, borderBottomColor: colors.border.subtle, borderBottomWidth: 1, flexDirection: 'row', gap: spacing.md, minHeight: 70, minWidth: 0, paddingHorizontal: spacing.lg, paddingVertical: spacing.md },
  settingsIcon: { alignItems: 'center', backgroundColor: '#EEF3E9', borderRadius: radius.pill, flexShrink: 0, height: 38, justifyContent: 'center', width: 38 }, settingsCopy: { flex: 1, flexShrink: 1, minWidth: 0 }, settingsLabel: { ...typography.label, color: colors.text.primary, flexShrink: 1, minWidth: 0 }, settingsValue: { ...typography.caption, color: colors.text.secondary, flexShrink: 1, marginTop: 2, minWidth: 0 },
});
