import { Feather } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React from 'react';
import { ActivityIndicator, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { colors, fontFamily, fontSize, MIN_TOUCH_TARGET, radius, scaleSize, spacing } from '../../theme';

/**
 * The chrome every settings sub-page shares: a back header, a scroll body, and — only for pages that own
 * editable fields — one save button at the bottom.
 *
 * Settings was a single screen with one "Save changes" button sitting between the privacy switch and "Export
 * my data". It actually saved the caregiver's name, phone, three notification preferences AND that switch,
 * while the loved-one rows next to it saved themselves on their own screen. Nothing about its position said
 * any of that, so pressing it was a guess.
 *
 * One page owning one group fixes the meaning rather than the styling: a save button here can only be about
 * the fields above it, because there is nothing else on the page.
 */
export function SettingsSubPage({
  children,
  isSaving = false,
  onSave,
  saveLabel = 'Save changes',
  subtitle,
  title,
}: {
  children: React.ReactNode;
  isSaving?: boolean;
  /** Omit on a page that only navigates or toggles-and-writes immediately — then no button is rendered. */
  onSave?: () => void;
  saveLabel?: string;
  subtitle?: string;
  title: string;
}) {
  const router = useRouter();

  return (
    <SafeAreaView style={styles.safe} edges={['top', 'bottom']}>
      <View style={styles.header}>
        <TouchableOpacity
          accessibilityLabel="Back"
          accessibilityRole="button"
          hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
          onPress={() => router.back()}
          style={styles.backButton}
        >
          <Feather color={colors.text.primary} name="chevron-left" size={24} />
        </TouchableOpacity>
        {/* The title is the page's accessible heading, so it must not be a decorative sibling of the back arrow. */}
        <Text accessibilityRole="header" maxFontSizeMultiplier={1.3} style={styles.title}>{title}</Text>
      </View>

      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
        {children}
      </ScrollView>

      {onSave ? (
        <View style={styles.footer}>
          <TouchableOpacity
            // Spelled out because the spinner replaces the visible text while saving.
            accessibilityLabel={isSaving ? 'Saving' : saveLabel}
            accessibilityRole="button"
            accessibilityState={{ busy: isSaving, disabled: isSaving }}
            disabled={isSaving}
            onPress={onSave}
            style={[styles.saveButton, isSaving && styles.saveButtonDisabled]}
          >
            {isSaving
              ? <ActivityIndicator color={colors.text.onAccent} />
              : <Text style={styles.saveButtonText}>{saveLabel}</Text>}
          </TouchableOpacity>
        </View>
      ) : null}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { backgroundColor: colors.surface.page, flex: 1 },
  header: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: spacing.xs,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  backButton: {
    alignItems: 'center',
    height: MIN_TOUCH_TARGET,
    justifyContent: 'center',
    width: MIN_TOUCH_TARGET,
  },
  title: {
    color: colors.text.primary,
    flexShrink: 1,
    fontFamily: fontFamily.display,
    fontSize: fontSize.title,
  },
  content: { paddingBottom: spacing.xxl },
  subtitle: {
    color: colors.text.secondary,
    fontSize: fontSize.body,
    lineHeight: scaleSize(21),
    paddingBottom: spacing.md,
    paddingHorizontal: spacing.xl,
  },
  // The button sits outside the ScrollView so it cannot be scrolled away from — on a short screen the old
  // in-flow button could be below the fold with nothing indicating it was there.
  footer: {
    backgroundColor: colors.surface.page,
    borderTopColor: colors.border.default,
    borderTopWidth: 1,
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.md,
  },
  saveButton: {
    alignItems: 'center',
    backgroundColor: colors.accent,
    borderRadius: radius.lg,
    justifyContent: 'center',
    minHeight: MIN_TOUCH_TARGET,
    paddingVertical: spacing.md,
  },
  saveButtonDisabled: { opacity: 0.6 },
  saveButtonText: {
    color: colors.text.onAccent,
    fontSize: fontSize.subheading,
    fontWeight: '600',
  },
});
