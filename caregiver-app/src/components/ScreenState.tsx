import React from 'react';
import { ActivityIndicator, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Feather } from '@expo/vector-icons';
import { colors, spacing, radius, fontSize, fontFamily, MIN_TOUCH_TARGET } from '../theme';

// Shared loading / empty / failed placeholders.
//
// Why this exists: 14 of 18 screens used to have no error branch at all, so a failed request rendered as
// an ordinary empty state — "no data yet" and "we could not reach the server" looked identical, which is
// how four dead endpoints went unnoticed in production.
//
// Two rules these components enforce:
//  1. A failure NEVER shows the raw error text. Server strings are for logs, not for a caregiver who is
//     wondering whether their mother is alright ("Not found" was once a real on-screen headline).
//  2. Anything technical is framed as a connection problem, explicitly not as news about the person —
//     the same non-clinical, reassurance-first rule the status screens follow.

type StateProps = {
  title?: string;
  message?: string;
  icon?: keyof typeof Feather.glyphMap;
  /** Rendered as a retry button when provided. */
  onRetry?: () => void;
  retryLabel?: string;
  compact?: boolean;
};

export function LoadingState({ title = 'Bear with us', message }: { title?: string; message?: string }) {
  return (
    <View accessibilityRole="progressbar" accessibilityLabel={title} style={styles.wrap}>
      <ActivityIndicator color={colors.accent} />
      <Text style={styles.title}>{title}</Text>
      {message ? <Text style={styles.message}>{message}</Text> : null}
    </View>
  );
}

export function EmptyState({ title, message, icon = 'inbox', compact }: StateProps) {
  return (
    <View style={[styles.wrap, compact && styles.wrapCompact]}>
      <View style={styles.iconWrap}>
        <Feather name={icon} size={26} color={colors.accent} />
      </View>
      {title ? <Text style={styles.title}>{title}</Text> : null}
      {message ? <Text style={styles.message}>{message}</Text> : null}
    </View>
  );
}

export function ErrorState({
  title = 'We could not load this just now',
  message = 'This is usually a connection problem, not something about your loved one.',
  icon = 'cloud-off',
  onRetry,
  retryLabel = 'Try again',
  compact,
}: StateProps) {
  return (
    <View style={[styles.wrap, compact && styles.wrapCompact]}>
      <View style={styles.iconWrap}>
        <Feather name={icon} size={26} color={colors.accent} />
      </View>
      <Text style={styles.title}>{title}</Text>
      <Text style={styles.message}>{message}</Text>
      {onRetry ? (
        <TouchableOpacity
          accessibilityLabel={retryLabel}
          accessibilityRole="button"
          onPress={onRetry}
          style={styles.retry}
        >
          <Text style={styles.retryText}>{retryLabel}</Text>
        </TouchableOpacity>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    alignItems: 'center',
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: radius.xl,
    borderWidth: 1,
    gap: spacing.sm,
    marginBottom: spacing.lg,
    paddingHorizontal: 24,
    paddingVertical: spacing.xxl,
  },
  wrapCompact: { paddingVertical: spacing.xl },
  iconWrap: {
    alignItems: 'center',
    backgroundColor: colors.surface.muted,
    borderRadius: 28,
    height: 56,
    justifyContent: 'center',
    marginBottom: spacing.xs,
    width: 56,
  },
  title: {
    color: colors.text.primary,
    fontFamily: fontFamily.display,
    fontSize: 19,
    fontWeight: '500',
    textAlign: 'center',
  },
  message: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 21, textAlign: 'center' },
  retry: {
    alignItems: 'center',
    backgroundColor: colors.accent,
    borderRadius: radius.md,
    justifyContent: 'center',
    marginTop: spacing.md,
    // 44pt is the smallest reliable touch target; caregivers use this one-handed and often in a hurry.
    minHeight: MIN_TOUCH_TARGET,
    paddingHorizontal: 24,
  },
  retryText: { color: colors.text.onAccent, fontSize: fontSize.subheading, fontWeight: '700' },
});
