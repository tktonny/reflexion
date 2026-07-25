import { StyleSheet } from 'react-native';
import { colors, fontSize, radius, spacing } from '../../theme';

// The option chip look, shared by the picker rows in the settings list and the pickers inside the edit sheet.

// Option chips stay ~32pt tall on purpose — they sit in a wrapped row and growing them to 44 reflows the
// whole group. hitSlop lifts the tappable area to the 44pt floor instead.
export const PILL_HIT_SLOP = { bottom: 8, left: 8, right: 8, top: 8 };

export const pillStyles = StyleSheet.create({
  pickerOptions: { flexDirection: 'row', gap: spacing.sm, marginTop: 10, flexWrap: 'wrap' },
  pill: {
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: radius.pill,
    backgroundColor: colors.surface.muted,
    borderWidth: 1,
    borderColor: colors.border.default,
  },
  pillActive: { backgroundColor: colors.accent, borderColor: colors.accent },
  pillText: { fontSize: fontSize.body, color: colors.text.secondary },
  pillTextActive: { color: colors.text.onAccent, fontWeight: '600' },
});
