import React, { useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { EmptyState } from '../src/components/ScreenState';
import { FAQ_ITEMS } from '../src/data/faqContent';
import { colors, spacing, radius, fontSize, scaleSize } from '../src/theme';

export default function FAQScreen() {
  const router = useRouter();
  const [open, setOpen] = useState<number | null>(0);

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.intro}>
          Everything you need to know about Reflexion and Aria.
        </Text>

        {FAQ_ITEMS.length === 0 ? (
          // The questions ship inside the app, so an empty list can only be a bad build. Saying so beats
          // leaving a caregiver on a screen that looks like it is still loading and never finishes.
          <EmptyState
            icon="help-circle"
            title="The guide is not showing right now"
            message="You can still reach us below and we will answer you directly."
          />
        ) : null}

        {FAQ_ITEMS.map((item, i) => {
          const expanded = open === i;
          return (
            <TouchableOpacity
              key={i}
              // One label for the whole row. Without it a screen reader announces the question, the
              // chevron glyph and the answer as three separate fragments.
              accessibilityLabel={expanded ? `${item.q}. ${item.a}` : item.q}
              accessibilityRole="button"
              accessibilityState={{ expanded }}
              style={[styles.card, expanded && styles.cardOpen]}
              onPress={() => setOpen(expanded ? null : i)}
              activeOpacity={0.8}
            >
              <View style={styles.question}>
                <Text style={styles.questionText}>{item.q}</Text>
                <Text
                  accessibilityElementsHidden
                  importantForAccessibility="no"
                  style={styles.chevron}
                >
                  {expanded ? '▲' : '▼'}
                </Text>
              </View>
              {expanded ? (
                <Text style={styles.answerText}>{item.a}</Text>
              ) : null}
            </TouchableOpacity>
          );
        })}

        <TouchableOpacity
          accessibilityLabel="Still have questions? Chat with the Reflexion team."
          accessibilityRole="button"
          activeOpacity={0.82}
          onPress={() => router.push('/chatbot')}
          style={styles.bottomCard}
        >
          <Text style={styles.bottomTitle}>Still have questions?</Text>
          <Text style={styles.bottomText}>Click here to chat with the Reflexion team.</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

// Brought onto src/theme.ts. This screen and the support chat were the last two on an older blue palette
// that predated the theme, so they read as a different product from every other screen — and its muted grey
// (#AAA on white, 2.32:1) failed WCAG AA outright.
const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.surface.page },
  content: { padding: spacing.xl, paddingBottom: scaleSize(48) },
  intro: { fontSize: fontSize.subheading, color: colors.text.secondary, marginBottom: spacing.xl, lineHeight: scaleSize(22) },
  card: {
    backgroundColor: colors.surface.card, borderRadius: radius.xl, padding: spacing.lg, marginBottom: scaleSize(10),
    borderWidth: 1.5, borderColor: colors.border.default,
    shadowColor: colors.shadow, shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.04, shadowRadius: 3, elevation: 1,
  },
  cardOpen: { borderColor: colors.accent },
  question: { flexDirection: 'row', alignItems: 'flex-start', justifyContent: 'space-between', gap: spacing.sm },
  questionText: { fontSize: fontSize.subheading, fontWeight: '700', color: colors.text.primary, flex: 1, lineHeight: scaleSize(22) },
  chevron: { fontSize: fontSize.caption, color: colors.accent, marginTop: spacing.xs },
  answerText: { fontSize: fontSize.bodyLarge, color: colors.text.secondary, marginTop: spacing.md, lineHeight: scaleSize(22) },
  bottomCard: {
    backgroundColor: colors.surface.muted, borderRadius: radius.xl, padding: spacing.xl, marginTop: spacing.sm, alignItems: 'center',
  },
  bottomTitle: { fontSize: scaleSize(16), fontWeight: '700', color: colors.accent, marginBottom: spacing.xs },
  bottomText: { fontSize: fontSize.bodyLarge, color: colors.text.secondary, textAlign: 'center', lineHeight: scaleSize(20) },
});
