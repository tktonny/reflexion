import { Feather } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { AppHeader, SecondaryButton, ScreenLayout } from '../../src/components/AppUI';
import { MotionDisclosure, MotionPressable } from '../../src/components/Motion';
import { FAQ_GROUPS } from '../../src/data/faqContent';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';

export default function Help() {
  const router = useRouter();
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());

  const toggle = (id: string) => {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <AppHeader title="Help & Support" onBack={() => router.back()} />
      <View style={styles.intro}>
        <Text accessibilityRole="header" style={styles.title}>Help & Support</Text>
        <Text style={styles.copy}>Find quick answers about setup, the Mirror, routines, messages and your account.</Text>
      </View>

      <View style={styles.groups}>
        {FAQ_GROUPS.map((group) => (
          <View key={group.category} style={styles.group}>
            <Text style={styles.category}>{group.category}</Text>
            <View style={styles.items}>
              {group.items.map((item) => {
                const isOpen = expanded.has(item.id);
                return (
                  <View key={item.id} style={styles.item}>
                    <MotionPressable
                      accessibilityRole="button"
                      accessibilityState={{ expanded: isOpen }}
                      feedback="card"
                      haptic="selection"
                      onPress={() => toggle(item.id)}
                      style={styles.question}
                    >
                      <Text style={styles.questionText}>{item.question}</Text>
                      <Feather color={colors.accent} name={isOpen ? 'minus' : 'plus'} size={20} />
                    </MotionPressable>
                    <MotionDisclosure open={isOpen}>
                      <Text style={styles.answer}>{item.answer}</Text>
                    </MotionDisclosure>
                  </View>
                );
              })}
            </View>
          </View>
        ))}
      </View>

      <View style={styles.supportCard}>
        <Feather color={colors.accent} name="message-circle" size={24} />
        <View style={styles.supportCopy}>
          <Text style={styles.supportTitle}>Still need help?</Text>
          <Text style={styles.copy}>Contact the Reflexion support team if the answers above do not solve the issue.</Text>
        </View>
        <SecondaryButton label="Contact support" onPress={() => router.push('/settings/contact-support')} />
      </View>
    </ScreenLayout>
  );
}

const styles = StyleSheet.create({
  content: { gap: spacing.xl, minWidth: 0 },
  intro: { gap: spacing.sm, minWidth: 0 },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg, minWidth: 0 },
  copy: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22, minWidth: 0 },
  groups: { gap: spacing.lg, minWidth: 0 },
  group: { gap: spacing.sm, minWidth: 0 },
  category: { color: colors.accent, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 22, minWidth: 0 },
  items: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, minWidth: 0, overflow: 'hidden' },
  item: { borderBottomColor: colors.border.subtle, borderBottomWidth: 1, minWidth: 0 },
  question: { alignItems: 'center', flexDirection: 'row', gap: spacing.md, minHeight: 60, minWidth: 0, paddingHorizontal: spacing.lg, paddingVertical: spacing.md },
  questionText: { color: colors.text.primary, flex: 1, flexShrink: 1, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 22, minWidth: 0 },
  answer: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22, minWidth: 0, paddingBottom: spacing.lg, paddingHorizontal: spacing.lg, paddingTop: 0 },
  supportCard: { alignItems: 'flex-start', backgroundColor: '#F2F8F6', borderColor: '#D4E6DF', borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, minWidth: 0, padding: spacing.lg },
  supportCopy: { gap: spacing.xs, minWidth: 0 },
  supportTitle: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 22, minWidth: 0 },
});
