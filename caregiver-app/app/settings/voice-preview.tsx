import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, SecondaryButton } from '../../src/components/AppUI';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';

export default function VoicePreviewScreen() {
  const router = useRouter();
  const [playing, setPlaying] = useState(false);
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Voice preview" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Voice preview</Text><Text style={styles.subtitle}>Review the words and selected settings before saving. The paired Mirror uses the saved language, voice, pace and volume for its next spoken prompt.</Text><View style={styles.settings}><Text style={styles.label}>Language</Text><Text style={styles.value}>English</Text><Text style={styles.label}>Voice</Text><Text style={styles.value}>Warm female</Text><Text style={styles.label}>Pace</Text><Text style={styles.value}>Normal</Text><Text style={styles.label}>Volume</Text><Text style={styles.value}>70%</Text></View><View style={styles.preview}><Text style={styles.previewLabel}>Text preview</Text><Text style={styles.sample}>“Hello, I’m here to help you through your day. I’ll remind you of what’s important and keep things simple.”</Text><Text style={styles.playing}>{playing ? 'This is the sample preview for the selected Mirror settings.' : 'Audio preview is not available on this device yet.'}</Text><SecondaryButton label={playing ? 'Hide text preview' : 'Show text preview'} onPress={() => setPlaying((value) => !value)} /></View><SecondaryButton label="Change settings" onPress={() => router.back()} /><PrimaryButton label="Continue" onPress={() => router.back()} /></ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36 }, subtitle: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 24 }, settings: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.sm, padding: spacing.lg }, label: { color: colors.text.secondary, fontSize: fontSize.caption, fontWeight: '700' }, value: { color: colors.text.primary, fontSize: fontSize.bodyLarge, marginBottom: spacing.sm }, preview: { backgroundColor: '#F1F7F2', borderColor: '#DCE9DD', borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, padding: spacing.xl }, previewLabel: { color: colors.accent, fontSize: fontSize.caption, fontWeight: '700' }, sample: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.heading, lineHeight: 28 }, playing: { color: colors.text.secondary, fontSize: fontSize.caption, lineHeight: 19 },
});
