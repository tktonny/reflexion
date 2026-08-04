import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { AppHeader, ChoiceCard, PrimaryButton, ScreenLayout } from '../../src/components/AppUI';
import { colors, fontFamily, fontSize, spacing } from '../../src/theme';
const FILTERS = ['Person', 'Date range', 'Conversations', 'Routines', 'Chat', 'Technical', 'Caregiver actions'];
export default function ActivityFilterScreen() { const router = useRouter(); const [selected, setSelected] = useState(0); return <ScreenLayout><AppHeader title="Activity filter" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Filter activity</Text><Text style={styles.subtitle}>Choose the updates you would like to see.</Text><View style={styles.cards}>{FILTERS.map((filter, index) => <ChoiceCard key={filter} icon="sliders" title={filter} description={index === 0 ? 'Mum' : 'All updates'} selected={selected === index} onPress={() => setSelected(index)} />)}</View><PrimaryButton label="Show activity" onPress={() => router.back()} /></ScreenLayout>; }
const styles = StyleSheet.create({ title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', marginTop: spacing.xl }, subtitle: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, flexShrink: 1 }, cards: { gap: spacing.md, marginBottom: spacing.lg, marginTop: spacing.md } });
