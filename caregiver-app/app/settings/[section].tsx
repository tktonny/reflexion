import { useLocalSearchParams, useRouter } from 'expo-router';
import React from 'react';
import { StyleSheet, Text } from 'react-native';
import { AppHeader, ScreenLayout, TertiaryButton } from '../../src/components/AppUI';
import { colors, fontFamily, fontSize, spacing } from '../../src/theme';

/** Fallback for old deep links only. It intentionally exposes no simulated settings controls. */
const LABELS: Record<string, string> = { routines: 'Routine Management', consent: 'Older-Adult Consent & Control', 'care-circle': 'Care Circle', privacy: 'Privacy & Data', research: 'Research participation', subscription: 'Subscription', accessibility: 'Language & accessibility', away: 'Mark as away', about: 'About Reflexion' };
export default function LegacySettingsFallback() { const router = useRouter(); const { section } = useLocalSearchParams<{ section?: string }>(); const title = LABELS[section || ''] || 'This setting'; return <ScreenLayout><AppHeader title="Settings" onBack={() => router.back()} /><Text accessibilityRole="header" style={s.title}>{title} is planned</Text><Text style={s.copy}>This Version 4 feature remains visible because it is part of the product architecture. It is not available in this pilot yet, because its secure server contract has not been implemented. No preferences or care data are changed here.</Text><TertiaryButton label="Back to Settings" onPress={() => router.replace('/(tabs)/settings')} /></ScreenLayout>; }
const s=StyleSheet.create({title:{color:colors.text.primary,fontFamily:fontFamily.display,fontSize:fontSize.title,fontWeight:'500',marginTop:spacing.xl},copy:{color:colors.text.secondary,flexShrink:1,fontSize:fontSize.body,lineHeight:22}});
