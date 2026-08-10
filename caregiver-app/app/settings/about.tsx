import * as Linking from 'expo-linking';
import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { AppHeader, SecondaryButton, ScreenLayout } from '../../src/components/AppUI';
import { BrandLockup } from '../../src/components/BrandLockup';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';

const APP_VERSION = '1.0.7';

export default function AboutReflexionScreen() {
  const router = useRouter();
  const [error, setError] = useState('');

  const openLegal = (url: string) => {
    setError('');
    void Linking.openURL(url).catch(() => setError('We could not open that page. Please try again or contact support.'));
  };

  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <AppHeader title="About Reflexion" onBack={() => router.back()} />
      <BrandLockup />
      <Text accessibilityRole="header" style={styles.title}>About Reflexion</Text>
      <Text style={styles.copy}>Reflexion helps families stay connected through everyday conversations, routines and family messages.</Text>

      <View style={styles.brandCard}>
        <Text style={styles.tagline}>Care. Connected.</Text>
        <Text style={styles.version}>App version {APP_VERSION}</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Our story</Text>
        <Text style={styles.copy}>Reflexion was created to make it easier for families to stay close in the rhythm of everyday life, with a calm and familiar way to connect.</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Safety & privacy</Text>
        <Text style={styles.copy}>Your loved one’s choice matters. Review consent, privacy choices and optional research participation any time in Settings.</Text>
      </View>

      <View style={styles.links}>
        <SecondaryButton label="Terms of Service" onPress={() => openLegal('https://reflexion.care/terms')} />
        <SecondaryButton label="Privacy Policy" onPress={() => openLegal('https://reflexion.care/privacy')} />
        <SecondaryButton label="Contact us" onPress={() => router.push('/settings/contact-support')} />
      </View>
      {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
    </ScreenLayout>
  );
}

const styles = StyleSheet.create({
  content: { alignItems: 'center', gap: spacing.xl, minWidth: 0 },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg, minWidth: 0, textAlign: 'center' },
  copy: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22, minWidth: 0, textAlign: 'center' },
  brandCard: { alignSelf: 'stretch', backgroundColor: '#F2F8F6', borderColor: '#D4E6DF', borderRadius: radius.xl, borderWidth: 1, gap: spacing.sm, padding: spacing.lg },
  tagline: { color: colors.accent, fontSize: fontSize.heading, fontWeight: '700', lineHeight: 26, textAlign: 'center' },
  version: { color: colors.text.secondary, fontSize: fontSize.caption, lineHeight: 18, textAlign: 'center' },
  section: { alignSelf: 'stretch', gap: spacing.sm, minWidth: 0 },
  sectionTitle: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 22, minWidth: 0 },
  links: { alignSelf: 'stretch', gap: spacing.md, minWidth: 0 },
  error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22, minWidth: 0, textAlign: 'center' },
});
