import * as ImagePicker from 'expo-image-picker';
import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useState } from 'react';
import { Image, StyleSheet, Text, View } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, SecondaryButton } from '../../../src/components/AppUI';
import { Field } from '../../../src/components/Field';
import { MessageTypePicker } from '../../../src/components/MessageTypePicker';
import { colors, fontFamily, fontSize, radius, spacing } from '../../../src/theme';

export default function PhotoMessageComposerScreen() {
  const router = useRouter();
  const { id } = useLocalSearchParams<{ id: string }>();
  const [uri, setUri] = useState('');
  const [caption, setCaption] = useState('');
  const [error, setError] = useState('');
  const explainUnavailable = () => setError('Photo messages are not available for this Mirror yet. You can send a text message instead.');

  const choose = async () => {
    setError('');
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) { setError('Photo access is needed to choose a photo.'); return; }
    const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 0.8 });
    if (!result.canceled) setUri(result.assets[0]?.uri || '');
  };

  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Photo message" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Photo message</Text><Text style={styles.subtitle}>Choose a photo and add a caption that can be read aloud on the Mirror.</Text><MessageTypePicker selected="photo" onSelect={(type) => { if (type === 'text') router.push(`/chat/${id}/compose`); if (type === 'voice') router.push(`/chat/${id}/voice`); }} /><SecondaryButton label={uri ? 'Choose a different photo' : 'Choose photo'} onPress={() => void choose()} />{uri ? <Image accessibilityLabel="Selected photo preview" source={{ uri }} style={styles.preview} /> : <View style={styles.placeholder}><Text style={styles.placeholderTitle}>No photo selected</Text><Text style={styles.placeholderCopy}>The photo will stay on this device until it is ready to share.</Text></View>}<Field label="Caption" multiline helperText="A caption helps the Mirror explain the photo." onChangeText={setCaption} placeholder="Write a short caption" value={caption} />{error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}<View style={styles.notice}><Text style={styles.noticeTitle}>Photo messages are not available yet</Text><Text style={styles.noticeCopy}>You can choose a photo here, then use a text message instead.</Text></View><PrimaryButton disabled label="Photo messages unavailable" onPress={explainUnavailable} /><SecondaryButton label="Use text message instead" onPress={() => router.replace(`/chat/${id}/compose`)} /></ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, subtitle: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 22 }, preview: { alignSelf: 'center', borderRadius: radius.xl, height: 220, width: '100%' }, placeholder: { alignItems: 'center', backgroundColor: colors.surface.muted, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.sm, justifyContent: 'center', minHeight: 180, padding: spacing.xl }, placeholderTitle: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '700' }, placeholderCopy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 21, textAlign: 'center' }, notice: { backgroundColor: colors.status.greyBg, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.sm, padding: spacing.lg }, noticeTitle: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '700' }, noticeCopy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 21 }, error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 21 } });
