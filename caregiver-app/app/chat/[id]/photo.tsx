import * as ImagePicker from 'expo-image-picker';
import { useLocalSearchParams, usePathname, useRouter } from 'expo-router';
import React, { useMemo, useState } from 'react';
import { Feather } from '@expo/vector-icons';
import { Image, StyleSheet, Text, View } from 'react-native';

import { PrimaryButton, ScreenLayout } from '../../../src/components/AppUI';
import { Field } from '../../../src/components/Field';
import { ChatBotanical, ChatBrandLockup, ChatRecipientCard, ChatSurfaceCard } from '../../../src/components/ChatVisuals';
import { MessageTypePicker } from '../../../src/components/MessageTypePicker';
import { MotionPressable } from '../../../src/components/Motion';
import { colors, fontFamily, fontSize, radius, spacing } from '../../../src/theme';

export default function PhotoMessageComposerScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ id: string | string[] }>();
  const pathname = usePathname();
  const routeId = pathname.split('/').filter(Boolean).find((segment, index, segments) => segments[index - 1] === 'chat');
  const id = (Array.isArray(params.id) ? params.id[0] : params.id) || routeId;
  const [uri, setUri] = useState('');
  const [caption, setCaption] = useState('');
  const [error, setError] = useState('');
  const recipient = useMemo(() => id?.toLowerCase().includes('margaret') ? 'Mum Mary' : 'Loved one', [id]);
  const choose = async () => {
    setError('');
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) { setError('Photo access is needed to choose a photo.'); return; }
    const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 0.8 });
    if (!result.canceled) setUri(result.assets[0]?.uri || '');
  };
  const explainUnavailable = () => setError('Photo messages are not available for this Mirror yet. You can choose a photo here, then use a text message instead.');
  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <View style={styles.hero}><ChatBotanical /><ChatBrandLockup /><Text accessibilityRole="header" style={styles.title}>New photo message</Text><Text style={styles.subtitle}>CHAT-04 · Photo message composer</Text></View>
      <ChatRecipientCard name={recipient} onEdit={() => router.back()} />
      <ChatSurfaceCard style={styles.photoCard}><Text style={styles.sectionLabel}>Photo</Text>{uri ? <Image accessibilityLabel="Selected photo preview" source={{ uri }} style={styles.preview} /> : <MotionPressable accessibilityLabel="Choose a photo" accessibilityRole="button" onPress={() => void choose()} style={styles.placeholder}><Feather color={colors.accent} name="image" size={38} /><Text style={styles.placeholderTitle}>Choose a photo to preview</Text><Text style={styles.placeholderCopy}>Your photo stays on this device until a message is supported.</Text></MotionPressable>}<MotionPressable accessibilityLabel={uri ? 'Choose a different photo' : 'Choose photo'} accessibilityRole="button" onPress={() => void choose()} style={styles.chooseButton}><Text style={styles.chooseText}>{uri ? 'Choose a different photo' : 'Choose photo'}</Text></MotionPressable></ChatSurfaceCard>
      <ChatSurfaceCard style={styles.captionCard}><Field label="Add a caption (recommended)" multiline maxLength={500} onChangeText={setCaption} placeholder="Write a short caption" style={styles.captionInput} value={caption} /><Text style={styles.counter}>{caption.length}/500</Text></ChatSurfaceCard>
      <ChatSurfaceCard style={styles.scheduleCard}><Text style={styles.sectionLabel}>Send options</Text><ScheduleOption label="Send now" selected /><ScheduleOption label="Schedule" detail="Choose a date and time" showChevron /></ChatSurfaceCard>
      {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
      <PrimaryButton disabled label="Review message" onPress={explainUnavailable} />
      <MotionPressable accessibilityLabel="Use text message instead" accessibilityRole="button" onPress={() => router.replace(`/chat/${id}/compose`)} style={styles.textLink}><Text style={styles.textLinkLabel}>Use text message instead</Text></MotionPressable>
      <View style={styles.hiddenTypeControl}><MessageTypePicker selected="photo" onSelect={(type) => { if (type === 'text') router.push(`/chat/${id}/compose`); if (type === 'voice') router.push(`/chat/${id}/voice`); }} /></View>
    </ScreenLayout>
  );
}

function ScheduleOption({ label, detail, selected = false, showChevron = false }: { label: string; detail?: string; selected?: boolean; showChevron?: boolean }) {
  return <View style={styles.scheduleOption}><View style={[styles.radio, selected && styles.radioSelected]}>{selected ? <View style={styles.radioInner} /> : null}</View><View style={styles.optionCopy}><Text style={styles.optionLabel}>{label}</Text>{detail ? <Text style={styles.optionDetail}>{detail}</Text> : null}</View>{showChevron ? <Text accessibilityElementsHidden style={styles.chevron}>›</Text> : null}</View>;
}

const styles = StyleSheet.create({
  content: { gap: spacing.md, minWidth: 0, paddingTop: spacing.xs },
  hero: { minHeight: 176, minWidth: 0, position: 'relative', zIndex: 1 },
  title: { color: colors.text.primary, fontFamily: fontFamily.ui, fontSize: fontSize.display, fontWeight: '700', lineHeight: 42, marginTop: spacing.sm, minWidth: 0 },
  subtitle: { color: '#405A73', fontSize: fontSize.bodyLarge, lineHeight: 24, marginTop: spacing.sm, minWidth: 0 },
  photoCard: { gap: spacing.md, padding: spacing.lg },
  sectionLabel: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '600', lineHeight: 24 },
  placeholder: { alignItems: 'center', backgroundColor: '#F8F8F5', borderColor: colors.border.default, borderRadius: radius.lg, borderWidth: 1, gap: spacing.sm, justifyContent: 'center', minHeight: 260, padding: spacing.xl },
  placeholderTitle: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '600', textAlign: 'center' },
  placeholderCopy: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 21, textAlign: 'center' },
  preview: { borderRadius: radius.lg, height: 260, width: '100%' },
  chooseButton: { alignItems: 'center', borderColor: colors.accent, borderRadius: radius.pill, borderWidth: 1, minHeight: 44, justifyContent: 'center', paddingHorizontal: spacing.lg },
  chooseText: { color: colors.accent, fontSize: fontSize.body, fontWeight: '700' },
  captionCard: { gap: spacing.xs, padding: spacing.lg },
  captionInput: { borderRadius: radius.xl, fontSize: fontSize.bodyLarge, lineHeight: 24, minHeight: 118, paddingHorizontal: spacing.lg, paddingTop: spacing.lg },
  counter: { alignSelf: 'flex-end', color: colors.text.secondary, fontSize: fontSize.caption, marginRight: spacing.sm },
  scheduleCard: { gap: spacing.sm, padding: spacing.lg },
  scheduleOption: { alignItems: 'center', borderBottomColor: colors.border.subtle, borderBottomWidth: 1, flexDirection: 'row', gap: spacing.md, minHeight: 58, minWidth: 0, paddingVertical: spacing.sm },
  radio: { alignItems: 'center', borderColor: colors.text.secondary, borderRadius: radius.pill, borderWidth: 2, height: 25, justifyContent: 'center', width: 25 },
  radioSelected: { borderColor: colors.accent },
  radioInner: { backgroundColor: colors.accent, borderRadius: radius.pill, height: 13, width: 13 },
  optionCopy: { flex: 1, flexShrink: 1, minWidth: 0 },
  optionLabel: { color: colors.text.primary, fontSize: fontSize.bodyLarge, lineHeight: 23 },
  optionDetail: { color: colors.text.secondary, fontSize: fontSize.body, lineHeight: 21 },
  chevron: { color: colors.text.secondary, fontSize: 33, lineHeight: 33, paddingHorizontal: spacing.xs },
  error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22 },
  textLink: { alignItems: 'center', minHeight: 44, justifyContent: 'center' },
  textLinkLabel: { color: colors.accent, fontSize: fontSize.bodyLarge, fontWeight: '700' },
  hiddenTypeControl: { height: 1, opacity: 0.001, overflow: 'hidden' },
});
