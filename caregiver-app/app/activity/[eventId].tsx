import { Feather } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import React from 'react';
import { Image, Share, StyleSheet, Text, View } from 'react-native';

import { MotionPressable } from '../../src/components/Motion';
import { ScreenLayout, type IconName } from '../../src/components/AppUI';
import { cardShadow, colors, fontFamily, fontSize, radius, scaleSize, spacing } from '../../src/theme';
import { useTabBarClearance } from '../../src/lib/useTabBarClearance';

type DetailParams = {
  detail?: string | string[];
  eventId?: string | string[];
  patientId?: string | string[];
  person?: string | string[];
  time?: string | string[];
  title?: string | string[];
};

export default function ActivityDetailScreen() {
  const router = useRouter();
  const clearance = useTabBarClearance();
  const params = useLocalSearchParams<DetailParams>();
  const eventId = decodeParam(params.eventId);
  const patientId = decodeParam(params.patientId);
  const rawPerson = decodeParam(params.person);
  const rawTitle = decodeParam(params.title);
  const rawDetail = decodeParam(params.detail);
  const rawTime = decodeParam(params.time);
  const isConversation = eventId.startsWith('session-') || rawTitle.toLowerCase().includes('conversation');
  const personName = patientId === 'demo-margaret' || rawPerson.toLowerCase() === 'margaret' ? 'Mum' : rawPerson || 'Mum';
  const title = rawTitle || 'Conversation recorded';
  const subtitle = isConversation ? 'Morning check-in' : rawDetail || 'Activity record';
  const duration = isConversation ? rawDetail.replace(/^duration:\s*/i, '') || '12 min' : rawDetail || '—';
  const time = timeParts(rawTime);
  const sessionId = getSessionId(eventId, patientId);
  const portraitAvailable = personName === 'Mum' || patientId === 'demo-margaret';
  const summary = isConversation
    ? `You had a good morning check-in with ${personName}. She mentioned feeling well, enjoyed her breakfast, and is looking forward to your visit tomorrow.`
    : rawDetail || 'This event was recorded as part of the activity timeline.';

  const share = () => {
    void Share.share({
      message: `Reflexion activity: ${title} for ${personName}. ${rawDetail || subtitle}`,
      title: 'Reflexion activity',
    }).catch(() => undefined);
  };

  return (
    <ScreenLayout bottomInset={clearance} contentContainerStyle={styles.content}>
      <View pointerEvents="none" style={styles.botanicalLayer}>
        <Image
          accessibilityElementsHidden
          importantForAccessibility="no-hide-descendants"
          resizeMode="contain"
          source={require('../../assets/act02-botanical.png')}
          style={styles.botanical}
        />
      </View>

      <DetailHeader onBack={() => router.back()} />

      <View style={[styles.heroCard, styles.layer]}>
        <View style={styles.heroTop}>
          {portraitAvailable ? (
            <Image
              accessibilityLabel="Mum Mary"
              source={require('../../assets/act02-mum-mary.png')}
              style={styles.portrait}
            />
          ) : (
            <View accessible accessibilityLabel={personName} style={styles.portraitFallback}>
              <Text style={styles.portraitInitial}>{personName.slice(0, 1).toUpperCase()}</Text>
            </View>
          )}
          <View style={styles.heroInfo}>
            <Text style={styles.personName}>{personName}</Text>
            <Text style={styles.eventTitle}>{title}</Text>
            <Text style={styles.eventSubtitle}>{subtitle}</Text>
            <View style={styles.metaStack}>
              <MetaRow icon="calendar" value={time.dateLabel} />
              <MetaRow icon="clock" value={`${time.timeLabel}  •  Duration: ${duration}`} />
            </View>
          </View>
        </View>

        <View style={styles.rule} />
        <Text style={styles.cardSectionTitle}>Summary</Text>
        <Text style={styles.summaryCopy}>{summary}</Text>
      </View>

      <View style={[styles.sectionCard, styles.layer]}>
        <Text style={styles.cardSectionTitle}>Underlying facts</Text>
        <View style={styles.factGrid}>
          <Fact icon="clock" label="Duration" value={duration} style={styles.durationFact} />
          <View style={[styles.factItem, styles.topicsFact]}>
            <IconCircle icon="tag" />
            <View style={styles.factCopy}>
              <Text style={styles.factLabel}>Topics</Text>
              <View style={styles.topicRow}>
                <TopicPill label="Wellness" />
                <TopicPill label="Routine" />
                <TopicPill label="Mood" />
              </View>
            </View>
          </View>
        </View>
      </View>

      <View style={[styles.sectionCard, styles.layer]}>
        <Text style={styles.cardSectionTitle}>Provenance</Text>
        <View style={styles.factGrid}>
          <Fact icon="smartphone" label="Source device" value="Chloe’s iPhone 15 Pro" />
          <Fact icon="shield" label="Recorded locally" value="Stored securely on device" />
        </View>
      </View>

      <View style={[styles.relatedCard, styles.layer]}>
        <Text style={styles.cardSectionTitle}>Related</Text>
        <RelatedRow icon="activity" label="View Session Detail" onPress={sessionId && patientId ? () => router.push(`/loved-one/${patientId}/sessions/${sessionId}`) : undefined} />
        <RelatedRow icon="calendar" label="View related Routine" />
        <RelatedRow icon="message-circle" label="View related Message" />
        <RelatedRow icon="smartphone" label="View device events" last />
      </View>

      <MotionPressable accessibilityLabel="Share this event" accessibilityRole="button" feedback="button" haptic="selection" onPress={share} style={styles.shareButton}>
        <Feather color={colors.text.onAccent} name="upload" size={22} />
        <Text style={styles.shareText}>Share this event</Text>
      </MotionPressable>
    </ScreenLayout>
  );
}

function DetailHeader({ onBack }: { onBack: () => void }) {
  return (
    <View style={[styles.header, styles.layer]}>
      <MotionPressable accessibilityLabel="Back to Activity" accessibilityRole="button" feedback="button" onPress={onBack} style={styles.headerBack}>
      <Feather color={colors.text.primary} name="chevron-left" size={24} />
        <Text style={styles.headerBackText}>Activity</Text>
      </MotionPressable>
      <Text accessibilityRole="header" numberOfLines={1} style={styles.headerTitle}>Activity detail</Text>
      <View accessibilityElementsHidden importantForAccessibility="no-hide-descendants" style={styles.headerBalance} />
    </View>
  );
}

function MetaRow({ icon, value }: { icon: IconName; value: string }) {
  return <View style={styles.metaRow}><Feather color={colors.text.secondary} name={icon} size={20} /><Text style={styles.metaText} numberOfLines={1}>{value}</Text></View>;
}

function IconCircle({ icon }: { icon: IconName }) {
  return <View accessibilityElementsHidden importantForAccessibility="no-hide-descendants" style={styles.iconCircle}><Feather color={colors.accent} name={icon} size={21} /></View>;
}

function Fact({ icon, label, value, style }: { icon: IconName; label: string; value: string; style?: object }) {
  return <View style={[styles.factItem, style]}><IconCircle icon={icon} /><View style={styles.factCopy}><Text style={styles.factLabel}>{label}</Text><Text style={styles.factValue}>{value}</Text></View></View>;
}

function TopicPill({ label }: { label: string }) {
  return <View style={styles.topicPill}><Text style={styles.topicText}>{label}</Text></View>;
}

function RelatedRow({ icon, label, last = false, onPress }: { icon: IconName; label: string; last?: boolean; onPress?: () => void }) {
  const content = <><View style={styles.relatedIcon}><Feather color={colors.accent} name={icon} size={19} /></View><Text style={styles.relatedLabel}>{label}</Text><Feather color={colors.text.primary} name="chevron-right" size={22} /></>;
  const rowStyle = [styles.relatedRow, !last && styles.relatedRowBorder];
  if (!onPress) return <View accessible accessibilityLabel={label} style={rowStyle}>{content}</View>;
  return <MotionPressable accessibilityLabel={label} accessibilityRole="button" feedback="none" onPress={onPress} style={rowStyle}>{content}</MotionPressable>;
}

function getSessionId(eventId: string, patientId: string) {
  if (!eventId.startsWith('session-')) return '';
  const prefix = patientId ? `session-${patientId}-` : 'session-';
  return eventId.startsWith(prefix) ? eventId.slice(prefix.length) : eventId.slice('session-'.length);
}

function timeParts(value: string) {
  const fallback = { dateLabel: 'Tuesday, 20 May 2025', timeLabel: '8:15 AM' };
  if (!value || value === 'Time unavailable') return fallback;
  const match = value.match(/^(.+?)\s+at\s+(.+)$/i);
  const rawDate = match?.[1] || value;
  const date = new Date(rawDate);
  const dateLabel = Number.isNaN(date.getTime())
    ? rawDate
    : new Intl.DateTimeFormat('en-SG', { day: 'numeric', month: 'long', weekday: 'long', year: 'numeric' }).format(date);
  return { dateLabel, timeLabel: match?.[2] || fallback.timeLabel };
}

function decodeParam(value: string | string[] | undefined) {
  const raw = Array.isArray(value) ? value[0] || '' : value || '';
  try { return decodeURIComponent(raw); } catch { return raw; }
}

const styles = StyleSheet.create({
  content: { gap: spacing.sm, minWidth: 0, paddingTop: 0, position: 'relative' },
  layer: { position: 'relative', zIndex: 1 },
  botanicalLayer: { height: scaleSize(154), position: 'absolute', right: -spacing.sm, top: spacing.sm, width: scaleSize(120), zIndex: 0 },
  botanical: { height: '100%', width: '100%' },
  header: { alignItems: 'center', flexDirection: 'row', minHeight: 44, minWidth: 0 },
  headerBack: { alignItems: 'center', flexDirection: 'row', gap: 1, minHeight: 44, minWidth: 88, paddingRight: spacing.sm },
  headerBackText: { color: colors.text.primary, fontFamily: fontFamily.ui, fontSize: fontSize.bodyLarge, lineHeight: 22 },
  headerTitle: { color: colors.text.primary, flex: 1, fontFamily: fontFamily.ui, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 22, minWidth: 0, textAlign: 'center' },
  headerBalance: { minWidth: 96 },
  heroCard: { ...cardShadow, backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.sm, minWidth: 0, padding: spacing.md },
  heroTop: { alignItems: 'flex-start', flexDirection: 'row', gap: spacing.sm, minWidth: 0 },
  portrait: { borderRadius: radius.pill, flexShrink: 0, height: scaleSize(84), width: scaleSize(84) },
  portraitFallback: { alignItems: 'center', backgroundColor: '#E7F0EA', borderRadius: radius.pill, flexShrink: 0, height: scaleSize(84), justifyContent: 'center', width: scaleSize(84) },
  portraitInitial: { color: colors.accent, fontFamily: fontFamily.display, fontSize: scaleSize(30), fontWeight: '500' },
  heroInfo: { flex: 1, flexShrink: 1, minWidth: 0 },
  personName: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.heading, fontStyle: 'italic', lineHeight: 25, minWidth: 0 },
  eventTitle: { color: colors.text.primary, fontFamily: fontFamily.ui, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 21, marginTop: 1, minWidth: 0 },
  eventSubtitle: { color: colors.text.secondary, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 18, marginTop: 1, minWidth: 0 },
  metaStack: { gap: spacing.xs, marginTop: spacing.sm, minWidth: 0 },
  metaRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.xs, minWidth: 0 },
  metaText: { color: colors.text.primary, flex: 1, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 18, minWidth: 0 },
  rule: { borderTopColor: colors.border.subtle, borderTopWidth: 1, marginTop: spacing.xs },
  cardSectionTitle: { color: colors.text.primary, fontFamily: fontFamily.ui, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 22, minWidth: 0 },
  summaryCopy: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.caption, lineHeight: 17, minWidth: 0 },
  sectionCard: { ...cardShadow, backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.sm, minWidth: 0, padding: spacing.md },
  factGrid: { flexDirection: 'row', gap: spacing.sm, minWidth: 0 },
  factItem: { alignItems: 'flex-start', flex: 1, flexDirection: 'row', gap: spacing.xs, minWidth: 0 },
  durationFact: { flex: 0.78 },
  topicsFact: { flex: 1.32 },
  iconCircle: { alignItems: 'center', backgroundColor: '#E7F1EC', borderRadius: radius.pill, flexShrink: 0, height: scaleSize(40), justifyContent: 'center', width: scaleSize(40) },
  factCopy: { flex: 1, flexShrink: 1, minWidth: 0 },
  factLabel: { color: colors.text.primary, fontFamily: fontFamily.ui, fontSize: fontSize.caption, lineHeight: 16, minWidth: 0 },
  factValue: { color: colors.text.primary, fontFamily: fontFamily.ui, fontSize: fontSize.caption, lineHeight: 16, marginTop: 2, minWidth: 0 },
  topicRow: { flexDirection: 'row', flexWrap: 'nowrap', gap: spacing.xs, marginTop: spacing.xs, minWidth: 0 },
  topicPill: { alignItems: 'center', backgroundColor: '#E7F1EC', borderRadius: radius.pill, minHeight: 28, paddingHorizontal: spacing.xs, paddingVertical: 3 },
  topicText: { color: '#225F5D', fontFamily: fontFamily.ui, fontSize: fontSize.caption, lineHeight: 16 },
  relatedCard: { ...cardShadow, backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, minWidth: 0, paddingHorizontal: spacing.md, paddingTop: spacing.md },
  relatedRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, minHeight: 40, minWidth: 0, paddingVertical: spacing.xs },
  relatedRowBorder: { borderBottomColor: colors.border.subtle, borderBottomWidth: 1 },
  relatedIcon: { alignItems: 'center', backgroundColor: '#E7F1EC', borderRadius: radius.pill, flexShrink: 0, height: scaleSize(34), justifyContent: 'center', width: scaleSize(34) },
  relatedLabel: { color: colors.text.primary, flex: 1, flexShrink: 1, fontFamily: fontFamily.ui, fontSize: fontSize.body, lineHeight: 18, minWidth: 0 },
  shareButton: { alignItems: 'center', alignSelf: 'stretch', backgroundColor: colors.accent, borderRadius: radius.xl, flexDirection: 'row', gap: spacing.sm, justifyContent: 'center', minHeight: 48, minWidth: 0, paddingHorizontal: spacing.md },
  shareText: { color: colors.text.onAccent, fontFamily: fontFamily.ui, fontSize: fontSize.bodyLarge, fontWeight: '600', lineHeight: 22 },
});
