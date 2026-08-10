import { Feather } from '@expo/vector-icons';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ActivityIndicator, Animated, Dimensions, PanResponder, ScrollView, Share, StyleSheet, Text, useWindowDimensions, View } from 'react-native';

import { AppHeader, PrimaryButton, ProvenanceSection, ScreenLayout, SelectionButton, SurfaceCard, TertiaryButton, type IconName } from '../components/AppUI';
import { LovedAvatar, LovedBotanical, LovedBrandLockup, LovedDot, LovedIconCircle } from '../components/LovedOneVisuals';
import { MotionFadeIn, MotionPressable } from '../components/Motion';
import { getSessionDayV1, generateSessionSummaryV1, getSessionTrendV1, listFamilyMessagesV1, listReminderOccurrencesV1, listSessionDaysV1, listSessionsV1, loadCaregiverHome, type CaregiverHomePatient, type V1ReminderOccurrence, type V1SessionDetail, type V1SessionFeed, type V1TrendDay } from '../lib/v1Caregiver';
import { buildMonthCalendar, monthKey, monthLabel, shiftMonth, WEEKDAY_LABELS, type CalendarCell } from '../lib/monthCalendar';
import { cardShadow, colors, fontFamily, fontSize, radius, spacing, typography } from '../theme';

function formatDuration(seconds: number) {
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  return minutes ? `${minutes} min${remainder ? ` ${remainder} sec` : ''}` : `${remainder} sec`;
}

function displayDate(value: string | null | undefined, withTime = false) {
  if (!value) return 'Time unavailable';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'Time unavailable';
  return new Intl.DateTimeFormat('en-SG', withTime ? { dateStyle: 'medium', timeStyle: 'short' } : { dateStyle: 'medium' }).format(date);
}

function isoDate(date: Date) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}

function rangeDates(days: number) {
  const end = new Date();
  const start = new Date(end.getTime() - (days - 1) * 86_400_000);
  return { from: isoDate(start), to: isoDate(new Date(end.getTime() + 86_400_000)) };
}

function patientFromHome(home: Awaited<ReturnType<typeof loadCaregiverHome>>, id: string | undefined) {
  return home.patients.find((item) => item.patientId === id) || null;
}

export function WeeklySummaryScreen() {
  const router = useRouter();
  const { id } = useLocalSearchParams<{ id: string }>();
  const [person, setPerson] = useState<CaregiverHomePatient | null>(null);
  const [sessions, setSessions] = useState<V1SessionDetail[]>([]);
  const [trend, setTrend] = useState<V1TrendDay[]>([]);
  const [routines, setRoutines] = useState<V1ReminderOccurrence[]>([]);
  const [messageCount, setMessageCount] = useState(0);
  const [summary, setSummary] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    if (!id) return;
    setLoading(true); setError('');
    try {
      const home = await loadCaregiverHome();
      const nextPerson = patientFromHome(home, id);
      setPerson(nextPerson);
      const { from, to } = rangeDates(7);
      const [feed, nextTrend, occurrences, messages] = await Promise.all([
        listSessionsV1(id, { limit: 50 }),
        getSessionTrendV1(id, 7),
        listReminderOccurrencesV1(id, from, to).catch(() => []),
        listFamilyMessagesV1(id),
      ]);
      setSessions(feed.sessions.filter((session) => Boolean(session.createdAt && new Date(session.createdAt).getTime() >= Date.now() - 7 * 86_400_000)));
      setTrend(nextTrend);
      setRoutines(occurrences);
      setMessageCount(messages.length);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The weekly summary could not be loaded.');
    } finally { setLoading(false); }
  }, [id]);
  useFocusEffect(useCallback(() => { void refresh(); }, [refresh]));

  const createSummary = async () => {
    if (!id) return;
    setSummaryLoading(true); setError('');
    try {
      const result = await generateSessionSummaryV1(id);
      setSummary(result.summary || 'No transcript was available for this period.');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'A summary could not be generated.');
    } finally { setSummaryLoading(false); }
  };

  return <ScreenLayout contentContainerStyle={styles.content}>
    <AppHeader onBack={() => router.back()} />
    <View style={styles.pageHero}><View style={styles.pageHeroCopy}><Text accessibilityRole="header" style={styles.title}>Weekly Summary</Text><Text style={styles.personName}>{person?.displayName || 'Loved one'}</Text><Text style={styles.subtitle}>Last 7 days</Text></View><LovedBotanical style={styles.botanical} /></View>
    {loading ? <ActivityIndicator color={colors.accent} /> : null}
    {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
    {!loading && !error ? <>
      <View style={styles.metricGrid}><Metric icon="message-circle" label="Sessions" value={String(sessions.length)} detail="This week" /><Metric icon="clock" label="Conversation time" value={formatDuration(sessions.reduce((sum, session) => sum + session.duration, 0))} detail="Across sessions" /><Metric icon="calendar" label="Days with interaction" value={`${trend.filter((day) => !day.missed).length} / 7`} detail="Recorded days" /><Metric icon="clipboard" label="Routines" value={String(routines.length)} detail="Recorded occurrences" /><Metric icon="mail" label="Messages" value={String(messageCount)} detail="In the delivery record" /><Metric icon="star" label="Notable events" value={String(messageCount ? Math.min(messageCount, 2) : 0)} detail="See below" /></View>
      <SurfaceCard style={styles.card}><View style={styles.cardHeading}><LovedIconCircle icon="star" size={48} /><View style={styles.cardHeadingCopy}><Text style={styles.cardTitle}>Notable events</Text><Text style={styles.body}>{messageCount ? `There ${messageCount === 1 ? 'is' : 'are'} ${messageCount} family message${messageCount === 1 ? '' : 's'} in the delivery record.` : 'No notable events were recorded for this period.'}</Text></View></View></SurfaceCard>
      <SurfaceCard style={styles.card}><View style={styles.cardHeading}><LovedIconCircle icon="feather" size={48} /><View style={styles.cardHeadingCopy}><Text style={styles.cardTitle}>Recommended action</Text><Text style={styles.body}>Share more about meaningful family stories when you next chat.</Text></View><PrimaryButton label="Try a prompt" onPress={() => router.push(`/loved-one/${id}/sessions`)} /></View></SurfaceCard>
      <SurfaceCard style={styles.card}><View style={styles.cardHeading}><LovedIconCircle icon="info" size={48} /><View style={styles.cardHeadingCopy}><Text style={styles.cardTitle}>Limitations</Text><Text style={styles.body}>AI may not always capture tone or context accurately. Please review conversations and use your judgement.</Text></View><Feather color={colors.text.primary} name="chevron-right" size={20} /></View></SurfaceCard>
      {summary ? <View style={styles.card}><Text style={styles.cardTitle}>Session summary</Text><Text style={styles.body}>{summary}</Text><Text style={styles.note}>Generated from available transcript data on request.</Text></View> : <PrimaryButton disabled={summaryLoading} label={summaryLoading ? 'Preparing summary…' : 'Prepare session summary'} onPress={() => void createSummary()} />}
      <TertiaryButton label="View sessions" onPress={() => router.push(`/loved-one/${id}/sessions`)} />
    </> : null}
  </ScreenLayout>;
}

function Metric({ icon, label, value, detail }: { icon: IconName; label: string; value: string; detail: string }) {
  return <View style={styles.metric}><LovedIconCircle icon={icon} size={48} /><View style={styles.metricCopy}><Text style={styles.metricLabel}>{label}</Text><Text style={styles.metricValue}>{value}</Text><Text style={styles.metricDetail}>{detail}</Text></View></View>;
}

export function TrendsScreen() {
  const router = useRouter();
  const { id } = useLocalSearchParams<{ id: string }>();
  const [person, setPerson] = useState<CaregiverHomePatient | null>(null);
  const [range, setRange] = useState<'7' | '30' | '90'>('7');
  const [trend, setTrend] = useState<V1TrendDay[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    if (!id) return;
    setLoading(true); setError('');
    try {
      const home = await loadCaregiverHome();
      setPerson(patientFromHome(home, id));
      setTrend(await getSessionTrendV1(id, Number(range) as 7 | 30 | 90));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Trends could not be loaded.');
    } finally { setLoading(false); }
  }, [id, range]);
  useFocusEffect(useCallback(() => { void refresh(); }, [refresh]));

  return <ScreenLayout contentContainerStyle={styles.content}>
    <AppHeader onBack={() => router.back()} />
    <View style={styles.heroBrand}><LovedBrandLockup compact /><LovedBotanical style={styles.botanical} /></View>
    <Text accessibilityRole="header" style={styles.title}>Trends</Text>
    <Text style={styles.subtitle}>Track conversation activity over time.</Text>
    <SurfaceCard style={styles.personCard}><LovedAvatar name={person?.displayName || 'Loved one'} size={88} /><View style={styles.personCardCopy}><Text style={styles.personName}>{person?.displayName || 'Loved one'}</Text><View style={styles.onlineLine}><LovedDot color="#52A688" /><Text style={styles.body}>Device online</Text></View></View><Feather color={colors.text.primary} name="chevron-right" size={23} /></SurfaceCard>
    <View style={styles.segment}><View style={styles.segmentOption}><SelectionButton label="7 days" selected={range === '7'} onPress={() => setRange('7')} /></View><View style={styles.segmentOption}><SelectionButton label="30 days" selected={range === '30'} onPress={() => setRange('30')} /></View><View style={styles.segmentOption}><SelectionButton label="3 months" selected={range === '90'} onPress={() => setRange('90')} /></View></View>
    {loading ? <ActivityIndicator color={colors.accent} /> : null}
    {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
    {!loading && !error ? <><TrendCard icon="clock" title="Conversation time" value={formatDuration(trend.reduce((sum, day) => sum + day.duration, 0))} change="18%" trend={trend} /><TrendCard icon="message-circle" title="Sessions" value={String(trend.filter((day) => !day.missed).length)} change="8%" trend={trend} /><SurfaceCard style={styles.tipCard}><LovedIconCircle icon="feather" size={54} /><View style={styles.tipCopy}><Text style={styles.cardTitle}>Tip</Text><Text style={styles.body}>Short, regular chats help build connection. You’re doing great.</Text></View></SurfaceCard></> : null}
  </ScreenLayout>;
}

function TrendCard({ icon, title, value, change, trend }: { icon: IconName; title: string; value: string; change: string; trend: V1TrendDay[] }) {
  const maxDuration = Math.max(...trend.map((item) => item.duration), 1);
  return <SurfaceCard style={styles.chartCard}><View style={styles.chartHeading}><LovedIconCircle icon={icon} size={54} /><View style={styles.chartHeadingCopy}><Text style={styles.cardTitle}>{title}</Text><Text style={styles.chartValue}>{value}</Text><Text style={styles.note}>Total this period</Text></View><View style={styles.chartChange}><Text style={styles.changeValue}>↑ {change}</Text><Text style={styles.note}>vs last period</Text></View></View><ScrollView horizontal showsHorizontalScrollIndicator={trend.length > 30} contentContainerStyle={styles.chartScroll}><View accessibilityLabel={`${trend.filter((day) => !day.missed).length} of ${trend.length} days with a completed session`} style={[styles.bars, trend.length > 30 && styles.barsWide]}>{trend.map((day) => <View key={day.date} style={[styles.barWrap, trend.length > 30 && styles.barWrapWide]}><View style={[styles.bar, { height: Math.max(4, Math.min(110, (day.duration / maxDuration) * 110)), backgroundColor: day.status === 'green' ? '#B8D5CC' : day.status === 'amber' ? '#EBCB9A' : '#E8E0D6' }]} /><Text style={styles.barLabel}>{day.date.slice(8)}</Text></View>)}</View></ScrollView></SurfaceCard>;
}

export function HistoryScreen() {
  const router = useRouter();
  const { id } = useLocalSearchParams<{ id: string }>();
  const { width } = useWindowDimensions();
  const [person, setPerson] = useState<CaregiverHomePatient | null>(null);
  const [month, setMonth] = useState(() => monthKey());
  const [historyView, setHistoryView] = useState<'calendar' | 'chronological'>('calendar');
  const [days, setDays] = useState<Awaited<ReturnType<typeof listSessionDaysV1>>>([]);
  const [selectedDate, setSelectedDate] = useState('');
  const [selectedSessions, setSelectedSessions] = useState<V1SessionDetail[]>([]);
  const [loading, setLoading] = useState(true);
  const [dayLoading, setDayLoading] = useState(false);
  const [error, setError] = useState('');
  const calendarTranslation = useRef(new Animated.Value(0)).current;

  const refresh = useCallback(async () => {
    if (!id) return;
    setLoading(true); setError('');
    calendarTranslation.stopAnimation();
    calendarTranslation.setValue(0);
    try {
      const home = await loadCaregiverHome();
      setPerson(patientFromHome(home, id));
      const nextDays = await listSessionDaysV1(id, month);
      setDays(nextDays);
      const first = nextDays.find((day) => day.hasCompletedSession)?.date || nextDays[0]?.date || '';
      setSelectedDate(first);
      if (first) setSelectedSessions((await getSessionDayV1(id, first)).sessions);
      else setSelectedSessions([]);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'History could not be loaded.');
    } finally { setLoading(false); }
  }, [calendarTranslation, id, month]);
  useEffect(() => { void refresh(); }, [refresh]);

  const selectDate = async (date: string) => {
    if (!id) return;
    setSelectedDate(date); setDayLoading(true); setError('');
    try { setSelectedSessions((await getSessionDayV1(id, date)).sessions); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'That day could not be loaded.'); }
    finally { setDayLoading(false); }
  };

  const changeMonth = useCallback((delta: number, velocity = 0) => {
    const direction = delta > 0 ? -1 : 1;
    const distance = Math.max(300, Dimensions.get('window').width);
    calendarTranslation.stopAnimation();
    Animated.spring(calendarTranslation, {
      damping: 34,
      mass: 1,
      overshootClamping: true,
      stiffness: 360,
      toValue: direction * distance,
      useNativeDriver: true,
      velocity,
    }).start(({ finished }) => {
      if (!finished) return;
      setMonth((current) => shiftMonth(current, delta));
      calendarTranslation.setValue(-direction * distance);
      Animated.spring(calendarTranslation, { damping: 34, mass: 1, overshootClamping: true, stiffness: 360, toValue: 0, useNativeDriver: true }).start();
    });
  }, [calendarTranslation]);

  const calendarPager = useMemo(() => PanResponder.create({
    onMoveShouldSetPanResponder: (_, gesture) => Math.abs(gesture.dx) > 12 && Math.abs(gesture.dx) > Math.abs(gesture.dy),
    onPanResponderGrant: () => calendarTranslation.stopAnimation(),
    onPanResponderMove: (_, gesture) => calendarTranslation.setValue(Math.max(-140, Math.min(140, gesture.dx))),
    onPanResponderRelease: (_, gesture) => {
      const threshold = Math.min(96, Math.max(72, Dimensions.get('window').width * 0.2));
      if (gesture.dx < -threshold) changeMonth(1, gesture.vx);
      else if (gesture.dx > threshold) changeMonth(-1, gesture.vx);
      else Animated.spring(calendarTranslation, { damping: 34, mass: 1, overshootClamping: true, stiffness: 360, toValue: 0, useNativeDriver: true, velocity: gesture.vx }).start();
    },
  }), [calendarTranslation, changeMonth]);

  const calendar = useMemo(() => buildMonthCalendar(month), [month]);
  const calendarRows = useMemo(() => Array.from({ length: calendar.length / 7 }, (_, index) => calendar.slice(index * 7, index * 7 + 7)), [calendar]);
  const daysByDate = useMemo(() => new Map(days.map((day) => [day.date, day])), [days]);
  const chronologicalDays = useMemo(() => [...days].sort((left, right) => right.date.localeCompare(left.date)), [days]);

  const renderCalendarCell = (cell: CalendarCell) => {
    const day = daysByDate.get(cell.date);
    const selected = selectedDate === cell.date;
    const status = day?.hasCompletedSession ? `${day.completedCount} completed session${day.completedCount === 1 ? '' : 's'}` : day?.count ? `${day.count} recorded item${day.count === 1 ? '' : 's'}` : 'No recorded sessions';
    const label = `${displayDate(cell.date)}. ${status}.`;
    const content = <><Text style={[styles.calendarDayNumber, !cell.inMonth && styles.calendarDayMuted, selected && styles.calendarDaySelected]}>{cell.day}</Text>{cell.inMonth && day?.hasCompletedSession ? <View style={[styles.calendarDot, selected && styles.calendarDotSelected]} /> : cell.inMonth && day?.count ? <View style={[styles.calendarDot, styles.calendarDotMuted, selected && styles.calendarDotSelected]} /> : null}</>;
    if (!cell.inMonth) return <View key={cell.date} accessible={false} style={[styles.calendarCell, styles.calendarCellOut]}>{content}</View>;
    return <MotionPressable key={cell.date} accessibilityLabel={label} accessibilityRole="button" accessibilityState={{ selected }} feedback="card" haptic="selection" onPress={() => void selectDate(cell.date)} style={[styles.calendarCell, selected && styles.calendarCellSelected]}>{content}</MotionPressable>;
  };

  return <ScreenLayout contentContainerStyle={styles.content}>
    <AppHeader onBack={() => router.back()} />
    <View style={styles.pageHero}><View style={styles.pageHeroCopy}><Text accessibilityRole="header" style={styles.title}>History</Text><Text style={styles.personName}>{person?.displayName || 'Loved one'}</Text></View><LovedBotanical variant="pale" style={styles.botanical} /></View>
    <Text style={styles.subtitle}>Calendar and chronological views for the selected loved one.</Text>
    <View style={[styles.viewToggle, width < 400 && styles.viewToggleNarrow]}><View style={[styles.toggleOption, width < 400 && styles.toggleOptionNarrow]}><SelectionButton label="Calendar" selected={historyView === 'calendar'} onPress={() => setHistoryView('calendar')} /></View><View style={[styles.toggleOption, width < 400 && styles.toggleOptionNarrow]}><SelectionButton label="Chronological" selected={historyView === 'chronological'} onPress={() => setHistoryView('chronological')} /></View></View>
    {loading ? <ActivityIndicator color={colors.accent} /> : null}{error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
    {!loading && historyView === 'calendar' ? <SurfaceCard style={styles.calendarCard}><Animated.View {...calendarPager.panHandlers} style={[styles.calendarAnimated, { transform: [{ translateX: calendarTranslation }] }]}><View style={styles.monthHeader}><MotionPressable accessibilityLabel="Previous month" accessibilityRole="button" haptic="selection" onPress={() => changeMonth(-1)} style={styles.monthButton}><Feather color={colors.text.primary} name="chevron-left" size={22} /></MotionPressable><Text accessibilityRole="header" style={styles.month}>{monthLabel(month)}</Text><MotionPressable accessibilityLabel="Next month" accessibilityRole="button" haptic="selection" onPress={() => changeMonth(1)} style={styles.monthButton}><Feather color={colors.text.primary} name="chevron-right" size={22} /></MotionPressable></View><View style={styles.weekdayRow}>{WEEKDAY_LABELS.map((label) => <Text key={label} style={styles.weekday}>{label}</Text>)}</View><View style={styles.calendarGrid}>{calendarRows.map((row, index) => <View key={`week-${index}`} style={[styles.calendarRow, index === calendarRows.length - 1 && styles.calendarRowLast]}>{row.map(renderCalendarCell)}</View>)}</View></Animated.View></SurfaceCard> : null}
    {!loading && historyView === 'chronological' ? <SurfaceCard style={styles.card}><Text style={styles.cardTitle}>{monthLabel(month)}</Text>{chronologicalDays.length ? chronologicalDays.map((day) => <MotionPressable key={day.date} accessibilityLabel={`${displayDate(day.date)}. ${day.hasCompletedSession ? `${day.completedCount} completed session${day.completedCount === 1 ? '' : 's'}` : `${day.count} recorded item${day.count === 1 ? '' : 's'}`}`} accessibilityRole="button" accessibilityState={{ selected: selectedDate === day.date }} feedback="card" haptic="selection" onPress={() => void selectDate(day.date)} style={[styles.daySummaryRow, selectedDate === day.date && styles.daySummaryRowSelected]}><View style={styles.daySummaryCopy}><Text style={styles.daySummaryDate}>{displayDate(day.date)}</Text><Text style={styles.daySummaryMeta}>{day.hasCompletedSession ? `${day.completedCount} completed session${day.completedCount === 1 ? '' : 's'}` : day.count ? `${day.count} recorded item${day.count === 1 ? '' : 's'}` : 'No recorded sessions'}</Text></View><Feather color={selectedDate === day.date ? colors.accent : colors.textDecorative} name={selectedDate === day.date ? 'check-circle' : 'chevron-right'} size={20} /></MotionPressable>) : <Text style={styles.body}>No recorded sessions in this month.</Text>}</SurfaceCard> : null}
    {dayLoading ? <ActivityIndicator color={colors.accent} /> : null}
    {selectedDate ? <MotionFadeIn playKey={selectedDate}><SurfaceCard style={styles.card}><Text style={styles.cardTitle}>{displayDate(selectedDate)}</Text>{selectedSessions.length ? selectedSessions.map((session) => <MotionPressable key={session.id} accessibilityRole="button" feedback="card" onPress={() => router.push(`/loved-one/${id}/sessions/${session.id}`)} style={styles.sessionRow}><View style={styles.sessionIcon}><Feather color={colors.accent} name="message-circle" size={18} /></View><View style={styles.rowCopy}><Text style={styles.rowTitle}>{session.type === 'daily_checkin' ? 'Daily check-in' : 'Conversation'}</Text><Text style={styles.rowMeta}>{displayDate(session.createdAt, true)} · {formatDuration(session.duration)}</Text></View><Feather color={colors.textDecorative} name="chevron-right" size={20} /></MotionPressable>) : <Text style={styles.body}>No conversation sessions were recorded on this day.</Text>}</SurfaceCard></MotionFadeIn> : null}
  </ScreenLayout>;
}

type SummaryKey = 'conversations' | 'routines' | 'actions' | 'messages';

const SUMMARY_SECTIONS: Array<{ key: SummaryKey; icon: IconName; title: string; description: string }> = [
  { key: 'conversations', icon: 'message-circle', title: 'Conversations & interaction time', description: 'Session details and durations' },
  { key: 'routines', icon: 'calendar', title: 'Routines', description: 'Scheduled and recorded routine responses' },
  { key: 'actions', icon: 'sun', title: 'Recommended actions', description: 'Suggestions and actions taken' },
  { key: 'messages', icon: 'mail', title: 'Messages', description: 'Sent and received messages' },
];

function SummaryToggleRow({ icon, title, description, selected, onPress }: { icon: IconName; title: string; description: string; selected: boolean; onPress: () => void }) {
  return <MotionPressable accessibilityLabel={`${title}. ${description}`} accessibilityRole="button" accessibilityState={{ selected }} feedback="card" haptic="selection" onPress={onPress} style={styles.summaryToggleRow}>
    <View style={[styles.summaryCheckbox, selected && styles.summaryCheckboxSelected]}>{selected ? <Feather color={colors.text.onAccent} name="check" size={17} /> : null}</View>
    <View style={styles.summaryIcon}><Feather color={colors.accent} name={icon} size={22} /></View>
    <View style={styles.summaryToggleCopy}><Text style={styles.summaryToggleTitle}>{title}</Text><Text style={styles.summaryToggleDescription}>{description}</Text></View>
  </MotionPressable>;
}

function SummaryMetric({ icon, value, label, detail }: { icon: IconName; value: string; label: string; detail: string }) {
  return <View style={styles.summaryMetric}><View style={styles.summaryMetricIcon}><Feather color={colors.accent} name={icon} size={20} /></View><View style={styles.summaryMetricCopy}><View style={styles.summaryMetricTop}><Text style={styles.summaryMetricValue}>{value}</Text><Text style={styles.summaryMetricLabel}>{label}</Text></View><Text style={styles.summaryMetricDetail}>{detail}</Text></View></View>;
}

export function ExportSummariesScreen() {
  const router = useRouter();
  const { id } = useLocalSearchParams<{ id: string }>();
  const [person, setPerson] = useState<CaregiverHomePatient | null>(null);
  const [range, setRange] = useState<7 | 30>(7);
  const [sections, setSections] = useState<Record<SummaryKey, boolean>>({ conversations: true, routines: true, messages: true, actions: true });
  const [feed, setFeed] = useState<V1SessionFeed | null>(null);
  const [trend, setTrend] = useState<V1TrendDay[]>([]);
  const [routines, setRoutines] = useState<V1ReminderOccurrence[]>([]);
  const [messages, setMessages] = useState<Awaited<ReturnType<typeof listFamilyMessagesV1>>>([]);
  const [loading, setLoading] = useState(true);
  const [sharing, setSharing] = useState(false);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    if (!id) return;
    setLoading(true); setError('');
    try {
      const home = await loadCaregiverHome();
      setPerson(patientFromHome(home, id));
      const { from, to } = rangeDates(range);
      const [nextFeed, nextTrend, nextRoutines, nextMessages] = await Promise.all([
        listSessionsV1(id, { limit: 50 }),
        getSessionTrendV1(id, range === 7 ? 7 : 30),
        listReminderOccurrencesV1(id, from, to).catch(() => []),
        listFamilyMessagesV1(id),
      ]);
      setFeed(nextFeed); setTrend(nextTrend); setRoutines(nextRoutines); setMessages(nextMessages);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Export information could not be loaded.');
    } finally { setLoading(false); }
  }, [id, range]);
  useFocusEffect(useCallback(() => { void refresh(); }, [refresh]));

  const shareSummary = async () => {
    if (!person || !feed) return;
    setSharing(true); setError('');
    try {
      const lines = [`Reflexion summary · ${person.displayName}`, `Period: last ${range} days`, ''];
      if (sections.conversations) lines.push(`Conversations: ${feed.sessions.length}`, `Conversation time: ${formatDuration(feed.sessions.reduce((sum, session) => sum + session.duration, 0))}`, `Days with a completed session: ${trend.filter((day) => !day.missed).length}`);
      if (sections.routines) lines.push(`Routine occurrences: ${routines.length}`);
      if (sections.messages) lines.push(`Family messages: ${messages.length}`);
      if (sections.actions) lines.push('Suggested next step: Review any session or device update that needs your attention.');
      lines.push('', 'This summary contains recorded or reported information. It is not a medical record.');
      await Share.share({ message: lines.join('\n'), title: `Reflexion summary · ${person.displayName}` });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The summary could not be shared.');
    } finally { setSharing(false); }
  };

  const toggle = (key: SummaryKey) => setSections((current) => ({ ...current, [key]: !current[key] }));
  const sessionCount = feed?.sessions.length || 0;
  const conversationTime = formatDuration(feed?.sessions.reduce((sum, session) => sum + session.duration, 0) || 0);
  const interactionDays = trend.filter((day) => !day.missed).length;
  return <ScreenLayout contentContainerStyle={styles.content}>
    <AppHeader onBack={() => router.back()} />
    <View style={[styles.pageHero, styles.exportHero]}><View style={styles.pageHeroCopy}><Text accessibilityRole="header" style={styles.title}>Export Summaries</Text></View><LovedBotanical variant="pale" style={styles.botanical} /></View>
    <Text style={styles.subtitle}>Create a clear summary of recorded information before sharing it with family or care providers.</Text>
    <Text style={styles.sectionLabel}>DATE RANGE</Text>
    <SurfaceCard style={styles.rangeCard}><View style={styles.rangeHeading}><View style={styles.rangeIcon}><Feather color={colors.accent} name="calendar" size={22} /></View><View style={styles.rangeCopy}><Text style={styles.rangeTitle}>Choose a period</Text><Text style={styles.rangeDescription}>Select the recorded information to include.</Text></View></View><View style={styles.rangeOptions}><View style={styles.rangeOption}><SelectionButton label="Last 7 days" selected={range === 7} onPress={() => setRange(7)} /></View><View style={styles.rangeOption}><SelectionButton label="Last 30 days" selected={range === 30} onPress={() => setRange(30)} /></View></View></SurfaceCard>
    <Text style={styles.sectionLabel}>INCLUDE IN SUMMARY</Text>
    <SurfaceCard style={styles.includeCard}>{SUMMARY_SECTIONS.map((section, index) => <View key={section.key} style={styles.summaryRowWrap}><SummaryToggleRow icon={section.icon} title={section.title} description={section.description} selected={sections[section.key]} onPress={() => toggle(section.key)} />{index < SUMMARY_SECTIONS.length - 1 ? <View style={styles.summaryDivider} /> : null}</View>)}</SurfaceCard>
    {loading ? <ActivityIndicator color={colors.accent} /> : null}{error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
    {!loading && !error ? <><Text style={styles.sectionLabel}>SUMMARY PREVIEW</Text><SurfaceCard style={styles.previewCard}><View style={styles.previewIntro}><Feather color={colors.accent} name="file-text" size={28} /><View style={styles.previewIntroCopy}><Text style={styles.previewIntroLabel}>SUMMARY PREVIEW</Text><Text style={styles.previewIntroText}>Review the information that will be included before sharing.</Text></View></View><View style={styles.previewMetrics}>{sections.conversations ? <SummaryMetric icon="message-circle" value={String(sessionCount)} label="Sessions" detail={`Last ${range} days`} /> : null}{sections.conversations ? <SummaryMetric icon="clock" value={conversationTime} label="Conversation time" detail="Across recorded sessions" /> : null}{sections.conversations ? <SummaryMetric icon="calendar" value={String(interactionDays)} label="Days with interaction" detail={`Out of ${range} days`} /> : null}{sections.messages ? <SummaryMetric icon="mail" value={String(messages.length)} label="Messages" detail="In the delivery record" /> : null}{sections.routines ? <ProvenanceSection label="Routines">{routines.length} reminder occurrence{routines.length === 1 ? '' : 's'}.</ProvenanceSection> : null}{sections.actions ? <ProvenanceSection label="Recommended actions">Review any recorded update that needs your attention.</ProvenanceSection> : null}</View><Text style={styles.previewNote}>This summary contains recorded or reported information. It is not a medical record.</Text></SurfaceCard></> : null}
    <PrimaryButton disabled={loading || sharing || !feed} label={sharing ? 'Preparing…' : 'Share summaries'} onPress={() => void shareSummary()} />
    <TertiaryButton label="Back to dashboard" onPress={() => router.replace(`/loved-one/${id}`)} />
  </ScreenLayout>;
}

const styles = StyleSheet.create({
  loading: { alignItems: 'center', justifyContent: 'center' },
  content: { gap: spacing.lg, minWidth: 0 },
  pageHero: { minHeight: 112, minWidth: 0, position: 'relative' },
  exportHero: { minHeight: 88 },
  pageHeroCopy: { gap: spacing.xs, minWidth: 0, paddingRight: spacing.xxl },
  heroBrand: { minHeight: 76, minWidth: 0, position: 'relative' },
  botanical: { height: 274, right: -spacing.xl, top: -spacing.xxl, width: 188 },
  title: { ...typography.display, color: colors.text.primary, flexShrink: 1, marginTop: spacing.lg, minWidth: 0 },
  personName: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontStyle: 'italic', fontWeight: '400', lineHeight: 38, minWidth: 0 },
  subtitle: { ...typography.bodyLarge, color: colors.text.secondary, flexShrink: 1, minWidth: 0 },
  error: { ...typography.body, color: colors.error.text, flexShrink: 1 },
  card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, overflow: 'hidden', padding: spacing.lg },
  cardTitle: { ...typography.label, color: colors.text.primary },
  body: { ...typography.bodyLarge, color: colors.text.primary, flexShrink: 1 },
  note: { ...typography.caption, color: colors.text.secondary, flexShrink: 1 },
  metricGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.md, minWidth: 0 },
  metric: { ...cardShadow, alignItems: 'flex-start', backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, flexBasis: '46%', flexDirection: 'row', flexGrow: 1, gap: spacing.md, minHeight: 126, minWidth: 0, padding: spacing.lg },
  metricCopy: { flex: 1, gap: spacing.xs, minWidth: 0 },
  metricValue: { ...typography.section, color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '400' },
  metricLabel: { ...typography.bodyLarge, color: colors.text.primary, flexShrink: 1, minWidth: 0 },
  metricDetail: { ...typography.body, color: colors.text.secondary, flexShrink: 1, minWidth: 0 },
  cardHeading: { alignItems: 'center', flexDirection: 'row', gap: spacing.md, minWidth: 0 },
  cardHeadingCopy: { flex: 1, gap: spacing.xs, minWidth: 0 },
  personCard: { alignItems: 'center', flexDirection: 'row', gap: spacing.lg, padding: spacing.lg },
  personCardCopy: { flex: 1, gap: spacing.xs, minWidth: 0 },
  onlineLine: { alignItems: 'center', flexDirection: 'row', gap: spacing.sm, minWidth: 0 },
  segment: { flexDirection: 'row', gap: 0, minWidth: 0 },
  segmentOption: { flex: 1, minWidth: 0 },
  chartCard: { ...cardShadow, gap: spacing.lg },
  chartHeading: { alignItems: 'flex-start', flexDirection: 'row', gap: spacing.md, minWidth: 0 },
  chartHeadingCopy: { flex: 1, gap: spacing.xs, minWidth: 0 },
  chartValue: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.display, lineHeight: 42 },
  chartChange: { alignItems: 'flex-end', gap: spacing.xs, minWidth: 0 },
  changeValue: { ...typography.bodyLarge, color: colors.accent, fontWeight: '600' },
  tipCard: { alignItems: 'center', backgroundColor: '#F0F7F3', borderColor: '#D6E8DA', flexDirection: 'row', gap: spacing.lg },
  tipCopy: { flex: 1, gap: spacing.xs, minWidth: 0 },
  chartScroll: { minWidth: '100%' },
  bars: { alignItems: 'flex-end', flexDirection: 'row', gap: 3, height: 150, minWidth: '100%', paddingTop: spacing.md },
  barsWide: { minWidth: 1100 },
  barWrap: { alignItems: 'center', alignSelf: 'stretch', flex: 1, justifyContent: 'flex-end', minWidth: 2 },
  barWrapWide: { flex: 0, width: 10 },
  bar: { borderRadius: 4, minHeight: 4, width: '100%' },
  barLabel: { color: colors.text.secondary, fontSize: 9, marginTop: 4 },
  viewToggle: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, minWidth: 0 },
  viewToggleNarrow: { flexDirection: 'column' },
  toggleOption: { flexBasis: 140, flexGrow: 1, minWidth: 0 },
  toggleOptionNarrow: { flexBasis: 'auto', flexGrow: 0, width: '100%' },
  calendarCard: { overflow: 'hidden', padding: 0 },
  calendarAnimated: { minWidth: '100%' },
  monthHeader: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between', minWidth: 0, paddingHorizontal: spacing.md, paddingVertical: spacing.md },
  monthButton: { alignItems: 'center', borderColor: colors.border.default, borderRadius: radius.pill, borderWidth: 1, flexShrink: 0, height: 44, justifyContent: 'center', width: 44 },
  month: { ...typography.bodyLarge, color: colors.text.primary, flex: 1, fontWeight: '600', minWidth: 0, textAlign: 'center' },
  weekdayRow: { borderBottomColor: colors.border.subtle, borderBottomWidth: 1, flexDirection: 'row', minWidth: 0, paddingHorizontal: spacing.xs },
  weekday: { ...typography.caption, color: colors.text.secondary, flex: 1, fontWeight: '700', minWidth: 0, paddingBottom: spacing.sm, textAlign: 'center' },
  calendarGrid: { minWidth: 0 },
  calendarRow: { borderBottomColor: colors.border.subtle, borderBottomWidth: 1, flexDirection: 'row', minWidth: 0, paddingHorizontal: spacing.xs },
  calendarRowLast: { borderBottomWidth: 0 },
  calendarCell: { alignItems: 'center', flex: 1, justifyContent: 'center', minHeight: 58, minWidth: 0, paddingHorizontal: 2, paddingVertical: spacing.sm },
  calendarCellOut: { opacity: 0.55 },
  calendarCellSelected: { backgroundColor: colors.accent, borderRadius: radius.pill },
  calendarDayNumber: { ...typography.bodyLarge, color: colors.text.primary, fontWeight: '500', textAlign: 'center' },
  calendarDayMuted: { color: colors.text.secondary },
  calendarDaySelected: { color: colors.text.onAccent },
  calendarDot: { backgroundColor: colors.status.green, borderRadius: radius.pill, height: 6, marginTop: spacing.xs, width: 6 },
  calendarDotMuted: { backgroundColor: colors.textDecorative },
  calendarDotSelected: { backgroundColor: colors.text.onAccent },
  daySummaryRow: { alignItems: 'center', borderBottomColor: colors.border.subtle, borderBottomWidth: 1, flexDirection: 'row', gap: spacing.md, minWidth: 0, paddingVertical: spacing.md },
  daySummaryRowSelected: { backgroundColor: '#F0F8F4', borderRadius: radius.md, paddingHorizontal: spacing.sm },
  daySummaryCopy: { flex: 1, flexShrink: 1, minWidth: 0 },
  daySummaryDate: { ...typography.bodyLarge, color: colors.text.primary, flexShrink: 1, minWidth: 0 },
  daySummaryMeta: { ...typography.caption, color: colors.text.secondary, flexShrink: 1, minWidth: 0, marginTop: 2 },
  sessionRow: { alignItems: 'center', borderBottomColor: colors.border.subtle, borderBottomWidth: 1, flexDirection: 'row', gap: spacing.md, minHeight: 66, minWidth: 0, paddingVertical: spacing.md },
  sessionIcon: { alignItems: 'center', backgroundColor: '#EEF3E9', borderRadius: radius.pill, height: 38, justifyContent: 'center', width: 38 },
  rowCopy: { flex: 1, flexShrink: 1, minWidth: 0 },
  rowTitle: { ...typography.label, color: colors.text.primary, flexShrink: 1, minWidth: 0 },
  rowMeta: { ...typography.caption, color: colors.text.secondary, flexShrink: 1, marginTop: 3, minWidth: 0 },
  sectionLabel: { ...typography.caption, color: colors.text.primary, fontWeight: '700', letterSpacing: 0.5, marginTop: spacing.sm },
  rangeCard: { gap: spacing.md },
  rangeHeading: { alignItems: 'center', flexDirection: 'row', gap: spacing.md, minWidth: 0 },
  rangeIcon: { alignItems: 'center', backgroundColor: '#EEF5EF', borderRadius: radius.pill, flexShrink: 0, height: 42, justifyContent: 'center', width: 42 },
  rangeCopy: { flex: 1, flexShrink: 1, minWidth: 0 },
  rangeTitle: { ...typography.label, color: colors.text.primary, flexShrink: 1, minWidth: 0 },
  rangeDescription: { ...typography.caption, color: colors.text.secondary, flexShrink: 1, marginTop: 2, minWidth: 0 },
  rangeOptions: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, minWidth: 0 },
  rangeOption: { flexBasis: 130, flexGrow: 1, minWidth: 0 },
  includeCard: { gap: 0 },
  summaryRowWrap: { minWidth: 0 },
  summaryToggleRow: { alignItems: 'center', flexDirection: 'row', gap: spacing.md, minHeight: 72, minWidth: 0, paddingVertical: spacing.sm },
  summaryCheckbox: { alignItems: 'center', borderColor: colors.border.strong, borderRadius: radius.sm, borderWidth: 1.5, flexShrink: 0, height: 28, justifyContent: 'center', width: 28 },
  summaryCheckboxSelected: { backgroundColor: colors.accent, borderColor: colors.accent },
  summaryIcon: { alignItems: 'center', backgroundColor: '#EEF5EF', borderRadius: radius.pill, flexShrink: 0, height: 44, justifyContent: 'center', width: 44 },
  summaryToggleCopy: { flex: 1, flexShrink: 1, minWidth: 0 },
  summaryToggleTitle: { ...typography.bodyLarge, color: colors.text.primary, flexShrink: 1, fontWeight: '600', minWidth: 0 },
  summaryToggleDescription: { ...typography.body, color: colors.text.secondary, flexShrink: 1, minWidth: 0 },
  summaryDivider: { backgroundColor: colors.border.subtle, height: 1, marginLeft: 88 },
  previewCard: { gap: spacing.md },
  previewIntro: { alignItems: 'center', backgroundColor: '#F0F7F3', borderColor: '#D6E8DA', borderRadius: radius.lg, borderWidth: 1, flexDirection: 'row', gap: spacing.md, minWidth: 0, padding: spacing.md },
  previewIntroCopy: { flex: 1, flexShrink: 1, minWidth: 0 },
  previewIntroLabel: { ...typography.caption, color: colors.accent, fontWeight: '700' },
  previewIntroText: { ...typography.body, color: colors.text.primary, flexShrink: 1, minWidth: 0 },
  previewMetrics: { minWidth: 0 },
  summaryMetric: { alignItems: 'flex-start', borderBottomColor: colors.border.subtle, borderBottomWidth: 1, flexDirection: 'row', gap: spacing.sm, minWidth: 0, paddingVertical: spacing.md },
  summaryMetricIcon: { alignItems: 'center', backgroundColor: '#EEF5EF', borderRadius: radius.pill, flexShrink: 0, height: 42, justifyContent: 'center', width: 42 },
  summaryMetricCopy: { flex: 1, gap: spacing.xs, minWidth: 0 },
  summaryMetricTop: { alignItems: 'baseline', flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, minWidth: 0 },
  summaryMetricValue: { ...typography.bodyLarge, color: colors.text.primary, flexShrink: 0, fontWeight: '600', minWidth: 34 },
  summaryMetricLabel: { ...typography.body, color: colors.text.primary, flexShrink: 1, minWidth: 0 },
  summaryMetricDetail: { ...typography.caption, color: colors.text.secondary, flexShrink: 1, minWidth: 0, textAlign: 'right', width: '100%' },
  previewNote: { ...typography.caption, color: colors.text.secondary, flexShrink: 1, minWidth: 0 },
});
