import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, Alert, StyleSheet, Text, View } from 'react-native';
import { AppHeader, PrimaryButton, ScreenLayout, SecondaryButton, SettingsRow, TertiaryButton } from '../../../src/components/AppUI';
import { listDeviceAssignmentsV1, revokeDeviceV1, type V1DeviceAssignment } from '../../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, spacing } from '../../../src/theme';

export default function DeviceDetailScreen() {
  const router = useRouter();
  const { id } = useLocalSearchParams<{ id: string }>();
  const [assignment, setAssignment] = useState<V1DeviceAssignment | null>(null);
  const [loading, setLoading] = useState(true);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    if (!id) return;
    setLoading(true); setError('');
    try {
      const next = (await listDeviceAssignmentsV1()).find((item) => item.patientId === id) || null;
      setAssignment(next);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The device status could not be loaded.');
    } finally { setLoading(false); }
  }, [id]);
  useFocusEffect(useCallback(() => { void refresh(); }, [refresh]));

  const remove = () => {
    if (!assignment?.deviceId) return;
    Alert.alert('Remove this Mirror?', 'The Mirror will stop sending updates until it is paired again. Past caregiver records remain available.', [
      { text: 'Keep device', style: 'cancel' },
      { text: 'Remove device', style: 'destructive', onPress: async () => {
        setWorking(true); setError('');
        try { await revokeDeviceV1(assignment.deviceId!, 'caregiver_requested'); router.replace('/settings/devices'); }
        catch (cause) { setError(cause instanceof Error ? cause.message : 'The device could not be removed.'); }
        finally { setWorking(false); }
      } },
    ]);
  };

  const technicalState = assignment?.device?.technicalState || 'unknown';
  const heartbeat = assignment?.device?.lastHeartbeatAt ? new Date(assignment.device.lastHeartbeatAt).toLocaleString() : 'No heartbeat yet';
  return <ScreenLayout>
    <AppHeader title="Device detail" onBack={() => router.back()} />
    <Text accessibilityRole="header" style={styles.title}>{assignment?.mirrorName || 'Reflexion Mirror'}</Text>
    <Text style={styles.subtitle}>Technical connection details only. This screen does not interpret how your loved one is doing.</Text>
    {loading ? <ActivityIndicator color={colors.accent} /> : null}
    {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
    {!loading && assignment ? <>
      <View style={styles.card}>
        <SettingsRow icon="monitor" label="Connection" value={assignment.deviceId ? 'Assigned' : 'Pairing needed'} />
        <SettingsRow icon="activity" label="Technical status" value={technicalState === 'ok' ? 'Online' : technicalState === 'possible_issue' ? 'Needs a connection check' : 'Unknown'} />
        <SettingsRow icon="clock" label="Last heartbeat" value={heartbeat} />
        <SettingsRow icon="info" label="Software" value={assignment.device?.softwareVersion || 'Not reported'} />
      </View>
      {assignment.deviceId ? <PrimaryButton disabled={working} label="Troubleshoot device" onPress={() => router.push(`/device/${id}/troubleshooting`)} /> : <PrimaryButton disabled={working} label="Pair a Mirror" onPress={() => router.push(`/device/${id}/pairing`)} />}
      {assignment.deviceId ? <SecondaryButton label="Pair again" onPress={() => router.push(`/device/${id}/code`)} /> : null}
      {assignment.deviceId ? <TertiaryButton disabled={working} label="Remove device" onPress={remove} /> : null}
    </> : null}
  </ScreenLayout>;
}

const styles = StyleSheet.create({
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', marginTop: spacing.lg },
  subtitle: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 },
  error: { color: colors.error.text, flexShrink: 1, fontSize: fontSize.body, lineHeight: 21 },
  card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, overflow: 'hidden' },
});
