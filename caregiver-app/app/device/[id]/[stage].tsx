import { CameraView, useCameraPermissions } from 'expo-camera';
import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useRef, useState } from 'react';
import { Alert, StyleSheet, Text, View } from 'react-native';
import { AppHeader, ChoiceCard, PrimaryButton, ScreenLayout, SecondaryButton, TertiaryButton } from '../../../src/components/AppUI';
import { Field } from '../../../src/components/Field';
import { claimDevicePairingV1, listDeviceAssignmentsV1 } from '../../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, spacing } from '../../../src/theme';

type Stage = { title: string; subtitle: string; action: string; next?: string; items: { icon: 'camera' | 'hash' | 'wifi' | 'mic' | 'volume-2' | 'map-pin' | 'check-circle'; title: string; description: string }[] };

const STAGES: Record<string, Stage> = {
  pairing: { title: 'Choose pairing method', subtitle: 'Connect the Mirror with a QR code or a six-digit code.', action: 'Continue', next: 'qr', items: [{ icon: 'camera', title: 'Scan QR code', description: 'Use your phone camera to scan the code on the Mirror.' }, { icon: 'hash', title: 'Use pairing code', description: 'Enter the six-digit code shown on the Mirror.' }] },
  qr: { title: 'Scan QR code', subtitle: 'Position the Mirror QR code inside the frame. The code is exchanged only with this device.', action: 'Use pairing code instead', next: 'code', items: [] },
  code: { title: 'Enter six-digit pairing code', subtitle: 'Enter the code displayed on the Mirror. A successful claim links the Mirror; this phone does not pretend to test its microphone or speaker.', action: 'Connect Mirror', items: [{ icon: 'hash', title: 'Pairing code', description: 'Use the code currently shown on the Mirror.' }] },
  wifi: { title: 'Connect to Wi-Fi', subtitle: 'The Mirror manages its own household Wi-Fi connection. Keep it powered on while it reports readiness here.', action: 'Check Mirror status', next: 'test', items: [{ icon: 'wifi', title: 'Connection is handled on the Mirror', description: 'This caregiver screen reads technical status; it never claims that Wi-Fi or audio passed without a Mirror heartbeat.' }] },
  test: { title: 'Connection and audio test', subtitle: 'Read the latest technical heartbeat from the Mirror before placing it.', action: 'View placement guide', next: 'placement', items: [{ icon: 'wifi', title: 'Connection', description: 'The Mirror must have reported to Reflexion recently.' }, { icon: 'mic', title: 'Microphone', description: 'The Mirror reports whether microphone permission is available.' }, { icon: 'volume-2', title: 'Speaker', description: 'The Mirror reports whether a speaker loopback check passed.' }] },
  placement: { title: 'Placement guide', subtitle: 'Place the Mirror where your loved one can see and hear it clearly.', action: 'Finish device setup', next: 'success', items: [{ icon: 'map-pin', title: 'Placement', description: 'Eye level, comfortable distance, near power and Wi-Fi. Avoid glare and loud appliances. Keep the microphone unobstructed.' }] },
  troubleshooting: { title: 'Device troubleshooting', subtitle: 'Review the latest technical status and retry after the Mirror has had time to reconnect.', action: 'Retry connection test', next: 'test', items: [{ icon: 'wifi', title: 'Check Wi-Fi', description: 'Confirm the household network is available to the Mirror.' }, { icon: 'mic', title: 'Check microphone', description: 'Keep the microphone unobstructed and allow permission on the Mirror.' }, { icon: 'volume-2', title: 'Check speaker', description: 'Turn up the Mirror volume and run its local audio check.' }] },
  success: { title: 'Device paired successfully', subtitle: 'The Mirror is connected to this loved one and can receive messages and report factual interactions.', action: 'Return to Home', items: [{ icon: 'check-circle', title: 'Mirror connected', description: 'Technical status remains separate from conclusions about your loved one.' }] },
};

export default function DeviceStageScreen() {
  const router = useRouter();
  const { id, stage, prefilledCode } = useLocalSearchParams<{ id: string; stage: string; prefilledCode?: string }>();
  const detail = STAGES[stage] || STAGES.pairing;
  const [code, setCode] = useState(prefilledCode || '');
  const [selected, setSelected] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [permission, requestPermission] = useCameraPermissions();
  const scannedRef = useRef(false);

  const claimCode = async (rawCode: string) => {
    if (!id || !/^\d{6}$/.test(rawCode)) {
      Alert.alert('Enter the six-digit code', 'Use the code currently shown on the Mirror.');
      return;
    }
    setSubmitting(true);
    try {
      await claimDevicePairingV1({ patientId: id, pairingCode: rawCode, mirrorName: 'Reflexion Mirror' });
      router.replace(`/device/${id}/wifi`);
    } catch (cause) {
      Alert.alert('The device was not paired', cause instanceof Error ? cause.message : 'Check the code on the Mirror and try again.');
    } finally {
      setSubmitting(false);
    }
  };

  const readTechnicalStatus = async () => {
    if (!id) return false;
    try {
      const assignments = await listDeviceAssignmentsV1();
      const row = assignments.find((item) => item.patientId === id);
      if (!row?.deviceId || !row.device) {
        Alert.alert('Waiting for the Mirror', 'The secure assignment is saved, but the Mirror has not reported readiness yet. Keep it powered on and try again.');
        return false;
      }
      if (stage === 'test') {
        const heartbeat = row.device.lastHeartbeatAt ? Date.parse(row.device.lastHeartbeatAt) : Number.NaN;
        const fresh = Number.isFinite(heartbeat) && Date.now() - heartbeat < 10 * 60 * 1000;
        if (row.device.technicalState !== 'ok' || !fresh) {
          Alert.alert('Mirror needs attention', `Technical status: ${row.device.technicalState || 'unknown'}. Ask someone at the Mirror to check its connection and audio, then retry.`);
          return false;
        }
      }
      return true;
    } catch (cause) {
      Alert.alert('Could not read Mirror status', cause instanceof Error ? cause.message : 'Please try again.');
      return false;
    }
  };

  const submit = async () => {
    if (stage === 'pairing') {
      router.push(`/device/${id}/${selected === 0 ? 'qr' : 'code'}`);
      return;
    }
    if (stage === 'code') { await claimCode(code); return; }
    if (stage === 'qr') {
      if (!permission?.granted) { await requestPermission(); return; }
      router.push(`/device/${id}/code`);
      return;
    }
    if (stage === 'wifi' || stage === 'test' || stage === 'troubleshooting') {
      if (!(await readTechnicalStatus())) return;
    }
    if (detail.next) router.push(`/device/${id}/${detail.next}`); else router.replace('/(tabs)');
  };

  const onBarcodeScanned = ({ data }: { data: string }) => {
    if (scannedRef.current || submitting) return;
    const parsed = parsePairingCode(data);
    if (!parsed) return;
    scannedRef.current = true;
    setCode(parsed);
    void claimCode(parsed).finally(() => { scannedRef.current = false; });
  };

  return <ScreenLayout>
    <AppHeader title="Pair device" onBack={() => router.back()} />
    <Text accessibilityRole="header" style={styles.title}>{detail.title}</Text>
    <Text style={styles.subtitle}>{detail.subtitle}</Text>
    {stage === 'qr' ? <View style={styles.cameraFrame}>{permission?.granted ? <CameraView barcodeScannerSettings={{ barcodeTypes: ['qr'] }} onBarcodeScanned={onBarcodeScanned} style={styles.camera} /> : <View style={styles.cameraPrompt}><Text style={styles.cameraPromptTitle}>Camera permission needed</Text><Text style={styles.cameraPromptCopy}>Allow camera access to scan the code on the Mirror.</Text><PrimaryButton label="Allow camera" onPress={() => void requestPermission()} /></View>}</View> : null}
    {stage === 'code' ? <Field label="Six-digit pairing code" keyboardType="number-pad" maxLength={6} onChangeText={(value) => setCode(value.replace(/\D/g, ''))} placeholder="Enter six-digit code" value={code} /> : null}
    {detail.items.length ? <View style={styles.cards}>{detail.items.map((item, index) => <ChoiceCard key={item.title} {...item} selected={stage === 'pairing' ? selected === index : false} onPress={() => setSelected(index)} />)}</View> : null}
    {stage !== 'qr' || permission?.granted ? <PrimaryButton disabled={submitting} label={submitting ? 'Pairing device…' : detail.action} onPress={() => void submit()} /> : null}
    {stage === 'qr' ? <SecondaryButton label="Use pairing code instead" onPress={() => router.push(`/device/${id}/code`)} /> : null}
    <TertiaryButton label="Set up later" onPress={() => router.replace('/(tabs)')} />
  </ScreenLayout>;
}

function parsePairingCode(value: string) {
  try {
    const parsed = JSON.parse(value) as { pairingCode?: unknown; displayCode?: unknown };
    const code = parsed.pairingCode || parsed.displayCode;
    return typeof code === 'string' && /^\d{6}$/.test(code) ? code : null;
  } catch {
    return /^\d{6}$/.test(value.trim()) ? value.trim() : null;
  }
}

const styles = StyleSheet.create({
  title: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', marginTop: spacing.xl }, subtitle: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 }, cards: { gap: spacing.md, marginBottom: spacing.lg, marginTop: spacing.md }, cameraFrame: { aspectRatio: 1, borderColor: colors.accent, borderRadius: radius.xl, borderWidth: 2, overflow: 'hidden', width: '100%' }, camera: { flex: 1 }, cameraPrompt: { alignItems: 'center', backgroundColor: colors.surface.card, flex: 1, gap: spacing.md, justifyContent: 'center', padding: spacing.xl }, cameraPromptTitle: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.heading, fontWeight: '700', textAlign: 'center' }, cameraPromptCopy: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 21, textAlign: 'center' },
});
