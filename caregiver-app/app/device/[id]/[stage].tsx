import { CameraView, useCameraPermissions } from 'expo-camera';
import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useRef, useState } from 'react';
import { Alert, StyleSheet, Text, View } from 'react-native';

import { useCaregiver } from '../../../src/architecture/CaregiverContext';
import { AppHeader, ChoiceCard, InfoCard, PrimaryButton, ScreenLayout, SecondaryButton, TertiaryButton } from '../../../src/components/AppUI';
import { Field } from '../../../src/components/Field';
import { claimDevicePairingV1, listDeviceAssignmentsV1 } from '../../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, spacing } from '../../../src/theme';

type PairingStage = 'pairing' | 'qr' | 'code' | 'placement' | 'success' | 'troubleshooting';

const STAGES: Record<PairingStage, { title: string; subtitle: string }> = {
  pairing: { title: 'Choose pairing method', subtitle: 'Connect the Mirror with a QR code or a six-digit code shown on the Mirror.' },
  qr: { title: 'Scan QR code', subtitle: 'Position the QR code shown on the Mirror inside the frame.' },
  code: { title: 'Enter six-digit pairing code', subtitle: 'Enter the code currently shown on the Mirror. The Mirror manages its own Wi-Fi and service readiness.' },
  placement: { title: 'Placement guide', subtitle: 'Place the Mirror where your loved one can see and hear it clearly.' },
  success: { title: 'Device paired successfully', subtitle: 'The Mirror is assigned to this loved one and ready for them to use.' },
  troubleshooting: { title: 'Device troubleshooting', subtitle: 'Review the latest technical status. Wi-Fi, microphone and speaker checks are performed by the Mirror.' },
};

const PLACEMENT_ITEMS = [
  ['map-pin', 'Eye level', 'Place the Mirror where your loved one can see the screen comfortably.'],
  ['maximize-2', 'Comfortable distance', 'Keep it close enough to hear without glare from a window or lamp.'],
  ['power', 'Power and connection', 'Keep the Mirror powered on. The Mirror reports its own Wi-Fi and service readiness.'],
  ['volume-2', 'Clear surroundings', 'Avoid loud appliances and keep the microphone and speaker unobstructed.'],
] as const;

export default function DeviceStageScreen() {
  const router = useRouter();
  const { id, stage } = useLocalSearchParams<{ id: string; stage: string }>();
  const { setSetupStatus } = useCaregiver();
  const requestedStage = Array.isArray(stage) ? stage[0] : stage;
  // These paths existed in an earlier build. Keep old deep links safe, but never expose caregiver-side
  // Wi-Fi/audio setup as an active product flow.
  const currentStage: PairingStage = requestedStage === 'wifi' || requestedStage === 'test'
    ? 'troubleshooting'
    : requestedStage === 'troubleshooting' || requestedStage === 'qr' || requestedStage === 'code' || requestedStage === 'placement' || requestedStage === 'success'
      ? requestedStage
      : 'pairing';
  const detail = STAGES[currentStage];
  const [selectedMethod, setSelectedMethod] = useState<'qr' | 'code'>('qr');
  const [code, setCode] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [statusText, setStatusText] = useState('');
  const [permission, requestPermission] = useCameraPermissions();
  const scannedRef = useRef(false);

  const claimCode = async (rawCode: string) => {
    if (!id || !/^\d{6}$/.test(rawCode)) {
      Alert.alert('Enter the six-digit code', 'Use the code currently shown on the Mirror.');
      return;
    }
    setSubmitting(true);
    setStatusText('');
    try {
      await claimDevicePairingV1({ patientId: id, pairingCode: rawCode });
      setSetupStatus('pair-device', 'in-progress');
      router.replace(`/device/${id}/placement`);
    } catch (cause) {
      Alert.alert('The device was not paired', cause instanceof Error ? cause.message : 'Check the code on the Mirror and try again.');
    } finally {
      setSubmitting(false);
    }
  };

  const readTechnicalStatus = async () => {
    if (!id) return;
    setSubmitting(true);
    setStatusText('Checking the latest status from the Mirror…');
    try {
      const assignments = await listDeviceAssignmentsV1();
      const row = assignments.find((item) => item.patientId === id);
      if (!row?.deviceId || !row.device) {
        setStatusText('The Mirror has not reported a device status yet. Keep it powered on and try again.');
        return;
      }
      const lastSeen = row.device.lastHeartbeatAt
        ? ` Last seen ${new Intl.DateTimeFormat('en-SG', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(row.device.lastHeartbeatAt))}.`
        : '';
      setStatusText(`${row.device.technicalState === 'ok' ? 'The Mirror reported online.' : 'The Mirror may be offline.'}${lastSeen}`);
    } catch (cause) {
      setStatusText(cause instanceof Error ? cause.message : 'The device status could not be loaded. Try again.');
    } finally {
      setSubmitting(false);
    }
  };

  const submit = async () => {
    if (currentStage === 'pairing') {
      router.push(`/device/${id}/${selectedMethod}`);
      return;
    }
    if (currentStage === 'code') {
      await claimCode(code);
      return;
    }
    if (currentStage === 'qr') {
      if (!permission?.granted) {
        await requestPermission();
      }
      return;
    }
    if (currentStage === 'placement') {
      router.replace(`/device/${id}/success`);
      return;
    }
    if (currentStage === 'success') {
      setSetupStatus('pair-device', 'complete');
      router.replace('/(tabs)');
      return;
    }
    await readTechnicalStatus();
  };

  const onBarcodeScanned = ({ data }: { data: string }) => {
    if (scannedRef.current || submitting) return;
    const parsed = parsePairingCode(data);
    if (!parsed) return;
    scannedRef.current = true;
    setCode(parsed);
    void claimCode(parsed).finally(() => { scannedRef.current = false; });
  };

  return (
    <ScreenLayout contentContainerStyle={styles.content}>
      <AppHeader title="Pair device" onBack={() => router.back()} />
      <Text accessibilityRole="header" style={styles.title}>{detail.title}</Text>
      <Text style={styles.subtitle}>{detail.subtitle}</Text>

      {currentStage === 'pairing' ? <View style={styles.cards}>
        <ChoiceCard icon="camera" title="Scan QR code" description="Use your phone camera to scan the code on the Mirror." selected={selectedMethod === 'qr'} onPress={() => setSelectedMethod('qr')} />
        <ChoiceCard icon="hash" title="Use pairing code" description="Enter the six-digit code shown on the Mirror." selected={selectedMethod === 'code'} onPress={() => setSelectedMethod('code')} />
      </View> : null}

      {currentStage === 'qr' ? <View style={styles.cameraFrame}>
        {permission?.granted
          ? <CameraView barcodeScannerSettings={{ barcodeTypes: ['qr'] }} onBarcodeScanned={onBarcodeScanned} style={styles.camera} />
          : <View style={styles.cameraPrompt}><Text style={styles.cardTitle}>Camera permission needed</Text><Text style={styles.copy}>Allow camera access only while scanning the pairing code.</Text><PrimaryButton label="Allow camera" onPress={() => void requestPermission()} /></View>}
      </View> : null}

      {currentStage === 'code' ? <Field label="Six-digit pairing code" keyboardType="number-pad" maxLength={6} onChangeText={(value) => setCode(value.replace(/\D/g, ''))} placeholder="Enter six-digit code" value={code} /> : null}

      {currentStage === 'placement' ? <View style={styles.cards}>{PLACEMENT_ITEMS.map(([icon, title, copy]) => <InfoCard key={title} icon={icon} title={title} description={copy} />)}</View> : null}

      {currentStage === 'success' ? <View style={styles.successCard}><Text style={styles.successIcon}>✓</Text><Text style={styles.cardTitle}>Ready for your loved one to use</Text><Text style={styles.copy}>The Mirror will receive the saved configuration after it checks in. Device status remains separate from information about your loved one.</Text></View> : null}

      {currentStage === 'troubleshooting' ? <View style={styles.statusCard}><Text style={styles.cardTitle}>Technical status</Text><Text style={styles.copy}>The caregiver app does not configure Wi-Fi or claim that audio passed. The Mirror reports connection, microphone and speaker readiness.</Text>{statusText ? <Text style={styles.statusText}>{statusText}</Text> : null}</View> : null}

      {currentStage !== 'qr' || permission?.granted ? <PrimaryButton disabled={submitting} label={submitting ? 'Working…' : currentStage === 'pairing' ? 'Continue' : currentStage === 'code' ? 'Connect Mirror' : currentStage === 'placement' ? 'Finish device setup' : currentStage === 'success' ? 'Return to Home' : 'Refresh device status'} onPress={() => void submit()} /> : null}
      {currentStage === 'qr' ? <SecondaryButton label="Use pairing code instead" onPress={() => router.push(`/device/${id}/code`)} /> : null}
      {currentStage !== 'success' ? <TertiaryButton label="Set up later" onPress={() => router.replace('/(tabs)')} /> : null}
    </ScreenLayout>
  );
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
  content: { gap: spacing.lg, minWidth: 0 },
  title: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, marginTop: spacing.xl, minWidth: 0 },
  subtitle: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22, minWidth: 0 },
  cards: { gap: spacing.md, minWidth: 0 },
  cameraFrame: { aspectRatio: 1, borderColor: colors.accent, borderRadius: radius.xl, borderWidth: 2, minWidth: 0, overflow: 'hidden', width: '100%' },
  camera: { flex: 1 },
  cameraPrompt: { alignItems: 'center', backgroundColor: colors.surface.card, flex: 1, gap: spacing.md, justifyContent: 'center', minWidth: 0, padding: spacing.xl },
  successCard: { alignItems: 'center', backgroundColor: colors.status.greenBg, borderColor: '#CDE2C8', borderRadius: radius.xl, gap: spacing.md, minWidth: 0, padding: spacing.xl },
  successIcon: { color: colors.status.green, fontSize: 40, fontWeight: '700' },
  statusCard: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.xl, borderWidth: 1, gap: spacing.md, minWidth: 0, padding: spacing.xl },
  cardTitle: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, fontWeight: '700', lineHeight: 23, minWidth: 0, textAlign: 'center' },
  copy: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22, minWidth: 0 },
  statusText: { color: colors.accent, flexShrink: 1, fontSize: fontSize.body, fontWeight: '700', lineHeight: 22, minWidth: 0 },
});
