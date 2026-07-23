import { Feather } from '@expo/vector-icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import * as ImagePicker from 'expo-image-picker';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  Modal,
  Platform,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { apiGet, apiSend } from '../../src/lib/apiClient';
import { clearStoredAuthSession, getStoredAuthSession, setStoredAuthSession } from '../../src/lib/authSession';
import { v1Logout } from '../../src/lib/v1Client';
import { registerPushNotificationDevice } from '../../src/lib/pushNotifications';

type AlertSensitivity =
  | 'notify_me_about_everything'
  | 'only_important_changes'
  | 'only_urgent_alerts';
type SummaryTime = '09:00' | '19:00';
type Gender = 'male' | 'female' | 'other';
type Language = 'english' | 'mandarin' | 'other';
type KeyTopic = 'family' | 'food' | 'travel' | 'work' | 'others';

type SettingsPatient = {
  id: string;
  patientId?: string;
  name: string;
  phoneNumber: string;
  age: number;
  gender: Gender | '';
  preferredLanguage: Language | '';
  usualWakeTime: string;
  speechOrHearingConditions: string;
  photoUrl?: string;
  keyTopics: KeyTopic[];
  keyTopicsOtherText: string;
};

type SettingsConfig = {
  nurseId: string;
  caregiverName: string;
  email: string;
  phoneNumber: string;
  pushNotificationsEnabled: boolean;
  alertSensitivity: AlertSensitivity;
  preferredDailySummaryTime: SummaryTime;
  storeSessionSummaries: boolean;
  patients: SettingsPatient[];
};

type PatientForm = Omit<SettingsPatient, 'age'> & { age: string };

const GENDER_OPTIONS: { value: Gender; label: string }[] = [
  { value: 'male', label: 'Male' },
  { value: 'female', label: 'Female' },
  { value: 'other', label: 'Other' },
];
const LANGUAGE_OPTIONS: { value: Language; label: string }[] = [
  { value: 'english', label: 'English' },
  { value: 'mandarin', label: 'Mandarin' },
  { value: 'other', label: 'Other' },
];
const TOPIC_OPTIONS: { value: KeyTopic; label: string }[] = [
  { value: 'family', label: 'Family' },
  { value: 'food', label: 'Food' },
  { value: 'travel', label: 'Travel' },
  { value: 'work', label: 'Work' },
  { value: 'others', label: 'Others' },
];

export default function SettingsScreen() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const session = getStoredAuthSession();
  const [notifs, setNotifs] = useState(true);
  const [summaryTime, setSummaryTime] = useState<SummaryTime>('09:00');
  const [alertLevel, setAlertLevel] = useState<AlertSensitivity>('only_important_changes');
  const [storeSummaries, setStoreSummaries] = useState(true);
  const [caregiverName, setCaregiverName] = useState('');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [config, setConfig] = useState<SettingsConfig | null>(null);
  const [editingPatient, setEditingPatient] = useState<PatientForm | null>(null);
  const latestConfigQuery = useQuery({
    queryKey: ['latestConfig', session?.nurseId || 'latest'],
    queryFn: async () => {
      const query = session?.nurseId ? `?nurseId=${encodeURIComponent(session.nurseId)}` : '';
      const body = await apiGet<Record<string, unknown> & { patients?: unknown[] }>(`/api/nurse-patient-config/latest${query}`);
      return {
        nurseId: String(body?.nurseId || ''),
        caregiverName: String(body?.caregiverName || ''),
        email: String(body?.email || ''),
        phoneNumber: String(body?.phoneNumber || ''),
        pushNotificationsEnabled: Boolean(body?.pushNotificationsEnabled),
        alertSensitivity: String(body?.alertSensitivity || 'only_important_changes') as AlertSensitivity,
        preferredDailySummaryTime: String(body?.preferredDailySummaryTime || '09:00') as SummaryTime,
        storeSessionSummaries: body?.storeSessionSummaries !== false,
        patients: Array.isArray(body?.patients) ? body.patients.map((patient) => normalizeSettingsPatient(patient as Partial<SettingsPatient>)) : [],
      } satisfies SettingsConfig;
    },
  });
  const { refetch: refetchLatestConfig } = latestConfigQuery;
  useFocusEffect(
    useCallback(() => {
      void refetchLatestConfig();
    }, [refetchLatestConfig]),
  );
  const saveNurseMutation = useMutation({
    mutationFn: (body: unknown) => apiSend<SettingsConfig>('/api/nurse-patient-config/settings', {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
    onSuccess: async (body) => {
      if (body.pushNotificationsEnabled) {
        const registration = await registerPushNotificationDevice({ nurseId: body.nurseId || config?.nurseId || '' });
        if (!registration.ok) console.warn('[SettingsScreen] push registration failed', registration.reason);
      }
      await setStoredAuthSession({
        nurseId: body.nurseId || config?.nurseId || '',
        name: body.caregiverName || caregiverName,
        email: body.email || config?.email || '',
      });
      setConfig((current) => current ? {
        ...current,
        caregiverName: body.caregiverName || caregiverName,
        phoneNumber: body.phoneNumber || phoneNumber,
        pushNotificationsEnabled: Boolean(body.pushNotificationsEnabled),
        alertSensitivity: body.alertSensitivity || alertLevel,
        preferredDailySummaryTime: body.preferredDailySummaryTime || summaryTime,
        storeSessionSummaries: body.storeSessionSummaries !== false,
      } : current);
      await queryClient.invalidateQueries({ queryKey: ['latestConfig'] });
      Alert.alert('Saved', 'Settings updated.');
    },
    onError: (err) => {
      Alert.alert('Unable to save', err instanceof Error ? err.message : 'Unable to save settings.');
    },
  });
  const savePatientMutation = useMutation({
    mutationFn: ({ patientId, body }: { patientId: string; body: unknown }) =>
      apiSend<{ patient: SettingsPatient }>(`/api/nurse-patient-config/settings/patients/${patientId}`, {
        method: 'PATCH',
        body: JSON.stringify(body),
      }),
    onSuccess: async (body) => {
      const normalized = normalizeSettingsPatient(body.patient);
      setConfig((current) => current ? {
        ...current,
        patients: current.patients.map((patient) =>
          patient.id === normalized.id || patient.patientId === normalized.id ? normalized : patient,
        ),
      } : current);
      setEditingPatient(null);
      await queryClient.invalidateQueries({ queryKey: ['latestConfig'] });
      await queryClient.invalidateQueries({ queryKey: ['patientTrend'] });
      await queryClient.invalidateQueries({ queryKey: ['sessionDay'] });
    },
    onError: (err) => {
      Alert.alert('Unable to save', err instanceof Error ? err.message : 'Unable to save loved one profile.');
    },
  });

  useEffect(() => {
    if (!latestConfigQuery.data) return;
    setConfig(latestConfigQuery.data);
    setCaregiverName(latestConfigQuery.data.caregiverName);
    setPhoneNumber(latestConfigQuery.data.phoneNumber);
    setNotifs(latestConfigQuery.data.pushNotificationsEnabled);
    setAlertLevel(latestConfigQuery.data.alertSensitivity);
    setSummaryTime(latestConfigQuery.data.preferredDailySummaryTime);
    setStoreSummaries(latestConfigQuery.data.storeSessionSummaries);
  }, [latestConfigQuery.data]);

  async function saveNurseSettings() {
    if (saveNurseMutation.isPending || !config) return;
    saveNurseMutation.mutate({
      nurseId: config.nurseId,
      name: caregiverName,
      phoneNumber,
      pushNotificationsEnabled: notifs,
      alertSensitivity: alertLevel,
      preferredDailySummaryTime: summaryTime,
      storeSessionSummaries: storeSummaries,
    });
  }

  async function savePatientProfile() {
    if (!editingPatient || !config || savePatientMutation.isPending) return;

    const age = Number(editingPatient.age);
    if (!Number.isInteger(age)) {
      Alert.alert('Invalid age', 'Age must be a whole number.');
      return;
    }

    savePatientMutation.mutate({
      patientId: editingPatient.patientId || editingPatient.id,
      body: {
        nurseId: config.nurseId,
        ...editingPatient,
        age,
      },
    });
  }

  async function logout() {
    await Promise.all([clearStoredAuthSession(), v1Logout()]);
    router.replace('/sign-in');
  }

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
        <View style={styles.titleRow}>
          <Text style={styles.pageTitle}>Settings</Text>
        </View>

        {latestConfigQuery.isLoading && !config ? (
          <View style={styles.loadingCard}>
            <ActivityIndicator color="#87566A" />
            <Text style={styles.loadingTitle}>Bear with us</Text>
            <Text style={styles.loadingText}>We are loading your settings.</Text>
          </View>
        ) : (
          <>
            <SectionHeader title="Account" />
            <InputRow label="Name" value={caregiverName} onChangeText={setCaregiverName} />
            <SettingRow label="Email" value={config?.email || 'Not connected'} />
            <InputRow label="Phone" value={phoneNumber} onChangeText={setPhoneNumber} keyboardType="phone-pad" />

            <SectionHeader title="Notifications" />
            <SwitchRow label="Enable push notifications" value={notifs} onChange={setNotifs} />
            <PickerRow
              label="Daily summary"
              options={[
                { value: '09:00', label: 'Morning (9am)' },
                { value: '19:00', label: 'Evening (7pm)' },
              ]}
              selected={summaryTime}
              onSelect={v => setSummaryTime(v as SummaryTime)}
            />
            <PickerRow
              label="Alert sensitivity"
              options={[
                { value: 'notify_me_about_everything', label: 'Notify me about everything' },
                { value: 'only_important_changes', label: 'Only important changes' },
                { value: 'only_urgent_alerts', label: 'Only urgent alerts' },
              ]}
              selected={alertLevel}
              onSelect={v => setAlertLevel(v as AlertSensitivity)}
            />

            <SectionHeader title="Mirrors" />
            <ActionRow label="Manage linked mirrors" onPress={() => router.push('/mirror-management')} />

            <SectionHeader title="Loved one profiles" />
            {(config?.patients || []).map(patient => (
              <ActionRow
                key={patient.id}
                label={patient.name}
                value={formatLanguage(patient.preferredLanguage)}
                imageUrl={patient.photoUrl}
                fallbackName={patient.name}
                onPress={() => setEditingPatient(toPatientForm(patient))}
              />
            ))}
            <ActionRow label="Add a loved one" onPress={() => router.push('/onboarding?mode=add-patient&returnTo=settings')} />

            <SectionHeader title="Privacy & Data" />
            <SwitchRow label="Store session summaries" value={storeSummaries} onChange={setStoreSummaries} />
            <TouchableOpacity
              disabled={saveNurseMutation.isPending}
              onPress={() => void saveNurseSettings()}
              style={[styles.saveBtn, saveNurseMutation.isPending && styles.saveBtnDisabled]}
            >
              {saveNurseMutation.isPending ? <ActivityIndicator color="#FFFFFF" /> : <Text style={styles.saveBtnText}>Save changes</Text>}
            </TouchableOpacity>
            <ActionRow label="Export my data" onPress={() => Alert.alert('Export', 'Data export coming in V2.')} />

            <SectionHeader title="Support" />
            <ActionRow label="FAQ & Guide" onPress={() => router.push('/faq')} />
            <ActionRow label="Chat with support" onPress={() => router.push('/chatbot')} />
            <ActionRow label="Subscription & Billing" onPress={() => Alert.alert('Billing', 'Billing portal coming soon.')} />
          </>
        )}

        <TouchableOpacity onPress={() => void logout()} style={styles.logoutBtn}>
          <Text style={styles.logoutText}>Log out</Text>
        </TouchableOpacity>
      </ScrollView>
      <PatientEditModal
        patient={editingPatient}
        isSaving={savePatientMutation.isPending}
        onChange={setEditingPatient}
        onClose={() => setEditingPatient(null)}
        onSave={() => void savePatientProfile()}
      />
    </SafeAreaView>
  );
}

function toPatientForm(patient: SettingsPatient): PatientForm {
  return {
    ...patient,
    patientId: patient.patientId || patient.id,
    age: String(patient.age || ''),
    speechOrHearingConditions: patient.speechOrHearingConditions || '',
    photoUrl: patient.photoUrl || '',
    keyTopics: normalizeKeyTopics(patient.keyTopics),
    keyTopicsOtherText: patient.keyTopicsOtherText || '',
  };
}

function normalizeSettingsPatient(patient: Partial<SettingsPatient>): SettingsPatient {
  return {
    id: patient.id || patient.patientId || '',
    patientId: patient.patientId || patient.id || '',
    name: patient.name || '',
    phoneNumber: patient.phoneNumber || '',
    age: Number(patient.age || 0),
    gender: normalizeGender(patient.gender),
    preferredLanguage: normalizeLanguage(patient.preferredLanguage),
    usualWakeTime: patient.usualWakeTime || '',
    speechOrHearingConditions: patient.speechOrHearingConditions || '',
    photoUrl: patient.photoUrl || '',
    keyTopics: normalizeKeyTopics(patient.keyTopics),
    keyTopicsOtherText: patient.keyTopicsOtherText || '',
  };
}

function normalizeGender(value: unknown): Gender | '' {
  const normalized = typeof value === 'string' ? value.toLowerCase() : '';
  return normalized === 'male' || normalized === 'female' || normalized === 'other' ? normalized : '';
}

function normalizeLanguage(value: unknown): Language | '' {
  const normalized = typeof value === 'string' ? value.toLowerCase() : '';
  return normalized === 'english' || normalized === 'mandarin' || normalized === 'other' ? normalized : '';
}

function normalizeKeyTopics(value: unknown): KeyTopic[] {
  if (!Array.isArray(value)) return [];

  return value
    .map((item) => (typeof item === 'string' ? item.trim().toLowerCase() : ''))
    .filter(isKeyTopic);
}

function isKeyTopic(value: string): value is KeyTopic {
  return value === 'family' || value === 'food' || value === 'travel' || value === 'work' || value === 'others';
}

function isTopicSelected(topics: unknown, topic: KeyTopic) {
  return normalizeKeyTopics(topics).includes(topic);
}

function formatLanguage(value: string) {
  if (!value) return '';
  return value.slice(0, 1).toUpperCase() + value.slice(1);
}

function SectionHeader({ title }: { title: string }) {
  return <Text style={styles.sectionHeader}>{title}</Text>;
}

function SettingRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.row}>
      <Text style={styles.rowLabel}>{label}</Text>
      <Text style={styles.rowValue} numberOfLines={2}>{value}</Text>
    </View>
  );
}

function InputRow({
  keyboardType = 'default',
  label,
  onChangeText,
  value,
}: {
  keyboardType?: 'default' | 'phone-pad' | 'numeric';
  label: string;
  onChangeText: (value: string) => void;
  value: string;
}) {
  return (
    <View style={styles.inputRow}>
      <Text style={styles.rowLabel}>{label}</Text>
      <TextInput
        keyboardType={keyboardType}
        onChangeText={onChangeText}
        style={styles.inlineInput}
        value={value}
      />
    </View>
  );
}

function getInitials(name: string) {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return '?';
  return parts.slice(0, 2).map(part => part[0]?.toUpperCase()).join('');
}

function ActionRow({
  fallbackName,
  imageUrl,
  label,
  value,
  onPress,
}: {
  fallbackName?: string;
  imageUrl?: string;
  label: string;
  value?: string;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity style={styles.row} onPress={onPress} activeOpacity={0.7}>
      <View style={styles.rowLeft}>
        {fallbackName ? (
          <View style={styles.patientAvatar}>
            {imageUrl ? (
              <Image source={{ uri: imageUrl }} style={styles.patientAvatarImage} />
            ) : (
              <Text style={styles.patientAvatarText}>{getInitials(fallbackName)}</Text>
            )}
          </View>
        ) : null}
        <Text style={styles.rowLabel}>{label}</Text>
      </View>
      <View style={styles.rowRight}>
        {value ? <Text style={styles.rowValue}>{value}</Text> : null}
        <Feather name="chevron-right" size={16} color="#C4B9AF" />
      </View>
    </TouchableOpacity>
  );
}

function SwitchRow({
  disabled = false,
  label,
  value,
  onChange,
}: {
  disabled?: boolean;
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <View style={styles.row}>
      <Text style={styles.rowLabel}>{label}</Text>
      <Switch
        disabled={disabled}
        value={value}
        onValueChange={onChange}
        trackColor={{ false: '#D8CFC3', true: '#87566A' }}
        thumbColor="#FFFFFF"
      />
    </View>
  );
}

function PickerRow({ label, options, selected, onSelect }: {
  label: string;
  options: { value: string; label: string }[];
  selected: string;
  onSelect: (v: string) => void;
}) {
  return (
    <View style={styles.pickerBlock}>
      <Text style={styles.rowLabel}>{label}</Text>
      <View style={styles.pickerOptions}>
        {options.map(o => (
          <TouchableOpacity
            key={o.value}
            style={[styles.pill, selected === o.value && styles.pillActive]}
            onPress={() => onSelect(o.value)}
          >
            <Text style={[styles.pillText, selected === o.value && styles.pillTextActive]}>{o.label}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}

function PatientEditModal({
  isSaving,
  onChange,
  onClose,
  onSave,
  patient,
}: {
  isSaving: boolean;
  onChange: (patient: PatientForm | null) => void;
  onClose: () => void;
  onSave: () => void;
  patient: PatientForm | null;
}) {
  if (!patient) return null;

  const selectedTopics = normalizeKeyTopics(patient.keyTopics);
  const showOtherTopicText = selectedTopics.includes('others') || Boolean(patient.keyTopicsOtherText?.trim());
  const update = (values: Partial<PatientForm>) => onChange({ ...patient, ...values });
  const toggleTopic = (topic: KeyTopic) => {
    const current = selectedTopics;
    update({
      keyTopics: current.includes(topic)
        ? current.filter((item) => item !== topic)
        : [...current, topic],
    });
  };

  return (
    <Modal animationType="slide" transparent visible onRequestClose={onClose}>
      <View style={styles.modalBackdrop}>
        <View style={styles.modalSheet}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Edit loved one</Text>
            <TouchableOpacity onPress={onClose} style={styles.iconButton}>
              <Feather name="x" size={20} color="#87566A" />
            </TouchableOpacity>
          </View>
          <ScrollView contentContainerStyle={styles.modalContent}>
            <ModalInput label="Name" value={patient.name} onChangeText={(name) => update({ name })} />
            <ModalInput label="Phone number" value={patient.phoneNumber} onChangeText={(phoneNumber) => update({ phoneNumber })} keyboardType="phone-pad" />
            <ModalInput label="Age" value={patient.age} onChangeText={(age) => update({ age })} keyboardType="numeric" />
            <ModalInput label="Usual wake time" value={patient.usualWakeTime} onChangeText={(usualWakeTime) => update({ usualWakeTime })} />
            <ModalPicker label="Gender" options={GENDER_OPTIONS} selected={patient.gender} onSelect={(gender) => update({ gender })} />
            <ModalPicker label="Preferred language" options={LANGUAGE_OPTIONS} selected={patient.preferredLanguage} onSelect={(preferredLanguage) => update({ preferredLanguage })} />
            <ModalInput
              label="Speech or hearing conditions"
              value={patient.speechOrHearingConditions}
              onChangeText={(speechOrHearingConditions) => update({ speechOrHearingConditions })}
              multiline
              placeholder="Optional"
            />
            <ModalPhotoInput photoUrl={patient.photoUrl || ''} onChange={(photoUrl) => update({ photoUrl })} />
            <Text style={styles.modalLabel}>Key topics they enjoy</Text>
            <View style={styles.pickerOptions}>
              {TOPIC_OPTIONS.map((topic) => {
                const selected = isTopicSelected(patient.keyTopics, topic.value);
                return (
                  <TouchableOpacity
                    key={topic.value}
                    onPress={() => toggleTopic(topic.value)}
                    style={[styles.pill, selected && styles.pillActive]}
                  >
                    <Text style={[styles.pillText, selected && styles.pillTextActive]}>
                      {topic.label}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>
            {showOtherTopicText ? (
              <ModalInput
                label="Other topic"
                value={patient.keyTopicsOtherText}
                onChangeText={(keyTopicsOtherText) => update({ keyTopicsOtherText })}
              />
            ) : null}
            <TouchableOpacity
              disabled={isSaving}
              onPress={onSave}
              style={[styles.saveBtn, styles.modalSaveBtn, isSaving && styles.saveBtnDisabled]}
            >
              {isSaving ? <ActivityIndicator color="#FFFFFF" /> : <Text style={styles.saveBtnText}>Save profile</Text>}
            </TouchableOpacity>
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
}

function ModalPhotoInput({ photoUrl, onChange }: { photoUrl: string; onChange: (value: string) => void }) {
  async function pickImage() {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Photo access needed', 'Allow photo library access to choose a profile photo.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      aspect: [1, 1],
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.7,
    });

    if (!result.canceled && result.assets[0]?.uri) {
      onChange(result.assets[0].uri);
    }
  }

  return (
    <View style={styles.modalField}>
      <Text style={styles.modalLabel}>Photo</Text>
      <View style={styles.modalPhotoBox}>
        {photoUrl ? (
          <Image source={{ uri: photoUrl }} style={styles.modalPhotoPreview} />
        ) : (
          <View style={styles.modalPhotoPlaceholder}>
            <Feather name="image" size={20} color="#A69C92" />
            <Text style={styles.modalPhotoPlaceholderText}>No photo selected</Text>
          </View>
        )}
        {Platform.OS === 'web' ? (
          <View style={styles.modalWebFileInput}>
            {React.createElement('input', {
                accept: 'image/*',
                type: 'file',
                onChange: (event: { target?: { files?: FileList | null } }) => {
                  const file = event.target?.files?.[0];
                  if (!file) return;

                  const reader = new FileReader();
                  reader.onload = () => {
                    if (typeof reader.result === 'string') {
                      onChange(reader.result);
                    }
                  };
                  reader.readAsDataURL(file);
                },
              })}
          </View>
        ) : null}
        {Platform.OS !== 'web' ? (
          <TouchableOpacity activeOpacity={0.82} onPress={() => void pickImage()} style={styles.modalPhotoButton}>
            <Feather name="upload" size={15} color="#FFFFFF" />
            <Text style={styles.modalPhotoButtonText}>{photoUrl ? 'Change photo' : 'Choose photo'}</Text>
          </TouchableOpacity>
        ) : null}
        {photoUrl ? (
          <TouchableOpacity activeOpacity={0.82} onPress={() => onChange('')} style={styles.modalClearPhotoButton}>
            <Text style={styles.modalClearPhotoText}>Remove photo</Text>
          </TouchableOpacity>
        ) : null}
      </View>
    </View>
  );
}

function ModalInput({
  keyboardType = 'default',
  label,
  multiline = false,
  onChangeText,
  placeholder,
  value,
}: {
  keyboardType?: 'default' | 'phone-pad' | 'numeric';
  label: string;
  multiline?: boolean;
  onChangeText: (value: string) => void;
  placeholder?: string;
  value: string;
}) {
  return (
    <View style={styles.modalField}>
      <Text style={styles.modalLabel}>{label}</Text>
      <TextInput
        keyboardType={keyboardType}
        multiline={multiline}
        onChangeText={onChangeText}
        placeholder={placeholder}
        style={[styles.modalInput, multiline && styles.modalTextArea]}
        value={value}
      />
    </View>
  );
}

function ModalPicker<T extends string>({
  label,
  onSelect,
  options,
  selected,
}: {
  label: string;
  onSelect: (value: T) => void;
  options: { value: T; label: string }[];
  selected: string;
}) {
  return (
    <View style={styles.modalField}>
      <Text style={styles.modalLabel}>{label}</Text>
      <View style={styles.pickerOptions}>
        {options.map((option) => (
          <TouchableOpacity
            key={option.value}
            onPress={() => onSelect(option.value)}
            style={[styles.pill, selected === option.value && styles.pillActive]}
          >
            <Text style={[styles.pillText, selected === option.value && styles.pillTextActive]}>
              {option.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F8F3EC' },
  scroll: { flex: 1 },
  content: { paddingBottom: 60 },
  titleRow: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 8,
    gap: 12,
  },
  pageTitle: { fontSize: 26, fontWeight: '500', color: '#2B2522', fontFamily: 'Georgia' },
  loadingCard: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 16,
    borderWidth: 1,
    gap: 8,
    marginHorizontal: 20,
    marginTop: 20,
    padding: 24,
  },
  loadingTitle: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 20, fontWeight: '500' },
  loadingText: { color: '#756C64', fontSize: 14, lineHeight: 20, textAlign: 'center' },
  sectionHeader: {
    fontSize: 12,
    fontWeight: '600',
    color: '#A69C92',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    paddingHorizontal: 20,
    paddingTop: 28,
    paddingBottom: 8,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#F3EDE6',
  },
  inputRow: {
    backgroundColor: '#FFFFFF',
    borderBottomColor: '#F3EDE6',
    borderBottomWidth: 1,
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  rowLabel: { color: '#2B2522', flexShrink: 0, fontSize: 15, fontWeight: '500' },
  rowValue: {
    color: '#A69C92',
    flex: 1,
    fontSize: 15,
    lineHeight: 20,
    marginLeft: 12,
    minWidth: 0,
    textAlign: 'right',
  },
  inlineInput: {
    borderColor: '#E7DED2',
    borderRadius: 10,
    borderWidth: 1,
    color: '#2B2522',
    fontSize: 15,
    marginTop: 8,
    paddingHorizontal: 12,
    paddingVertical: 9,
  },
  rowLeft: { alignItems: 'center', flex: 1, flexDirection: 'row', gap: 10, minWidth: 0 },
  rowRight: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  patientAvatar: {
    alignItems: 'center',
    backgroundColor: '#EEE7DE',
    borderRadius: 17,
    height: 34,
    justifyContent: 'center',
    overflow: 'hidden',
    width: 34,
  },
  patientAvatarImage: { height: '100%', width: '100%' },
  patientAvatarText: { color: '#87566A', fontFamily: 'Georgia', fontSize: 14, fontWeight: '600' },
  pickerBlock: {
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#F3EDE6',
  },
  pickerOptions: { flexDirection: 'row', gap: 8, marginTop: 10, flexWrap: 'wrap' },
  pill: {
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: 999,
    backgroundColor: '#F4F0EA',
    borderWidth: 1,
    borderColor: '#E7DED2',
  },
  pillActive: { backgroundColor: '#87566A', borderColor: '#87566A' },
  pillText: { fontSize: 13, color: '#756C64' },
  pillTextActive: { color: '#FFFFFF', fontWeight: '600' },
  saveBtn: {
    alignItems: 'center',
    backgroundColor: '#87566A',
    justifyContent: 'center',
    marginHorizontal: 20,
    marginTop: 14,
    minHeight: 46,
    borderRadius: 12,
  },
  saveBtnDisabled: { opacity: 0.7 },
  saveBtnText: { color: '#FFFFFF', fontSize: 15, fontWeight: '700' },
  logoutBtn: {
    margin: 20,
    marginTop: 36,
    padding: 16,
    backgroundColor: '#FFFFFF',
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#D8CFC3',
    alignItems: 'center',
  },
  logoutText: { color: '#87566A', fontSize: 15, fontWeight: '600' },
  modalBackdrop: {
    backgroundColor: 'rgba(43,37,34,0.28)',
    flex: 1,
    justifyContent: 'flex-end',
  },
  modalSheet: {
    backgroundColor: '#F8F3EC',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: '88%',
    overflow: 'hidden',
  },
  modalHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 16,
  },
  modalTitle: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 22, fontWeight: '600' },
  iconButton: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 18,
    borderWidth: 1,
    height: 36,
    justifyContent: 'center',
    width: 36,
  },
  modalContent: { padding: 20, paddingBottom: 32 },
  modalField: { marginBottom: 14 },
  modalLabel: { color: '#2B2522', fontSize: 14, fontWeight: '700', marginBottom: 8 },
  modalInput: {
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 10,
    borderWidth: 1,
    color: '#2B2522',
    fontSize: 15,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  modalTextArea: { minHeight: 82, textAlignVertical: 'top' },
  modalPhotoBox: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 12,
    borderWidth: 1,
    gap: 10,
    padding: 14,
  },
  modalPhotoPreview: {
    borderRadius: 12,
    height: 120,
    width: 120,
  },
  modalPhotoPlaceholder: {
    alignItems: 'center',
    backgroundColor: '#F4F0EA',
    borderRadius: 12,
    gap: 6,
    height: 120,
    justifyContent: 'center',
    width: 120,
  },
  modalPhotoPlaceholderText: { color: '#A69C92', fontSize: 12, fontWeight: '600' },
  modalPhotoButton: {
    alignItems: 'center',
    backgroundColor: '#87566A',
    borderRadius: 10,
    flexDirection: 'row',
    gap: 8,
    justifyContent: 'center',
    minHeight: 42,
    paddingHorizontal: 14,
    width: '100%',
  },
  modalPhotoButtonText: { color: '#FFFFFF', fontSize: 14, fontWeight: '700' },
  modalWebFileInput: {
    maxWidth: '100%',
    overflow: 'hidden',
    width: '100%',
  },
  modalClearPhotoButton: { paddingVertical: 4 },
  modalClearPhotoText: { color: '#87566A', fontSize: 13, fontWeight: '700' },
  modalSaveBtn: { marginHorizontal: 0, marginTop: 8 },
});
