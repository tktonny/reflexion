import { Feather } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import React from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  Modal,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { colors, fontFamily, fontSize, radius, spacing, scaleSize, MIN_TOUCH_TARGET } from '../../theme';
import { isTopicSelected, normalizeKeyTopics } from './helpers';
import { PILL_HIT_SLOP, pillStyles } from './optionPills';
import type { Gender, KeyTopic, Language, PatientForm } from './types';

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

export function PatientEditModal({
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
        {/* Keeps a screen reader inside the sheet instead of wandering into the settings list behind it. */}
        <View accessibilityViewIsModal style={styles.modalSheet}>
          <View style={styles.modalHeader}>
            <Text accessibilityRole="header" maxFontSizeMultiplier={1.3} style={styles.modalTitle}>Edit loved one</Text>
            <TouchableOpacity
              accessibilityLabel="Close without saving"
              accessibilityRole="button"
              // Stays a 36pt circle to match the header; hitSlop carries it past the 44pt floor.
              hitSlop={PILL_HIT_SLOP}
              onPress={onClose}
              style={styles.iconButton}
            >
              <Feather name="x" size={scaleSize(20)} color={colors.accent} />
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
            <View style={pillStyles.pickerOptions}>
              {TOPIC_OPTIONS.map((topic) => {
                const selected = isTopicSelected(patient.keyTopics, topic.value);
                return (
                  <TouchableOpacity
                    key={topic.value}
                    accessibilityLabel={`Key topic, ${topic.label}`}
                    accessibilityRole="button"
                    accessibilityState={{ selected }}
                    hitSlop={PILL_HIT_SLOP}
                    onPress={() => toggleTopic(topic.value)}
                    style={[pillStyles.pill, selected && pillStyles.pillActive]}
                  >
                    <Text style={[pillStyles.pillText, selected && pillStyles.pillTextActive]}>
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
              // Spinner replaces the text while saving, so the label cannot come from its children.
              accessibilityLabel={isSaving ? 'Saving profile' : 'Save profile'}
              accessibilityRole="button"
              accessibilityState={{ busy: isSaving, disabled: isSaving }}
              disabled={isSaving}
              onPress={onSave}
              style={[styles.saveBtn, styles.modalSaveBtn, isSaving && styles.saveBtnDisabled]}
            >
              {isSaving ? <ActivityIndicator color={colors.text.onAccent} /> : <Text style={styles.saveBtnText}>Save profile</Text>}
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
            <Feather name="image" size={20} color={colors.text.tertiary} />
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
          <TouchableOpacity
            accessibilityLabel={photoUrl ? 'Change photo' : 'Choose photo'}
            accessibilityRole="button"
            activeOpacity={0.82}
            onPress={() => void pickImage()}
            style={styles.modalPhotoButton}
          >
            <Feather name="upload" size={15} color={colors.text.onAccent} />
            <Text style={styles.modalPhotoButtonText}>{photoUrl ? 'Change photo' : 'Choose photo'}</Text>
          </TouchableOpacity>
        ) : null}
        {photoUrl ? (
          <TouchableOpacity
            accessibilityLabel="Remove photo"
            accessibilityRole="button"
            activeOpacity={0.82}
            // A quiet text link by design; hitSlop reaches 44pt without loosening the photo card spacing.
            hitSlop={{ bottom: 10, left: 12, right: 12, top: 10 }}
            onPress={() => onChange('')}
            style={styles.modalClearPhotoButton}
          >
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
        accessibilityLabel={label}
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
      <View style={pillStyles.pickerOptions}>
        {options.map((option) => (
          <TouchableOpacity
            key={option.value}
            accessibilityLabel={`${label}, ${option.label}`}
            accessibilityRole="button"
            accessibilityState={{ selected: selected === option.value }}
            hitSlop={PILL_HIT_SLOP}
            onPress={() => onSelect(option.value)}
            style={[pillStyles.pill, selected === option.value && pillStyles.pillActive]}
          >
            <Text style={[pillStyles.pillText, selected === option.value && pillStyles.pillTextActive]}>
              {option.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  saveBtn: {
    alignItems: 'center',
    backgroundColor: colors.accent,
    justifyContent: 'center',
    marginHorizontal: spacing.xl,
    marginTop: spacing.md,
    minHeight: MIN_TOUCH_TARGET,
    borderRadius: radius.md,
  },
  saveBtnDisabled: { opacity: 0.7 },
  saveBtnText: { color: colors.text.onAccent, fontSize: fontSize.subheading, fontWeight: '700' },
  modalBackdrop: {
    backgroundColor: 'rgba(43,37,34,0.28)',
    flex: 1,
    justifyContent: 'flex-end',
  },
  modalSheet: {
    backgroundColor: colors.surface.page,
    borderTopLeftRadius: scaleSize(20),
    borderTopRightRadius: scaleSize(20),
    maxHeight: '88%',
    overflow: 'hidden',
  },
  modalHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.lg,
  },
  modalTitle: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: scaleSize(22), flexShrink: 1, fontWeight: '600' },
  iconButton: {
    alignItems: 'center',
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: scaleSize(18),
    borderWidth: 1,
    height: scaleSize(36),
    justifyContent: 'center',
    width: scaleSize(36),
  },
  modalContent: { padding: spacing.xl, paddingBottom: scaleSize(32) },
  modalField: { marginBottom: scaleSize(14) },
  modalLabel: { color: colors.text.primary, fontSize: fontSize.bodyLarge, fontWeight: '700', marginBottom: spacing.sm },
  modalInput: {
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: radius.md,
    borderWidth: 1,
    color: colors.text.primary,
    fontSize: fontSize.subheading,
    paddingHorizontal: spacing.md,
    paddingVertical: scaleSize(10),
  },
  modalTextArea: { minHeight: scaleSize(82), textAlignVertical: 'top' },
  modalPhotoBox: {
    alignItems: 'center',
    backgroundColor: colors.surface.card,
    borderColor: colors.border.default,
    borderRadius: radius.md,
    borderWidth: 1,
    gap: scaleSize(10),
    padding: scaleSize(14),
  },
  modalPhotoPreview: {
    borderRadius: 12,
    height: scaleSize(120),
    width: scaleSize(120),
  },
  modalPhotoPlaceholder: {
    alignItems: 'center',
    backgroundColor: colors.surface.muted,
    borderRadius: 12,
    gap: 6,
    // minHeight, not height: the caption inside is text and clipped at large font sizes.
    minHeight: scaleSize(120),
    justifyContent: 'center',
    width: scaleSize(120),
  },
  modalPhotoPlaceholderText: { color: colors.text.tertiary, fontSize: fontSize.caption, fontWeight: '600', textAlign: 'center' },
  modalPhotoButton: {
    alignItems: 'center',
    backgroundColor: colors.accent,
    borderRadius: radius.md,
    flexDirection: 'row',
    gap: spacing.sm,
    justifyContent: 'center',
    minHeight: MIN_TOUCH_TARGET,
    paddingHorizontal: scaleSize(14),
    width: '100%',
  },
  modalPhotoButtonText: { color: colors.text.onAccent, fontSize: fontSize.bodyLarge, fontWeight: '700' },
  modalWebFileInput: {
    maxWidth: '100%',
    overflow: 'hidden',
    width: '100%',
  },
  modalClearPhotoButton: { paddingVertical: spacing.xs },
  modalClearPhotoText: { color: colors.accent, fontSize: fontSize.body, fontWeight: '700' },
  modalSaveBtn: { marginHorizontal: 0, marginTop: spacing.sm },
});
