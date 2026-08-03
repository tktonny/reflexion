import React from 'react';
import { StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { PhoneField } from '../../components/Field';
import { colors, fontSize, MIN_TOUCH_TARGET, radius, spacing } from '../../theme';
import { fieldStyles, Label, MultiOptionGrid, OptionGrid } from './fields';
import { PhotoInput } from './PhotoInput';
import type { Gender, PatientForm, PreferredLanguage, Topic } from './types';

const GENDER_OPTIONS: { value: Gender; label: string }[] = [
  { value: 'male', label: 'Male' },
  { value: 'female', label: 'Female' },
  { value: 'other', label: 'Other' },
];

const LANGUAGE_OPTIONS: { value: PreferredLanguage; label: string }[] = [
  { value: 'english', label: 'English' },
  { value: 'mandarin', label: 'Mandarin' },
  { value: 'other', label: 'Other' },
];

const TOPIC_OPTIONS: { value: Topic; label: string }[] = [
  { value: 'family', label: 'Family' },
  { value: 'food', label: 'Food' },
  { value: 'travel', label: 'Travel' },
  { value: 'work', label: 'Work' },
  { value: 'others', label: 'Others' },
];

export function ElderlyStep({
  addPatient,
  patient,
  patientIndex,
  patientNumberOffset,
  patients,
  removePatient,
  selectedPatientIndex,
  setSelectedPatientIndex,
  updatePatient,
}: {
  addPatient: () => void;
  patient: PatientForm;
  patientIndex: number;
  patientNumberOffset: number;
  patients: PatientForm[];
  removePatient: (index: number) => void;
  selectedPatientIndex: number;
  setSelectedPatientIndex: (index: number) => void;
  updatePatient: (index: number, updates: Partial<PatientForm>) => void;
}) {
  return (
    <View>
      <View style={styles.patientTabs}>
        {patients.map((item, index) => {
          // Placeholder names are announced too — an untouched tab must not read as an empty button.
          const tabName = item.name.trim() || `Person ${patientNumberOffset + index + 1}`;
          const isSelected = selectedPatientIndex === index;
          return (
            <View key={index} style={[styles.patientTab, isSelected && styles.patientTabActive]}>
              <TouchableOpacity
                accessibilityLabel={tabName}
                accessibilityRole="tab"
                accessibilityState={{ selected: isSelected }}
                onPress={() => setSelectedPatientIndex(index)}
                style={styles.patientTabLabel}
              >
                <Text style={[styles.patientTabText, isSelected && styles.patientTabTextActive]}>
                  {tabName}
                </Text>
              </TouchableOpacity>
              {patients.length > 1 ? (
                <TouchableOpacity
                  accessibilityLabel={`Remove ${tabName}`}
                  accessibilityRole="button"
                  // Deliberately not grown to 44 wide: it sits inside the tab strip, so it gets reach
                  // through hitSlop instead of pushing the tabs onto another line.
                  hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                  onPress={() => removePatient(index)}
                  style={[styles.patientTabRemove, isSelected && styles.patientTabRemoveActive]}
                >
                  <Text
                    accessibilityElementsHidden
                    importantForAccessibility="no"
                    // Tightly boxed glyph inside a 34pt cell; the label above carries the meaning.
                    maxFontSizeMultiplier={1.6}
                    style={[styles.patientTabRemoveText, isSelected && styles.patientTabRemoveTextActive]}
                  >
                    ×
                  </Text>
                </TouchableOpacity>
              ) : null}
            </View>
          );
        })}
        <TouchableOpacity
          accessibilityLabel="Add another elderly profile"
          accessibilityRole="button"
          onPress={addPatient}
          style={styles.addTab}
        >
          <Text style={styles.addTabText}>+ Add</Text>
        </TouchableOpacity>
      </View>

      <Label>Name they like to be called</Label>
      <TextInput
        accessibilityLabel="Name they like to be called"
        onChangeText={(name) => updatePatient(patientIndex, { name })}
        placeholder="e.g. Grandpa Tan"
        placeholderTextColor={colors.placeholder}
        style={fieldStyles.input}
        value={patient.name}
      />

      <PhoneField
        countryCode={splitPhone(patient.phoneNumber).countryCode}
        label="Phone number"
        onCountryCodeChange={(countryCode) => updatePatient(patientIndex, { phoneNumber: `${countryCode}${splitPhone(patient.phoneNumber).phoneNumber}` })}
        onPhoneNumberChange={(phoneNumber) => updatePatient(patientIndex, { phoneNumber: `${splitPhone(patient.phoneNumber).countryCode}${phoneNumber}` })}
        phoneNumber={splitPhone(patient.phoneNumber).phoneNumber}
      />

      <View style={styles.twoCol}>
        <View style={styles.col}>
          <Label>Age</Label>
          <TextInput
            accessibilityLabel="Age"
            keyboardType="number-pad"
            onChangeText={(age) => updatePatient(patientIndex, { age })}
            placeholder="82"
            placeholderTextColor={colors.placeholder}
            style={fieldStyles.input}
            value={patient.age}
          />
        </View>
        <View style={styles.col}>
          <Label>Usual wake time</Label>
          <TextInput
            accessibilityLabel="Usual wake time, for example 07:30"
            onChangeText={(usualWakeTime) => updatePatient(patientIndex, { usualWakeTime })}
            placeholder="07:30"
            placeholderTextColor={colors.placeholder}
            style={fieldStyles.input}
            value={patient.usualWakeTime}
          />
        </View>
      </View>

      <Label>Gender</Label>
      <OptionGrid
        groupLabel="Gender"
        options={GENDER_OPTIONS}
        selected={patient.gender}
        onSelect={(gender) => updatePatient(patientIndex, { gender })}
      />

      <Label>Preferred language</Label>
      <OptionGrid
        groupLabel="Preferred language"
        options={LANGUAGE_OPTIONS}
        selected={patient.preferredLanguage}
        onSelect={(preferredLanguage) => updatePatient(patientIndex, { preferredLanguage })}
      />

      <Label>Speech or hearing conditions</Label>
      <TextInput
        accessibilityLabel="Speech or hearing conditions, optional"
        multiline
        onChangeText={(speechOrHearingConditions) =>
          updatePatient(patientIndex, { speechOrHearingConditions })
        }
        placeholder="Optional"
        placeholderTextColor={colors.placeholder}
        style={[fieldStyles.input, fieldStyles.textArea]}
        value={patient.speechOrHearingConditions}
      />

      <Label>Photo upload</Label>
      <PhotoInput
        photoUrl={patient.photoUrl}
        onChange={(photoUrl) => updatePatient(patientIndex, { photoUrl })}
      />

      <Label>Key topics they enjoy</Label>
      <MultiOptionGrid
        groupLabel="Topic they enjoy"
        options={TOPIC_OPTIONS}
        selected={patient.keyTopics}
        onToggle={(topic) => {
          const isSelected = patient.keyTopics.includes(topic);
          const keyTopics = isSelected
            ? patient.keyTopics.filter((item) => item !== topic)
            : [...patient.keyTopics, topic];
          updatePatient(patientIndex, { keyTopics });
        }}
      />

      {patient.keyTopics.includes('others') ? (
        <>
          <Label>Other topics</Label>
          <TextInput
            accessibilityLabel="Other topics they enjoy"
            onChangeText={(keyTopicsOtherText) => updatePatient(patientIndex, { keyTopicsOtherText })}
            placeholder="Gardening, mahjong, music..."
            placeholderTextColor={colors.placeholder}
            style={fieldStyles.input}
            value={patient.keyTopicsOtherText}
          />
        </>
      ) : null}

      {patients.length > 1 ? (
        <TouchableOpacity
          accessibilityLabel={`Remove ${patient.name.trim() || `Person ${patientNumberOffset + patientIndex + 1}`} from this setup`}
          accessibilityRole="button"
          onPress={() => removePatient(patientIndex)}
          style={styles.removeBtn}
        >
          <Text style={styles.removeBtnText}>Remove this profile</Text>
        </TouchableOpacity>
      ) : null}
    </View>
  );
}

function splitPhone(value: string) {
  const match = value.match(/^(\+\d{1,3})(.*)$/);
  return { countryCode: match?.[1] || '+65', phoneNumber: (match?.[2] || value).replace(/[^0-9\s().-]/g, '') };
}

const styles = StyleSheet.create({
  twoCol: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  col: { flex: 1 },
  patientTabs: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
    marginBottom: 6,
  },
  patientTab: {
    alignItems: 'center',
    backgroundColor: colors.surface.card,
    borderColor: colors.border.strong,
    borderRadius: radius.sm,
    borderWidth: 1,
    flexDirection: 'row',
    overflow: 'hidden',
  },
  // The selected tab inverts to the app's ink; the theme has no dark-surface token, so text.primary is
  // the one that holds this value.
  patientTabActive: {
    backgroundColor: colors.text.primary,
    borderColor: colors.text.primary,
  },
  patientTabText: {
    color: colors.text.secondary,
    fontSize: fontSize.body,
    fontWeight: '800',
  },
  patientTabTextActive: { color: colors.text.onAccent },
  patientTabLabel: {
    justifyContent: 'center',
    minHeight: MIN_TOUCH_TARGET,
    paddingHorizontal: spacing.md,
    paddingVertical: 9,
  },
  patientTabRemove: {
    alignItems: 'center',
    borderLeftColor: colors.border.default,
    borderLeftWidth: 1,
    justifyContent: 'center',
    minHeight: MIN_TOUCH_TARGET,
    // minWidth, not width: at large system text the × glyph would otherwise be clipped by the cell.
    minWidth: 34,
  },
  patientTabRemoveActive: {
    borderLeftColor: 'rgba(255,255,255,0.22)',
  },
  patientTabRemoveText: {
    color: colors.accent,
    fontSize: fontSize.title,
    fontWeight: '800',
    // Pinned: without it the platform default line box shifts the glyph off centre.
    lineHeight: 20,
  },
  patientTabRemoveTextActive: {
    color: colors.text.onAccent,
  },
  addTab: {
    alignItems: 'center',
    backgroundColor: '#EFE7DD',
    borderRadius: radius.sm,
    justifyContent: 'center',
    minHeight: MIN_TOUCH_TARGET,
    paddingHorizontal: spacing.md,
    paddingVertical: 9,
  },
  addTabText: {
    color: colors.accent,
    fontSize: fontSize.body,
    fontWeight: '800',
  },
  removeBtn: {
    alignSelf: 'flex-start',
    justifyContent: 'center',
    marginTop: 18,
    minHeight: MIN_TOUCH_TARGET,
    paddingVertical: spacing.sm,
  },
  removeBtnText: {
    color: colors.accent,
    fontSize: fontSize.body,
    fontWeight: '800',
  },
});
