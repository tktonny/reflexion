import { Feather } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Alert, Image, Platform, StyleSheet, Text, View } from 'react-native';

import { useCaregiver } from '../architecture/CaregiverContext';
import { AppHeader, PrimaryButton, ScreenLayout, TertiaryButton } from '../components/AppUI';
import { Field, PhoneField, SelectField } from '../components/Field';
import { MotionPressable } from '../components/Motion';
import { createLovedOneV1, getCarePlanV1, listPatientRecordsV1, updatePatientV1, type V1PatientRecord } from '../lib/v1Caregiver';
import { normalizePhone, validatePhone } from '../lib/authValidation';
import { colors, fontFamily, fontSize, radius, spacing } from '../theme';

type ProfileMode = 'setup' | 'settings';
type Gender = 'female' | 'male' | 'other';

const GENDER_OPTIONS = [
  { value: 'female', label: 'Female' },
  { value: 'male', label: 'Male' },
  { value: 'other', label: 'Other' },
] as const;

const RELATIONSHIP_OPTIONS = ['Mum', 'Dad', 'Grandparent', 'Partner', 'Other'].map((value) => ({ value, label: value }));

export function LovedOneProfileScreen({ mode = 'setup', patientId }: { mode?: ProfileMode; patientId?: string }) {
  const router = useRouter();
  const { setSetupStatus } = useCaregiver();
  const editing = Boolean(patientId);
  const [person, setPerson] = useState<V1PatientRecord | null>(null);
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState<Gender>('female');
  const [relationship, setRelationship] = useState('Mum');
  const [countryCode, setCountryCode] = useState('+65');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [photoUrl, setPhotoUrl] = useState('');
  const [emergencyContact, setEmergencyContact] = useState('');
  const [livingArrangement, setLivingArrangement] = useState('');
  const [loading, setLoading] = useState(editing);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [fieldError, setFieldError] = useState('');

  useEffect(() => {
    if (!patientId) return;
    void listPatientRecordsV1().then(async (people) => {
      const next = people.find((item) => item.patientId === patientId) || null;
      setPerson(next);
      if (next) {
        setName(next.displayName);
        setAge(next.profile.age ? String(next.profile.age) : '');
        setGender(next.profile.gender || 'female');
        const storedPhone = next.profile.phoneNumber || '';
        const storedCountry = next.profile.phoneCountryCode || (storedPhone.startsWith('+65') ? '+65' : '+65');
        setCountryCode(storedCountry);
        setPhoneNumber(storedPhone.startsWith(storedCountry) ? storedPhone.slice(storedCountry.length) : storedPhone);
        setRelationship(next.profile.relationship || 'Mum');
        setPhotoUrl(next.profile.photoUrl || '');
        const legacyPlan = await getCarePlanV1(next.patientId).catch(() => null);
        const notes = legacyPlan?.safetyNotes || '';
        setEmergencyContact(next.profile.emergencyContact || notes.match(/Emergency contact:\s*([^\n]+)/i)?.[1] || '');
        setLivingArrangement(next.profile.livingArrangement || notes.match(/Living arrangement:\s*([^\n]+)/i)?.[1] || '');
      }
    }).catch(() => setError('We could not load this loved one. Check your connection and try again.')).finally(() => setLoading(false));
  }, [patientId]);

  async function pickFromLibrary() {
    if (Platform.OS === 'web') return;
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Photo access needed', permission.canAskAgain === false ? 'Photo access is blocked. Open phone settings to allow photo access, then try again.' : 'Allow photo library access to choose a profile photo.');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({ allowsEditing: true, aspect: [1, 1], mediaTypes: ImagePicker.MediaTypeOptions.Images, quality: 0.7 });
    if (!result.canceled && result.assets[0]?.uri) setPhotoUrl(result.assets[0].uri);
  }

  async function takePhoto() {
    if (Platform.OS === 'web') return;
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Camera access needed', permission.canAskAgain === false ? 'Camera access is blocked. Open phone settings to allow camera access, then try again.' : 'Allow camera access to take a profile photo.');
      return;
    }
    const result = await ImagePicker.launchCameraAsync({ allowsEditing: true, aspect: [1, 1], mediaTypes: ImagePicker.MediaTypeOptions.Images, quality: 0.7 });
    if (!result.canceled && result.assets[0]?.uri) setPhotoUrl(result.assets[0].uri);
  }

  function openPhotoActions() {
    Alert.alert(photoUrl ? 'Update photo' : 'Add a photo', 'Choose how to add the loved one’s photo.', [
      { text: 'Take photo', onPress: () => { void takePhoto(); } },
      { text: 'Choose from photo library', onPress: () => { void pickFromLibrary(); } },
      ...(photoUrl ? [{ text: 'Remove photo', style: 'destructive' as const, onPress: () => setPhotoUrl('') }] : []),
      { text: 'Cancel', style: 'cancel' as const },
    ]);
  }

  const save = async () => {
    const trimmedName = name.trim();
    if (!trimmedName) { setFieldError('Enter the name they like to be called.'); return; }
    const parsedAge = age.trim() ? Number(age) : null;
    if (parsedAge !== null && (!Number.isInteger(parsedAge) || parsedAge < 1 || parsedAge > 130)) { setFieldError('Enter an age between 1 and 130.'); return; }
    if (phoneNumber.trim()) {
      const phoneError = validatePhone(countryCode, phoneNumber);
      if (phoneError) { setFieldError(phoneError); return; }
    }
    setSaving(true); setError(''); setFieldError('');
    try {
      const isRemotePhoto = /^https?:\/\//i.test(photoUrl.trim());
      const retainedPhoto = photoUrl.trim() ? (isRemotePhoto ? photoUrl.trim() : person?.profile.photoUrl || null) : null;
      const profile = { age: parsedAge, gender, photoUrl: retainedPhoto, phoneCountryCode: phoneNumber.trim() ? countryCode : null, phoneNumber: phoneNumber.trim() ? normalizePhone(countryCode, phoneNumber) : null, relationship: relationship.trim() || null, emergencyContact: emergencyContact.trim() || null, livingArrangement: livingArrangement.trim() || null } as const;
      if (person) {
        await updatePatientV1(person.patientId, person.version, { displayName: trimmedName, profile });
      } else {
        await createLovedOneV1({ displayName: trimmedName, preferredLanguage: 'English', timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'Asia/Singapore', relationshipType: relationship.trim() || 'caregiver', profile });
      }
      if (mode === 'setup') {
        setSetupStatus('household', 'complete');
        router.replace('/setup/household-review');
      } else {
        router.replace('/settings/household');
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'We could not save this loved one. Check your connection and try again.');
    } finally { setSaving(false); }
  };

  if (loading) return <ScreenLayout scroll={false} contentContainerStyle={styles.loading}><ActivityIndicator color={colors.accent} /></ScreenLayout>;

  return <ScreenLayout contentContainerStyle={styles.content}>
    <AppHeader title={editing ? 'Edit loved-one profile' : 'Loved-One Profile'} onBack={() => router.back()} />
    <Text style={styles.eyebrow}>{editing ? 'Household' : 'Household · 1 of 7'}</Text>
    <Text accessibilityRole="header" style={styles.title}>{editing ? 'Edit loved-one profile' : 'Loved-One Profile'}</Text>
    <Text style={styles.subtitle}>Use the name your loved one prefers. These details help keep updates connected to the right person.</Text>
    <MotionPressable accessibilityLabel={photoUrl ? 'Change loved-one photo' : 'Add a photo'} accessibilityRole="button" haptic="selection" onPress={openPhotoActions} style={styles.photoCircle}>
      {photoUrl ? <Image accessibilityLabel="Selected loved-one photo" source={{ uri: photoUrl }} style={styles.photoImage} /> : <><View style={styles.photoIcon}><Feather color={colors.accent} name="camera" size={27} /></View><Text style={styles.photoText}>Add a photo</Text></>}
    </MotionPressable>
    {photoUrl && !/^https?:\/\//i.test(photoUrl) ? <Text style={styles.photoNote}>This photo is saved as a preview on this device.</Text> : null}
    <Field label="Preferred name" onChangeText={(value) => { setName(value); setFieldError(''); }} placeholder="e.g. Mum" value={name} error={fieldError && !fieldError.toLowerCase().includes('phone') ? fieldError : undefined} />
    <Field label="Age or date of birth" keyboardType="number-pad" onChangeText={(value) => { setAge(value.replace(/\D/g, '')); setFieldError(''); }} placeholder="e.g. 78" value={age} />
    <SelectField label="Gender" onChange={(value) => setGender(value as Gender)} options={[...GENDER_OPTIONS]} value={gender} />
    <SelectField label="Relationship to you" onChange={setRelationship} options={RELATIONSHIP_OPTIONS} value={relationship} />
    <PhoneField countryCode={countryCode} error={fieldError.toLowerCase().includes('phone') ? fieldError : undefined} helperText="Country code and phone number are stored separately." label="Phone number (optional)" onCountryCodeChange={setCountryCode} onPhoneNumberChange={setPhoneNumber} phoneNumber={phoneNumber} />
    <Field label="Emergency contact (optional)" onChangeText={setEmergencyContact} placeholder="Name and phone number" value={emergencyContact} />
    <Field label="Living arrangement (optional)" onChangeText={setLivingArrangement} placeholder="e.g. Lives with family" value={livingArrangement} />
    {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
    {saving ? <ActivityIndicator color={colors.accent} /> : <PrimaryButton label={editing ? 'Save changes' : 'Save and continue'} onPress={() => void save()} />}
    <TertiaryButton label="Set up later" onPress={() => router.replace(mode === 'setup' ? '/(tabs)' : '/settings/household')} />
  </ScreenLayout>;
}

const styles = StyleSheet.create({
  loading: { alignItems: 'center', justifyContent: 'center' },
  content: { gap: spacing.lg, minWidth: 0 },
  eyebrow: { color: colors.accent, fontSize: fontSize.caption, fontWeight: '700', marginTop: spacing.sm },
  title: { color: colors.text.primary, fontFamily: fontFamily.display, fontSize: fontSize.title, fontWeight: '500', lineHeight: 36, minWidth: 0 },
  subtitle: { color: colors.text.secondary, fontSize: fontSize.bodyLarge, lineHeight: 24, minWidth: 0 },
  photoCircle: { alignItems: 'center', alignSelf: 'center', backgroundColor: colors.surface.muted, borderColor: colors.border.default, borderRadius: 78, borderWidth: 1, height: 156, justifyContent: 'center', overflow: 'hidden', width: 156 },
  photoImage: { height: '100%', width: '100%' },
  photoIcon: { alignItems: 'center', backgroundColor: '#E7F3F0', borderRadius: radius.pill, height: 48, justifyContent: 'center', width: 48 },
  photoText: { color: colors.text.primary, fontSize: fontSize.body, fontWeight: '700', lineHeight: 20, marginTop: spacing.sm },
  photoNote: { color: colors.text.secondary, fontSize: fontSize.caption, lineHeight: 19, textAlign: 'center' },
  error: { color: colors.error.text, fontSize: fontSize.body, lineHeight: 22, minWidth: 0 },
});
