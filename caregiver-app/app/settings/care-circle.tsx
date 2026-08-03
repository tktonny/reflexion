import { useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Alert, StyleSheet, Text, View } from 'react-native';

import { AppHeader, PrimaryButton, ScreenLayout, SecondaryButton } from '../../src/components/AppUI';
import { Field, PhoneField } from '../../src/components/Field';
import { normalizePhone, validateEmail, validatePhone } from '../../src/lib/authValidation';
import { getCareCircleV1, inviteCaregiverV1, listPatientRecordsV1, revokeCareCircleMemberV1, updateCareCircleMemberV1, type V1CareCircle, type V1CareCircleMember, type V1PatientRecord } from '../../src/lib/v1Caregiver';
import { colors, fontFamily, fontSize, radius, spacing } from '../../src/theme';

const ROLES: V1CareCircleMember['role'][] = ['full-access', 'standard-access', 'view-only', 'custom-access'];

export default function CareCircleSettings() {
  const router = useRouter();
  const [patient, setPatient] = useState<V1PatientRecord | null>(null);
  const [circle, setCircle] = useState<V1CareCircle | null>(null);
  const [invitee, setInvitee] = useState('');
  const [inviteMethod, setInviteMethod] = useState<'email' | 'phone'>('email');
  const [countryCode, setCountryCode] = useState('+65');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [inviteError, setInviteError] = useState('');
  const [role, setRole] = useState<V1CareCircleMember['role']>('view-only');
  const [editing, setEditing] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const load = async (selectedPatient?: V1PatientRecord | null) => { const current = selectedPatient || patient; if (current) setCircle(await getCareCircleV1(current.patientId)); };
  useEffect(() => { void listPatientRecordsV1().then((people) => { const first = people[0] || null; setPatient(first); return first ? getCareCircleV1(first.patientId) : null; }).then((next) => { if (next) setCircle(next); }).catch((cause) => setError(cause instanceof Error ? cause.message : 'Could not load Care Circle.')); }, []);
  const invite = async () => {
    if (!patient) return;
    const contact = inviteMethod === 'email' ? invitee.trim().toLowerCase() : normalizePhone(countryCode, phoneNumber);
    const validation = inviteMethod === 'email' ? validateEmail(invitee) : validatePhone(countryCode, phoneNumber, true);
    if (validation) { setInviteError(validation); return; }
    setBusy(true); setError(''); setInviteError('');
    try { await inviteCaregiverV1(patient.patientId, { emailOrPhone: contact, role }); setInvitee(''); setPhoneNumber(''); await load(); Alert.alert('Invitation queued', 'The invitation is saved. Email or SMS delivery requires the configured provider.'); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'We could not send the invitation. Check your connection and try again.'); }
    finally { setBusy(false); }
  };
  const edit = async (member: V1CareCircleMember, nextRole: V1CareCircleMember['role']) => { if (!patient) return; setBusy(true); try { await updateCareCircleMemberV1(patient.patientId, member, { role: nextRole }); setEditing(null); await load(); } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not update permissions.'); } finally { setBusy(false); } };
  const revoke = (member: V1CareCircleMember) => { if (!patient) return; Alert.alert('Revoke access?', 'This caregiver will no longer see updates for this loved one.', [{ text: 'Cancel', style: 'cancel' }, { text: 'Revoke', style: 'destructive', onPress: () => void (async () => { setBusy(true); try { await revokeCareCircleMemberV1(patient.patientId, member.memberId); await load(); } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not revoke access.'); } finally { setBusy(false); } })() }]); };
  const items = circle ? [...circle.members, ...circle.invitations] : [];
  return <ScreenLayout contentContainerStyle={styles.content}><AppHeader title="Care Circle" onBack={() => router.back()} /><Text accessibilityRole="header" style={styles.title}>Care Circle</Text><Text style={styles.copy}>Invite trusted people, then decide what each person can see and do.</Text>{error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}{!patient || !circle ? <ActivityIndicator color={colors.accent} /> : <><View style={styles.card}><Text style={styles.label}>People connected to {patient.displayName}</Text>{items.length ? items.map((member) => <View key={member.memberId} style={styles.member}><View style={styles.memberCopy}><Text style={styles.memberName}>{member.name || member.invitee || 'Caregiver'}</Text><Text style={styles.help}>{member.kind === 'invitation' ? 'Invitation pending' : member.email || member.phoneNumber || 'Caregiver'} · {member.role}</Text></View><SecondaryButton label={editing === member.memberId ? 'Close' : 'Edit'} onPress={() => setEditing(editing === member.memberId ? null : member.memberId)} /><SecondaryButton label="Revoke" onPress={() => revoke(member)} />{editing === member.memberId ? <View style={styles.roleList}>{ROLES.map((option) => <SecondaryButton key={option} label={`${option === member.role ? '✓ ' : ''}${option}`} onPress={() => void edit(member, option)} />)}</View> : null}</View>) : <Text style={styles.help}>No other caregivers yet.</Text>}</View><View style={styles.card}><Text style={styles.label}>Invite a caregiver</Text><View style={styles.methodRow}><SecondaryButton label={inviteMethod === 'email' ? '✓ Email' : 'Email'} onPress={() => { setInviteMethod('email'); setInviteError(''); }} /><SecondaryButton label={inviteMethod === 'phone' ? '✓ Phone' : 'Phone'} onPress={() => { setInviteMethod('phone'); setInviteError(''); }} /></View>{inviteMethod === 'email' ? <Field error={inviteError} label="Email address" value={invitee} onChangeText={(value) => { setInvitee(value); setInviteError(''); }} keyboardType="email-address" /> : <PhoneField countryCode={countryCode} error={inviteError} label="Phone number" onCountryCodeChange={(value) => { setCountryCode(value); setInviteError(''); }} onPhoneNumberChange={(value) => { setPhoneNumber(value); setInviteError(''); }} phoneNumber={phoneNumber} />}<Text style={styles.help}>Role</Text>{ROLES.map((option) => <SecondaryButton key={option} label={`${role === option ? '✓ ' : ''}${option}`} onPress={() => setRole(option)} />)}{busy ? <ActivityIndicator color={colors.accent} /> : <PrimaryButton label="Send invitation" onPress={() => void invite()} />}</View></>}</ScreenLayout>;
}

const styles = StyleSheet.create({ content: { gap: spacing.lg }, title: { color: colors.text.primary, flexShrink: 1, fontFamily: fontFamily.display, fontSize: fontSize.title, lineHeight: 36, marginTop: spacing.lg }, copy: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 }, error: { color: colors.error.text, flexShrink: 1, fontSize: fontSize.body, lineHeight: 22 }, card: { backgroundColor: colors.surface.card, borderColor: colors.border.default, borderRadius: radius.lg, borderWidth: 1, gap: spacing.md, padding: spacing.lg }, label: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, fontWeight: '700' }, member: { alignItems: 'center', borderBottomColor: colors.border.subtle, borderBottomWidth: 1, flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, paddingVertical: spacing.md }, memberCopy: { flex: 1, minWidth: 160 }, memberName: { color: colors.text.primary, flexShrink: 1, fontSize: fontSize.bodyLarge, fontWeight: '700' }, help: { color: colors.text.secondary, flexShrink: 1, fontSize: fontSize.body, lineHeight: 21 }, roleList: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, width: '100%' }, methodRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm } });
