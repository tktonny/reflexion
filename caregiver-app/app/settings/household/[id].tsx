import { useLocalSearchParams } from 'expo-router';
import React from 'react';
import { LovedOneProfileScreen } from '../../../src/screens/LovedOneProfileScreen';

// LovedOneProfileScreen renders the shared ScreenLayout.
export default function EditLovedOneSettingsScreen() {
  const { id } = useLocalSearchParams<{ id?: string }>();
  return <LovedOneProfileScreen mode="settings" patientId={Array.isArray(id) ? id[0] : id} />;
}
