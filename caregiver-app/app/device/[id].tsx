import { Redirect, useLocalSearchParams } from 'expo-router';
import React from 'react';

/** The old parameterised device picker is now a bridge into the canonical pairing flow. */
export default function DeviceRedirect() {
  const { id } = useLocalSearchParams<{ id: string }>();
  return <Redirect href={id ? `/device/${id}/pairing` : '/device/select'} />;
}
