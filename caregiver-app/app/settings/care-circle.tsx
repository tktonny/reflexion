import { Redirect } from 'expo-router';
import React from 'react';

/** The former Care Circle route is outside the current architecture. Keep old links safe. */
export default function ObsoleteCareCircleBridge() {
  return <Redirect href="/settings/household" />;
}
