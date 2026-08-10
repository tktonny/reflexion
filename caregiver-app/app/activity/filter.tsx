import { Redirect } from 'expo-router';
import React from 'react';

/** Filters are an Activity sheet, not a separate canonical full-screen route. */
export default function ActivityFilterBridge() {
  return <Redirect href="/(tabs)/activity" />;
}
