import { Redirect, useLocalSearchParams } from 'expo-router';
import React from 'react';

/** Legacy deep-link bridge. Dashboard views are explicit canonical routes now. */
export default function LegacyLovedOneViewBridge() {
  const { id, view } = useLocalSearchParams<{ id: string; view: string }>();
  const destination = view === 'sessions'
    ? `/loved-one/${id}/sessions`
    : view === 'weekly-summary'
      ? `/loved-one/${id}/weekly-summary`
      : view === 'trends'
        ? `/loved-one/${id}/trends`
        : view === 'history'
          ? `/loved-one/${id}/history`
          : `/loved-one/${id}/export`;
  return <Redirect href={destination} />;
}
