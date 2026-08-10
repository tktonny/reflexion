import { Redirect, useLocalSearchParams } from 'expo-router';
import React from 'react';

/** Legacy Settings deep-link bridge. Generic section screens are no longer part of the route tree. */
const ROUTES: Record<string, string> = {
  accessibility: '/settings/language',
  'care-circle': '/settings/household',
  'loved-ones': '/settings/household',
  support: '/settings/help',
  subscription: '/settings/subscription',
  away: '/settings/away',
  about: '/settings/about',
  privacy: '/settings/privacy',
  research: '/settings/research',
  routines: '/settings/routines',
  consent: '/settings/consent',
};

export default function LegacySettingsBridge() {
  const { section } = useLocalSearchParams<{ section?: string }>();
  return <Redirect href={ROUTES[section || ''] || '/(tabs)/settings'} />;
}
