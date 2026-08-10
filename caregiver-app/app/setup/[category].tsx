import { Redirect, useLocalSearchParams } from 'expo-router';
import React from 'react';

/**
 * Backwards-compatible deep-link bridge. The latest architecture does not define a generic category
 * detail route; each category opens its canonical reusable screen instead.
 */
const DESTINATIONS: Record<string, string> = {
  household: '/setup/household',
  'pair-device': '/device/select',
  'language-accessibility': '/settings/language',
  routines: '/settings/routines',
  notifications: '/settings/notifications',
  'consent-control': '/settings/consent',
  'research-participation': '/research/overview',
};

export default function SetupCategoryRedirect() {
  const { category } = useLocalSearchParams<{ category?: string }>();
  return <Redirect href={DESTINATIONS[Array.isArray(category) ? category[0] : category || ''] || '/setup'} />;
}
