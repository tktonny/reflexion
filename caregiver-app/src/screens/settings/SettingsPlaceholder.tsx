import React from 'react';
import { EmptyState, ErrorState, LoadingState } from '../../components/ScreenState';
import type { SettingsState } from './types';

/**
 * Never renders the server's error text. Anything technical is a connection matter, not news about the
 * person — the same rule the status screens follow.
 */
export function SettingsPlaceholder({ onRetry, state }: { onRetry: () => void; state: SettingsState }) {
  if (state === 'signed-out') {
    return (
      <EmptyState
        icon="lock"
        title="Sign in again to see your settings"
        message="Your settings are kept private to your account. Signing out and back in will reconnect them."
      />
    );
  }

  if (state === 'loading') {
    return <LoadingState message="We are loading your settings." />;
  }

  if (state === 'failed') {
    return <ErrorState onRetry={onRetry} />;
  }

  // The endpoint answers 200 with an empty record when nothing is saved for this account yet, so an empty
  // answer is not a failure — and not a blank form either: saving one would be rejected.
  return (
    <EmptyState
      icon="sliders"
      title="Your settings are not ready yet"
      message="Once your account finishes setting up, your preferences and loved ones will appear here."
      onRetry={onRetry}
      retryLabel="Check again"
    />
  );
}
