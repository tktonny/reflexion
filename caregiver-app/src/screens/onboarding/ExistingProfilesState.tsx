import React from 'react';
import { EmptyState, ErrorState, LoadingState } from '../../components/ScreenState';

/**
 * The existing-profile count is a supporting query, and its failure used to reach the console only. Four
 * situations need telling apart here, because a silent failure quietly renumbers the form: a caregiver who
 * already has two loved ones is offered "Add elderly profile 1", and the heading they read while typing is
 * simply wrong. A genuine zero is the normal first-run case and needs no notice, so it renders nothing.
 * As everywhere else in the app, the failure copy is ours — never the server's words.
 */
export function ExistingProfilesState({
  hasSession,
  isLoading,
  hasError,
  onRetry,
}: {
  hasSession: boolean;
  isLoading: boolean;
  hasError: boolean;
  onRetry: () => void;
}) {
  if (!hasSession) {
    return (
      <EmptyState
        compact
        icon="lock"
        title="Sign in again to add a loved one"
        message="Your profiles are kept private to your account. Signing out and back in will reconnect them."
      />
    );
  }

  if (isLoading) {
    return <LoadingState message="Checking the profiles you already have." />;
  }

  if (hasError) {
    return (
      <ErrorState
        compact
        title="We could not check your existing profiles"
        message="This is usually a connection problem, not something about your loved one. You can still fill this in — only the profile number in the heading may be off."
        onRetry={onRetry}
      />
    );
  }

  return null;
}
