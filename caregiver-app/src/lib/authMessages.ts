import { LegacyApiError } from './apiClient';

// Caregiver-facing wording for auth failures.
//
// The rule this enforces: a server string never reaches the screen. Sign-in is the app's front door and
// apiClient builds its messages from the wire, so rendering err.message could put
// "Expected JSON from /api/auth/sign-in, received 502: <html><head><title>502 Bad Gateway…" in the red
// box — and, now that the box is a live region, read it aloud. Two things a caregiver can act on are
// worth distinguishing: their details were wrong, or we could not be reached. Everything else is framed
// as a connection problem, because that is what it almost always is.

function statusOf(error: unknown): number | null {
  return error instanceof LegacyApiError ? error.status : null;
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Mirrors MIN_PASSWORD_LENGTH in reflexion-server/src/v1/routes/identity.ts. The screens checked 8, so a
 * caregiver could type a password the client accepted and the server then refused with a 400 — the worst
 * place to learn a rule, because the form has already been submitted.
 */
export const MIN_PASSWORD_LENGTH = 12;

export function signInMessage(error: unknown): string {
  console.warn('[auth] sign-in failed:', describe(error));
  const status = statusOf(error);
  if (status === 400) return 'Please enter both your email and your password.';
  if (status === 401 || status === 403) return 'That email and password do not match. Please try again.';
  if (status === 429) return 'Too many attempts just now. Please wait a moment and try again.';
  return 'We could not reach Reflexion just now. Please check your connection and try again.';
}

export function passwordResetRequestMessage(error: unknown): string {
  console.warn('[auth] password-reset request failed:', describe(error));
  const status = statusOf(error);
  if (status === 400) return 'Please enter the email address you signed up with.';
  if (status === 429) return 'Too many requests just now. Please wait a moment and try again.';
  return 'We could not send the reset email just now. Please check your connection and try again.';
}

export function passwordResetMessage(error: unknown): string {
  console.warn('[auth] password reset failed:', describe(error));
  const status = statusOf(error);
  if (status === 400) return 'That reset link has expired or has already been used. Please request a new one.';
  if (status === 404) return 'That reset link is no longer valid. Please request a new one.';
  return 'We could not reset your password just now. Please check your connection and try again.';
}
