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
  return error instanceof LegacyApiError || isStatusError(error) ? error.status : null;
}

function codeOf(error: unknown): string | undefined {
  return isStatusError(error) ? error.code : undefined;
}

function isStatusError(error: unknown): error is { status: number; code?: string } {
  return typeof error === 'object' && error !== null && typeof (error as { status?: unknown }).status === 'number';
}

/**
 * Mirrors MIN_PASSWORD_LENGTH in reflexion-server/src/v1/routes/identity.ts. The screens checked 8, so a
 * caregiver could type a password the client accepted and the server then refused with a 400 — the worst
 * place to learn a rule, because the form has already been submitted.
 */
export const MIN_PASSWORD_LENGTH = 12;

function logFailure(area: string, error: unknown) {
  // Keep diagnostics useful without copying provider responses (which can contain email addresses or
  // request payloads) into device logs. Passwords are never included here.
  console.warn(`[auth] ${area} failed`, { status: statusOf(error), code: codeOf(error) });
}

export function signInMessage(error: unknown): string {
  logFailure('sign-in', error);
  const status = statusOf(error);
  const code = codeOf(error);
  if (code === 'ACCOUNT_NOT_FOUND' || status === 404) return 'We could not find an account with those details. Check your email or create an account.';
  if (code === 'EMAIL_NOT_VERIFIED' || status === 403) return 'Your email has not been verified. Resend the verification email and try again.';
  if (status === 400) return 'Please enter both your email and your password.';
  if (status === 401 || status === 403) return 'That email and password do not match. Please try again.';
  if (status === 429) return 'Too many attempts just now. Please wait a moment and try again.';
  return 'We could not reach Reflexion just now. Please check your connection and try again.';
}

export function passwordResetRequestMessage(error: unknown): string {
  logFailure('password-reset request', error);
  const status = statusOf(error);
  if (codeOf(error) === 'EMAIL_NOT_CONFIGURED') return 'Password reset email is unavailable during this pilot. Contact Reflexion support for help.';
  if (codeOf(error) === 'EMAIL_DELIVERY_FAILED') return 'The reset email could not be sent because email delivery is unavailable. Try again later.';
  if (status === 400) return 'Please enter the email address you signed up with.';
  if (status === 429) return 'Too many requests just now. Please wait a moment and try again.';
  return 'We could not send the reset email just now. Please check your connection and try again.';
}

export function passwordResetMessage(error: unknown): string {
  logFailure('password reset', error);
  const status = statusOf(error);
  if (codeOf(error) === 'PASSWORD_TOO_SHORT') return `Your password must be at least ${MIN_PASSWORD_LENGTH} characters.`;
  if (codeOf(error) === 'CURRENT_PASSWORD_INVALID') return 'Your current password is incorrect. Check it and try again.';
  if (status === 400) return 'That reset link has expired or has already been used. Please request a new one.';
  if (status === 404) return 'That reset link is no longer valid. Please request a new one.';
  return 'We could not reset your password just now. Please check your connection and try again.';
}

export function registrationMessage(error: unknown): string {
  logFailure('registration', error);
  const code = codeOf(error);
  if (code === 'EMAIL_VERIFICATION_PENDING') return 'An account already exists for this email and is waiting for verification. Resend the verification email or sign in.';
  if (code === 'EMAIL_IN_USE') return 'An account already exists for this email. Sign in or reset your password.';
  if (code === 'EMAIL_INVALID') return 'Enter a valid email address.';
  if (code === 'PASSWORD_TOO_SHORT') return `Your password must be at least ${MIN_PASSWORD_LENGTH} characters.`;
  if (statusOf(error) === 409) return 'An account already exists for this email. Sign in or reset your password.';
  return 'We could not create your account. Check your details and try again.';
}

export function verificationMessage(error: unknown): string {
  logFailure('verification', error);
  const code = codeOf(error);
  if (code === 'ACCOUNT_VERIFICATION_INVALID') return 'This verification code is invalid or expired. Request a new code and try again.';
  if (code === 'EMAIL_NOT_CONFIGURED' || code === 'EMAIL_DELIVERY_FAILED') return 'The verification email could not be sent because email delivery is unavailable. Try again later.';
  return 'We could not verify your account. Check your connection and try again.';
}

export function verificationResendMessage(error: unknown): string {
  logFailure('verification resend', error);
  const code = codeOf(error);
  if (code === 'EMAIL_NOT_CONFIGURED' || code === 'EMAIL_DELIVERY_FAILED') return 'The verification email could not be sent because email delivery is unavailable. Try again later.';
  return 'We could not request another verification email. Check your connection and try again.';
}

export function emailChangeMessage(error: unknown): string {
  logFailure('email change', error);
  const code = codeOf(error);
  if (code === 'EMAIL_IN_USE') return 'An account already exists for this email. Choose another address.';
  if (code === 'EMAIL_CHANGE_INVALID') return 'This email confirmation code is invalid or expired. Request a new one.';
  if (code === 'EMAIL_NOT_CONFIGURED' || code === 'EMAIL_DELIVERY_FAILED') return 'The confirmation email could not be sent because email delivery is unavailable. Try again later.';
  return 'We could not update your email. Check your connection and try again.';
}

export function phoneChangeMessage(error: unknown): string {
  logFailure('phone change', error);
  const code = codeOf(error);
  if (code === 'PHONE_INVALID') return 'Enter a valid phone number, including the country code.';
  if (code === 'PHONE_CHANGE_CODE_INVALID') return 'That verification code is invalid or expired. Request a new code and try again.';
  if (code === 'SMS_NOT_CONFIGURED' || code === 'SMS_DELIVERY_FAILED') return 'The verification code could not be sent because SMS delivery is unavailable. Try again later.';
  return 'We could not update your phone number. Check your connection and try again.';
}
