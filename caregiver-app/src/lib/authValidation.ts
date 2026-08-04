import { MIN_PASSWORD_LENGTH } from './authMessages';

export const EMAIL_PATTERN = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

export type CreateAccountValues = {
  name: string;
  email: string;
  countryCode: string;
  phoneNumber: string;
  password: string;
  repeatPassword: string;
};

export type FieldErrors = Partial<Record<keyof CreateAccountValues | 'identifier' | 'currentPassword' | 'code', string>>;

export function validateEmail(value: string): string | undefined {
  if (!value.trim()) return 'Enter your email address.';
  if (!EMAIL_PATTERN.test(value.trim())) return 'Enter a valid email address.';
  return undefined;
}

export function validatePhone(countryCode: string, phoneNumber: string, required = false): string | undefined {
  const digits = phoneNumber.replace(/\D/g, '');
  if (!digits && !required) return undefined;
  if (!countryCode || digits.length < 7 || digits.length > 15) return 'Enter a valid phone number, including the country code.';
  return undefined;
}

export function normalizePhone(countryCode: string, phoneNumber: string): string {
  return `${countryCode}${phoneNumber.replace(/\D/g, '')}`;
}

export function validateNewPassword(password: string): string | undefined {
  return password.length < MIN_PASSWORD_LENGTH ? `Your password must be at least ${MIN_PASSWORD_LENGTH} characters.` : undefined;
}

export function validatePasswordPair(password: string, repeatPassword: string): FieldErrors {
  const errors: FieldErrors = {};
  const passwordError = validateNewPassword(password);
  if (passwordError) errors.password = passwordError;
  if (password !== repeatPassword) errors.repeatPassword = 'The passwords do not match.';
  return errors;
}

export function validateCreateAccount(values: CreateAccountValues): FieldErrors {
  const errors: FieldErrors = {};
  if (!values.name.trim()) errors.name = 'Enter your name.';
  const emailError = validateEmail(values.email);
  if (emailError) errors.email = emailError;
  const phoneError = validatePhone(values.countryCode, values.phoneNumber);
  if (phoneError) errors.phoneNumber = phoneError;
  Object.assign(errors, validatePasswordPair(values.password, values.repeatPassword));
  return errors;
}

export function validateSignIn(identifier: string, password: string, method: 'email' | 'phone'): FieldErrors {
  const errors: FieldErrors = {};
  if (!identifier.trim()) errors.identifier = method === 'email' ? 'Enter your email address.' : 'Enter your phone number.';
  else if (method === 'email') {
    const emailError = validateEmail(identifier);
    if (emailError) errors.identifier = emailError;
  } else if (validatePhone('+', identifier, true)) {
    errors.identifier = 'Enter a valid phone number, including the country code.';
  }
  if (!password) errors.password = 'Enter your password.';
  return errors;
}

export function validateVerificationCode(code: string): string | undefined {
  if (!/^\d{6}$/.test(code)) return 'Enter the six-digit code from your email.';
  return undefined;
}
