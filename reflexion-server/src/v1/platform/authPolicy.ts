/** The server is the sole authority for pilot authentication policy. */
export function emailVerificationRequired(): boolean {
  return process.env.AUTH_EMAIL_VERIFICATION_REQUIRED?.trim().toLowerCase() !== 'false'
}

export function authPolicy() {
  return { emailVerificationRequired: emailVerificationRequired() }
}
