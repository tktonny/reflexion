import { ApiError } from '../platform/errors.js'

export async function sendPasswordResetEmail(input: { email: string; name?: string; token: string }) {
  const provider = process.env.EMAIL_PROVIDER?.toLowerCase()
  if (provider !== 'postmark') throw new ApiError(503, 'EMAIL_NOT_CONFIGURED', 'Transactional email is not configured.', true)
  const key = process.env.POSTMARK_SERVER_TOKEN?.trim()
  const from = process.env.EMAIL_FROM?.trim()
  const caregiverUrl = process.env.CAREGIVER_APP_URL?.trim()
  if (!key || !from || !caregiverUrl) throw new ApiError(503, 'EMAIL_NOT_CONFIGURED', 'POSTMARK_SERVER_TOKEN, EMAIL_FROM and CAREGIVER_APP_URL are required.', true)
  const resetUrl = new URL('/reset-password', caregiverUrl)
  resetUrl.searchParams.set('token', input.token)
  const response = await fetch('https://api.postmarkapp.com/email', {
    method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json', 'X-Postmark-Server-Token': key },
    body: JSON.stringify({ From: from, To: input.email, Subject: 'Reset your Reflexion password',
      TextBody: `Hello${input.name ? ` ${input.name}` : ''},\n\nUse this link within 30 minutes to reset your Reflexion password:\n${resetUrl.toString()}\n\nIf you did not request this, you can ignore this email.` }),
    signal: AbortSignal.timeout(10_000),
  })
  if (!response.ok) throw new ApiError(502, 'EMAIL_DELIVERY_FAILED', 'Unable to deliver the password reset email.', true)
}
