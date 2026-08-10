import { ApiError } from '../platform/errors.js'

export function emailDeliveryConfigured() {
  return process.env.EMAIL_PROVIDER?.toLowerCase() === 'postmark'
    && Boolean(process.env.POSTMARK_SERVER_TOKEN?.trim())
    && Boolean(process.env.EMAIL_FROM?.trim())
}

export async function sendPasswordResetEmail(input: { email: string; name?: string; code: string }) {
  return sendCodeEmail({ ...input, subject: 'Your Reflexion password reset code', copy: 'Enter this six-digit code in the Reflexion app to reset your password. It expires in 30 minutes:' })
}

export async function sendPasswordResetCodeEmail(input: { email: string; name?: string; code: string }) {
  return sendCodeEmail({ ...input, subject: 'Your Reflexion password reset code', copy: 'Use this six-digit code in the Reflexion app to reset your password:' })
}

export async function sendAccountVerificationEmail(input: { email: string; name?: string; code: string }) {
  return sendCodeEmail({ ...input, subject: 'Verify your Reflexion account', copy: 'Enter this six-digit code in the Reflexion app to verify your account. It expires in 24 hours:' })
}

export async function sendEmailChangeEmail(input: { email: string; name?: string; code: string }) {
  return sendCodeEmail({ ...input, subject: 'Confirm your Reflexion email address', copy: 'Enter this six-digit code in the Reflexion app to confirm this email address. It expires in 30 minutes:' })
}

export async function sendCareCircleInvitationEmail(input: { email: string; name?: string; token: string }) {
  return sendActionEmail({ ...input, subject: 'You have been invited to a Reflexion Care Circle', path: '/care-circle/invitations/accept', copy: 'Use this link to review the Care Circle invitation:' })
}

async function sendActionEmail(input: { email: string; name?: string; token: string; subject: string; path: string; copy: string }) {
  const provider = process.env.EMAIL_PROVIDER?.toLowerCase()
  if (provider !== 'postmark') throw new ApiError(503, 'EMAIL_NOT_CONFIGURED', 'Transactional email is not configured.', true)
  const key = process.env.POSTMARK_SERVER_TOKEN?.trim()
  const from = process.env.EMAIL_FROM?.trim()
  const caregiverUrl = process.env.CAREGIVER_APP_URL?.trim()
  if (!key || !from || !caregiverUrl) throw new ApiError(503, 'EMAIL_NOT_CONFIGURED', 'POSTMARK_SERVER_TOKEN, EMAIL_FROM and CAREGIVER_APP_URL are required for invitation links.', true)
  const url = new URL(input.path, caregiverUrl)
  url.searchParams.set('token', input.token)
  const response = await fetch('https://api.postmarkapp.com/email', {
    method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json', 'X-Postmark-Server-Token': key },
    body: JSON.stringify({ From: from, To: input.email, Subject: input.subject,
      TextBody: `Hello${input.name ? ` ${input.name}` : ''},\n\n${input.copy}\n${url.toString()}\n\nIf you did not request this, you can ignore this email.` }),
    signal: AbortSignal.timeout(10_000),
  })
  if (!response.ok) throw new ApiError(502, 'EMAIL_DELIVERY_FAILED', 'Unable to deliver the Reflexion email.', true)
}

/** SMS delivery is provider-backed and intentionally fails closed until staging supplies Twilio credentials. */
export async function sendPhoneChangeCode(input: { phoneNumber: string; code: string }) {
  if (process.env.SMS_PROVIDER?.toLowerCase() !== 'twilio') throw new ApiError(503, 'SMS_NOT_CONFIGURED', 'SMS verification is not configured on this server.', true)
  const accountSid = process.env.TWILIO_ACCOUNT_SID?.trim()
  const authToken = process.env.TWILIO_AUTH_TOKEN?.trim()
  const from = process.env.TWILIO_FROM_NUMBER?.trim()
  if (!accountSid || !authToken || !from) throw new ApiError(503, 'SMS_NOT_CONFIGURED', 'TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER are required.', true)
  const response = await fetch(`https://api.twilio.com/2010-04-01/Accounts/${encodeURIComponent(accountSid)}/Messages.json`, {
    method: 'POST',
    headers: { Authorization: `Basic ${Buffer.from(`${accountSid}:${authToken}`).toString('base64')}`, 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ To: input.phoneNumber, From: from, Body: `Your Reflexion verification code is ${input.code}. It expires in 30 minutes.` }),
    signal: AbortSignal.timeout(10_000),
  })
  if (!response.ok) throw new ApiError(502, 'SMS_DELIVERY_FAILED', 'Unable to deliver the Reflexion verification code.', true)
}

async function sendCodeEmail(input: { email: string; name?: string; code: string; subject: string; copy: string }) {
  const provider = process.env.EMAIL_PROVIDER?.toLowerCase()
  if (provider !== 'postmark') throw new ApiError(503, 'EMAIL_NOT_CONFIGURED', 'Transactional email is not configured.', true)
  const key = process.env.POSTMARK_SERVER_TOKEN?.trim()
  const from = process.env.EMAIL_FROM?.trim()
  if (!key || !from) throw new ApiError(503, 'EMAIL_NOT_CONFIGURED', 'POSTMARK_SERVER_TOKEN and EMAIL_FROM are required.', true)
  const response = await fetch('https://api.postmarkapp.com/email', {
    method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json', 'X-Postmark-Server-Token': key },
    body: JSON.stringify({ From: from, To: input.email, Subject: input.subject,
      TextBody: `Hello${input.name ? ` ${input.name}` : ''},\n\n${input.copy}\n\n${input.code}\n\nIf you did not request this, you can ignore this email.` }),
    signal: AbortSignal.timeout(10_000),
  })
  if (!response.ok) throw new ApiError(502, 'EMAIL_DELIVERY_FAILED', 'Unable to deliver the Reflexion email.', true)
}
