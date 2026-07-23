import { badRequest } from './errors.js'

export function objectBody(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw badRequest('INVALID_BODY', 'A JSON object body is required.')
  }
  return value as Record<string, unknown>
}

export function requiredString(body: Record<string, unknown>, key: string, maxLength = 5000) {
  const value = body[key]
  if (typeof value !== 'string' || !value.trim() || value.length > maxLength) {
    throw badRequest('VALIDATION_FAILED', `${key} is required and must be at most ${maxLength} characters.`, [{ field: key }])
  }
  return value.trim()
}

export function optionalString(body: Record<string, unknown>, key: string, maxLength = 5000) {
  const value = body[key]
  if (value === undefined || value === null || value === '') return undefined
  if (typeof value !== 'string' || value.length > maxLength) {
    throw badRequest('VALIDATION_FAILED', `${key} must be a string of at most ${maxLength} characters.`, [{ field: key }])
  }
  return value.trim()
}

export function enumValue<const T extends readonly string[]>(value: unknown, field: string, allowed: T): T[number] {
  if (typeof value !== 'string' || !allowed.includes(value)) {
    throw badRequest('VALIDATION_FAILED', `${field} must be one of: ${allowed.join(', ')}.`, [{ field, allowed }])
  }
  return value as T[number]
}

export function isoDate(value: unknown, field: string) {
  if (typeof value !== 'string' || !value || Number.isNaN(Date.parse(value))) {
    throw badRequest('VALIDATION_FAILED', `${field} must be an ISO 8601 date-time.`, [{ field }])
  }
  return new Date(value)
}

export function stringArray(value: unknown, field: string, maximum = 100) {
  if (!Array.isArray(value) || value.length > maximum || value.some((item) => typeof item !== 'string')) {
    throw badRequest('VALIDATION_FAILED', `${field} must be an array of strings.`, [{ field }])
  }
  return value as string[]
}

export function positiveInteger(value: unknown, field: string, allowZero = false) {
  if (!Number.isInteger(value) || (value as number) < (allowZero ? 0 : 1)) {
    throw badRequest('VALIDATION_FAILED', `${field} must be ${allowZero ? 'a non-negative' : 'a positive'} integer.`, [{ field }])
  }
  return value as number
}

export function pagination(query: Record<string, unknown>) {
  const rawLimit = typeof query.limit === 'string' ? Number(query.limit) : 25
  const limit = Number.isInteger(rawLimit) ? Math.min(Math.max(rawLimit, 1), 100) : 25
  const cursor = typeof query.cursor === 'string' && query.cursor ? query.cursor : undefined
  return { limit, cursor }
}
