export type ErrorDetail = Record<string, unknown>

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
    public readonly retryable = false,
    public readonly details: ErrorDetail[] = [],
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export function badRequest(code: string, message: string, details: ErrorDetail[] = []) {
  return new ApiError(400, code, message, false, details)
}

export function unauthorized(message = 'Authentication is required.') {
  return new ApiError(401, 'UNAUTHORIZED', message)
}

export function forbidden(message = 'You are not authorized to access this resource.') {
  return new ApiError(403, 'FORBIDDEN', message)
}

export function notFound(resource = 'Resource') {
  return new ApiError(404, 'NOT_FOUND', `${resource} was not found.`)
}

export function conflict(code: string, message: string, retryable = false) {
  return new ApiError(409, code, message, retryable)
}
