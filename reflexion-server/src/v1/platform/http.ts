import { randomUUID } from 'node:crypto'
import type { ErrorRequestHandler, NextFunction, Request, Response } from 'express'
import { ApiError } from './errors.js'

const REQUEST_ID_PATTERN = /^[A-Za-z0-9._:-]{8,128}$/

export function requestContext(request: Request, response: Response, next: NextFunction) {
  const supplied = request.header('X-Request-Id')
  request.requestId = supplied && REQUEST_ID_PATTERN.test(supplied) ? supplied : `req_${randomUUID().replaceAll('-', '')}`
  response.setHeader('X-Request-Id', request.requestId)
  next()
}

export function sendData(response: Response, data: unknown, status = 200) {
  return response.status(status).json({ data, meta: { requestId: response.req.requestId } })
}

export function sendPage(response: Response, data: unknown[], nextCursor: string | null) {
  return response.json({ data, meta: { requestId: response.req.requestId, nextCursor } })
}

export function v1NotFound(request: Request, _response: Response, next: NextFunction) {
  if (request.path.startsWith('/api/v1')) {
    next(new ApiError(404, 'ROUTE_NOT_FOUND', 'The requested API route does not exist.'))
    return
  }
  next()
}

export const apiErrorHandler: ErrorRequestHandler = (error, request, response, next) => {
  if (!request.path.startsWith('/api/v1')) {
    next(error)
    return
  }
  const apiError = error instanceof ApiError
    ? error
    : new ApiError(500, 'INTERNAL_ERROR', 'An unexpected server error occurred.', true)
  if (!(error instanceof ApiError)) console.error(`[${request.requestId}]`, error)
  response.status(apiError.status).json({
    error: {
      code: apiError.code,
      message: apiError.message,
      retryable: apiError.retryable,
      details: apiError.details,
    },
    meta: { requestId: request.requestId },
  })
}
