import type { NextFunction, Request, Response } from 'express'

type Handler = (request: Request, response: Response, next: NextFunction) => Promise<void>

export function asyncHandler(handler: Handler) {
  return (request: Request, response: Response, next: NextFunction) => {
    handler(request, response, next).catch(next)
  }
}
