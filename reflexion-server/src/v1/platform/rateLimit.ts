import type { NextFunction, Request, Response } from 'express'
import { ApiError } from './errors.js'

type Bucket = { count: number; resetsAt: number }

export function rateLimit(options: { maximum: number; windowMs?: number; namespace: string }) {
  const buckets = new Map<string, Bucket>()
  const windowMs = options.windowMs || 60_000
  return (request: Request, response: Response, next: NextFunction) => {
    const now = Date.now()
    const key = `${options.namespace}:${request.ip || request.socket.remoteAddress || 'unknown'}`
    const current = buckets.get(key)
    const bucket = !current || current.resetsAt <= now ? { count: 0, resetsAt: now + windowMs } : current
    bucket.count++
    buckets.set(key, bucket)
    response.setHeader('RateLimit-Limit', String(options.maximum))
    response.setHeader('RateLimit-Remaining', String(Math.max(options.maximum - bucket.count, 0)))
    response.setHeader('RateLimit-Reset', String(Math.ceil(bucket.resetsAt / 1000)))
    if (bucket.count > options.maximum) {
      response.setHeader('Retry-After', String(Math.ceil((bucket.resetsAt - now) / 1000)))
      next(new ApiError(429, 'RATE_LIMITED', 'Too many requests. Retry after the indicated delay.', true))
      return
    }
    if (buckets.size > 10_000) {
      for (const [bucketKey, value] of buckets) if (value.resetsAt <= now) buckets.delete(bucketKey)
    }
    next()
  }
}
