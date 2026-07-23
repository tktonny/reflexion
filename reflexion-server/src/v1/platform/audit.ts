import type { NextFunction, Request, Response } from 'express'
import { getDb } from '../../lib/mongo.js'
import { collections } from './collections.js'
import { newId } from './ids.js'

/** Append-only access audit. Bodies, tokens and query strings are deliberately never recorded. */
export function auditAccess(request: Request, response: Response, next: NextFunction) {
  response.once('finish', () => {
    const principal = request.principal
    if (!principal) return
    const path = request.originalUrl.split('?')[0]
    void record({
      _id: newId('audit'),
      tenantId: principal.tenantId,
      actor: { type: principal.kind, id: principal.subjectId },
      action: `${request.method.toUpperCase()} ${path}`,
      object: { type: classifyPath(path), id: lastIdentifier(path) },
      outcome: response.statusCode < 400 ? 'success' : 'failure',
      statusCode: response.statusCode,
      correlationId: request.requestId,
      occurredAt: new Date(),
    })
  })
  next()
}

async function record(event: Record<string, unknown>) {
  try {
    await (await getDb()).collection<any>(collections.auditEvents).insertOne(event)
  } catch (error) {
    console.error(`[${String(event.correlationId)}] audit write failed`, error)
  }
}

function classifyPath(path: string) {
  if (path.includes('/review-cases')) return 'review_case'
  if (path.includes('/sessions')) return 'session'
  if (path.includes('/devices') || path.includes('/device-')) return 'device'
  if (path.includes('/patients')) return 'patient'
  if (path.includes('/medication') || path.includes('/reminder')) return 'care_plan'
  return 'api_resource'
}

function lastIdentifier(path: string) {
  const parts = path.split('/').filter(Boolean)
  const candidate = parts.at(-1) || 'collection'
  return ['complete', 'abandon', 'configuration', 'heartbeats', 'revocations', 'responses', 'dispositions'].includes(candidate)
    ? parts.at(-2) || candidate
    : candidate
}
