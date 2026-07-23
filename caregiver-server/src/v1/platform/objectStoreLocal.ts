import { createHash, createHmac, timingSafeEqual } from 'node:crypto'
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises'
import { dirname, resolve, sep } from 'node:path'
import express, { type Express } from 'express'
import { ApiError } from './errors.js'
import type { ObjectStore, PreparedObject, UploadPlan } from './objectStore.js'

// Filesystem-backed object store for single-box / dev deployments (implementation baseline §6). It
// keeps the SAME presigned-upload contract the mirror already speaks: prepareUpload() returns a
// short-lived signed PUT URL back to THIS server, the mirror PUTs the bytes, and verify() confirms the
// stored object's size + SHA-256. No external S3 needed. NOT for multi-instance production (local disk).

const MAX_UPLOAD_BYTES = Number(process.env.OBJECT_STORE_LOCAL_MAX_BYTES || 64 * 1024 * 1024)

function config() {
  const baseUrl = required('OBJECT_STORE_LOCAL_BASE_URL').replace(/\/$/, '')
  const dir = process.env.OBJECT_STORE_LOCAL_DIR || resolve(process.cwd(), '.object-store')
  const secret = required('OBJECT_STORE_LOCAL_SECRET')
  if (secret.length < 16) throw new ApiError(503, 'OBJECT_STORE_NOT_CONFIGURED', 'OBJECT_STORE_LOCAL_SECRET must be at least 16 characters.')
  return { baseUrl, dir, secret }
}

// Resolve an object key to an absolute path INSIDE the storage dir, rejecting traversal.
function objectPath(dir: string, objectKey: string) {
  const full = resolve(dir, objectKey)
  const root = resolve(dir) + sep
  if (!full.startsWith(root)) throw new ApiError(400, 'INVALID_OBJECT_KEY', 'Object key escapes the storage root.')
  return full
}

function sign(secret: string, objectKey: string, expiresAtMs: number) {
  return createHmac('sha256', secret).update(`${objectKey}\n${expiresAtMs}`).digest('hex')
}

function safeEqualHex(a: string, b: string) {
  if (a.length !== b.length) return false
  try { return timingSafeEqual(Buffer.from(a, 'hex'), Buffer.from(b, 'hex')) } catch { return false }
}

export class LocalObjectStore implements ObjectStore {
  async prepareUpload(input: PreparedObject): Promise<UploadPlan> {
    const { baseUrl, secret } = config()
    const expiresInSeconds = Math.min(Math.max(input.expiresInSeconds || 900, 60), 3600)
    const expiresAt = new Date(Date.now() + expiresInSeconds * 1000)
    const signature = sign(secret, input.objectKey, expiresAt.getTime())
    const query = new URLSearchParams({ key: input.objectKey, exp: String(expiresAt.getTime()), sig: signature, sha256: input.hash })
    return {
      uploadUrl: `${baseUrl}/api/v1/object-store/objects?${query.toString()}`,
      expiresAt,
      requiredHeaders: { 'content-type': input.contentType, 'x-reflexion-sha256': input.hash },
    }
  }

  async verify(input: { objectKey: string; hash: string; sizeBytes: number }) {
    const { dir } = config()
    try {
      const path = objectPath(dir, input.objectKey)
      const info = await stat(path)
      if (info.size !== input.sizeBytes) return false
      const bytes = await readFile(path)
      return createHash('sha256').update(bytes).digest('hex') === input.hash
    } catch {
      return false
    }
  }
}

// Mounts the signed-URL PUT target. MUST be registered before express.json() so the raw binary body is
// not JSON-parsed, and outside the /api/v1 auth stack (a presigned URL carries its own HMAC token).
export function maybeMountLocalObjectStore(app: Express) {
  if ((process.env.OBJECT_STORE_DRIVER || '').toLowerCase() !== 'local') return
  app.put('/api/v1/object-store/objects', express.raw({ type: '*/*', limit: MAX_UPLOAD_BYTES }), (request, response) => {
    void (async () => {
      try {
        const { dir, secret } = config()
        const objectKey = String(request.query.key || '')
        const exp = Number(request.query.exp)
        const sig = String(request.query.sig || '')
        const declaredHash = String(request.query.sha256 || '')
        if (!objectKey || !Number.isFinite(exp)) return response.status(400).json({ error: 'INVALID_UPLOAD_URL' })
        if (exp < Date.now()) return response.status(403).json({ error: 'UPLOAD_URL_EXPIRED' })
        if (!safeEqualHex(sig, sign(secret, objectKey, exp))) return response.status(403).json({ error: 'UPLOAD_SIGNATURE_INVALID' })
        const body: Buffer = Buffer.isBuffer(request.body) ? request.body : Buffer.from([])
        const actualHash = createHash('sha256').update(body).digest('hex')
        const headerHash = String(request.header('x-reflexion-sha256') || declaredHash)
        if (headerHash && headerHash !== actualHash) return response.status(400).json({ error: 'ARTIFACT_HASH_MISMATCH' })
        const path = objectPath(dir, objectKey)
        await mkdir(dirname(path), { recursive: true })
        await writeFile(path, body)
        return response.status(200).json({ ok: true, sizeBytes: body.byteLength })
      } catch (error) {
        const status = error instanceof ApiError ? error.status : 500
        return response.status(status).json({ error: error instanceof Error ? error.message : 'UPLOAD_FAILED' })
      }
    })()
  })
}

function required(name: string) {
  const value = process.env[name]?.trim()
  if (!value) throw new ApiError(503, 'OBJECT_STORE_NOT_CONFIGURED', `${name} is not configured.`, true)
  return value
}
