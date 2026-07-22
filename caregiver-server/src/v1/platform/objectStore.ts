import { createHash, createHmac } from 'node:crypto'
import { ApiError } from './errors.js'

export type UploadPlan = {
  uploadUrl: string
  expiresAt: Date
  requiredHeaders: Record<string, string>
}

export type PreparedObject = {
  objectKey: string
  contentType: string
  hash: string
  expiresInSeconds?: number
}

export interface ObjectStore {
  prepareUpload(input: PreparedObject): Promise<UploadPlan>
  verify(input: { objectKey: string; hash: string; sizeBytes: number }): Promise<boolean>
}

export function getObjectStore(): ObjectStore {
  if ((process.env.OBJECT_STORE_DRIVER || '').toLowerCase() === 's3') return new S3CompatibleObjectStore()
  return new UnconfiguredObjectStore()
}

class UnconfiguredObjectStore implements ObjectStore {
  async prepareUpload(): Promise<UploadPlan> {
    throw new ApiError(503, 'OBJECT_STORE_NOT_CONFIGURED', 'Artifact storage is not configured on this server.', true)
  }
  async verify(): Promise<boolean> {
    return false
  }
}

class S3CompatibleObjectStore implements ObjectStore {
  private readonly endpoint = requiredUrl('OBJECT_STORE_ENDPOINT')
  private readonly bucket = required('OBJECT_STORE_BUCKET')
  private readonly accessKey = required('OBJECT_STORE_ACCESS_KEY')
  private readonly secretKey = required('OBJECT_STORE_SECRET_KEY')
  private readonly region = process.env.OBJECT_STORE_REGION || 'us-east-1'

  async prepareUpload(input: PreparedObject): Promise<UploadPlan> {
    const expiresInSeconds = Math.min(Math.max(input.expiresInSeconds || 900, 60), 3600)
    const requiredHeaders: Record<string, string> = {
      'content-type': input.contentType,
      'x-amz-meta-sha256': input.hash,
    }
    const encryption = process.env.OBJECT_STORE_SERVER_SIDE_ENCRYPTION || 'AES256'
    if (encryption) requiredHeaders['x-amz-server-side-encryption'] = encryption
    if (encryption === 'aws:kms' && process.env.OBJECT_STORE_KMS_KEY_ID) {
      requiredHeaders['x-amz-server-side-encryption-aws-kms-key-id'] = process.env.OBJECT_STORE_KMS_KEY_ID
    }
    return {
      uploadUrl: this.presign('PUT', input.objectKey, expiresInSeconds, requiredHeaders),
      expiresAt: new Date(Date.now() + expiresInSeconds * 1000),
      requiredHeaders,
    }
  }

  async verify(input: { objectKey: string; hash: string; sizeBytes: number }) {
    const response = await fetch(this.presign('HEAD', input.objectKey, 60, {}), { method: 'HEAD', signal: AbortSignal.timeout(10_000) })
    if (!response.ok) return false
    return Number(response.headers.get('content-length')) === input.sizeBytes
      && response.headers.get('x-amz-meta-sha256') === input.hash
  }

  private presign(method: 'PUT' | 'HEAD', objectKey: string, expires: number, headers: Record<string, string>) {
    const now = new Date()
    const amzDate = now.toISOString().replace(/[:-]|\.\d{3}/g, '')
    const date = amzDate.slice(0, 8)
    const scope = `${date}/${this.region}/s3/aws4_request`
    const path = `${this.endpoint.pathname.replace(/\/$/, '')}/${rfc3986(this.bucket)}/${objectKey.split('/').map(rfc3986).join('/')}` || '/'
    const canonicalHeaders = { host: this.endpoint.host, ...lowerCaseHeaders(headers) }
    const signedHeaders = Object.keys(canonicalHeaders).sort().join(';')
    const query: Record<string, string> = {
      'X-Amz-Algorithm': 'AWS4-HMAC-SHA256',
      'X-Amz-Credential': `${this.accessKey}/${scope}`,
      'X-Amz-Date': amzDate,
      'X-Amz-Expires': String(expires),
      'X-Amz-SignedHeaders': signedHeaders,
    }
    const canonicalQuery = Object.entries(query).sort(([a], [b]) => a.localeCompare(b))
      .map(([key, value]) => `${rfc3986(key)}=${rfc3986(value)}`).join('&')
    const canonicalHeaderText = Object.entries(canonicalHeaders).sort(([a], [b]) => a.localeCompare(b))
      .map(([key, value]) => `${key}:${value.trim().replace(/\s+/g, ' ')}\n`).join('')
    const canonicalRequest = [method, path, canonicalQuery, canonicalHeaderText, signedHeaders, 'UNSIGNED-PAYLOAD'].join('\n')
    const stringToSign = ['AWS4-HMAC-SHA256', amzDate, scope, createHash('sha256').update(canonicalRequest).digest('hex')].join('\n')
    const signature = signatureKey(this.secretKey, date, this.region, 's3', stringToSign).toString('hex')
    return `${this.endpoint.origin}${path}?${canonicalQuery}&X-Amz-Signature=${signature}`
  }
}

function signatureKey(secret: string, date: string, region: string, service: string, stringToSign: string) {
  const dateKey = createHmac('sha256', `AWS4${secret}`).update(date).digest()
  const regionKey = createHmac('sha256', dateKey).update(region).digest()
  const serviceKey = createHmac('sha256', regionKey).update(service).digest()
  const signingKey = createHmac('sha256', serviceKey).update('aws4_request').digest()
  return createHmac('sha256', signingKey).update(stringToSign).digest()
}

function lowerCaseHeaders(headers: Record<string, string>) {
  return Object.fromEntries(Object.entries(headers).map(([key, value]) => [key.toLowerCase(), value]))
}

function rfc3986(value: string) {
  return encodeURIComponent(value).replace(/[!'()*]/g, (character) => `%${character.charCodeAt(0).toString(16).toUpperCase()}`)
}

function required(name: string) {
  const value = process.env[name]?.trim()
  if (!value) throw new ApiError(503, 'OBJECT_STORE_NOT_CONFIGURED', `${name} is not configured.`, true)
  return value
}

function requiredUrl(name: string) {
  const url = new URL(required(name))
  if (url.protocol !== 'https:' && process.env.NODE_ENV === 'production') {
    throw new ApiError(503, 'OBJECT_STORE_INSECURE', `${name} must use HTTPS in production.`)
  }
  return url
}
