import { ApiError } from '../platform/errors.js'

export type EmbeddingResult = { vector: number[]; modelId: string; dimensions: number; family: string }

export interface EmbeddingProvider { embed(text: string): Promise<EmbeddingResult> }

export function configuredEmbeddingProvider(): EmbeddingProvider | null {
  const provider = process.env.EMBEDDING_PROVIDER?.toLowerCase()
  if (!provider) return null
  if (provider === 'openai_compatible') return new OpenAICompatibleEmbeddingProvider()
  throw new ApiError(503, 'EMBEDDING_PROVIDER_INVALID', `Unsupported embedding provider: ${provider}.`, true)
}

class OpenAICompatibleEmbeddingProvider implements EmbeddingProvider {
  private readonly base = required('EMBEDDING_API_BASE').replace(/\/$/, '')
  private readonly apiKey = required('EMBEDDING_API_KEY')
  private readonly model = required('EMBEDDING_MODEL')
  private readonly expectedDimensions = Number(process.env.EMBEDDING_DIMENSIONS || 0)

  async embed(text: string) {
    const response = await fetch(`${this.base}/embeddings`, {
      method: 'POST', headers: { Authorization: `Bearer ${this.apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: this.model, input: text }), signal: AbortSignal.timeout(30_000),
    })
    const body = await response.json().catch(() => null) as { data?: Array<{ embedding?: number[] }>; error?: { message?: string } } | null
    const vector = body?.data?.[0]?.embedding
    if (!response.ok || !Array.isArray(vector) || !vector.length || vector.some((value) => !Number.isFinite(value))) {
      throw new ApiError(502, 'EMBEDDING_FAILED', body?.error?.message || 'Embedding provider returned an invalid result.', true)
    }
    if (this.expectedDimensions && vector.length !== this.expectedDimensions) {
      throw new ApiError(502, 'EMBEDDING_DIMENSION_MISMATCH', `Expected ${this.expectedDimensions} embedding dimensions but received ${vector.length}.`)
    }
    return { vector, modelId: this.model, dimensions: vector.length, family: 'transcript_semantic' }
  }
}

function required(name: string) {
  const value = process.env[name]?.trim()
  if (!value) throw new ApiError(503, 'EMBEDDING_NOT_CONFIGURED', `${name} is not configured.`, true)
  return value
}
