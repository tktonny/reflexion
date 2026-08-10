export function getMongoUri() {
  const uri = process.env.MONGODB_URI
  if (!uri) {
    throw httpError(500, 'MONGODB_URI is not set')
  }
  return uri
}

export function getOpenAIApiKey() {
  const apiKey = process.env.OPENAI_API_KEY
  if (!apiKey) {
    throw httpError(500, 'OPENAI_API_KEY is not set')
  }
  return apiKey
}

export function httpError(status: number, message: string) {
  const error = new Error(message) as Error & { status?: number }
  error.status = status
  return error
}
