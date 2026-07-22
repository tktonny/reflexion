const TRUSTED_QWEN_ASSET_HOST = /^(?:[a-z0-9-]+\.)*aliyuncs\.com$/i

/**
 * DashScope occasionally returns an http:// OSS result URL even though the same object is
 * available over HTTPS. Release Android builds correctly reject cleartext traffic, so upgrade
 * only trusted Aliyun asset hosts and reject every other insecure remote URL.
 */
export function secureQwenAssetUrl(value: unknown): string | null {
  if (typeof value !== 'string' || !value.trim()) return null
  const raw = value.trim()

  try {
    const parsed = new URL(raw)
    if (parsed.protocol === 'https:') return parsed.toString()
    if (parsed.protocol === 'http:' && TRUSTED_QWEN_ASSET_HOST.test(parsed.hostname)) {
      parsed.protocol = 'https:'
      return parsed.toString()
    }
  } catch {
    return null
  }

  return null
}
