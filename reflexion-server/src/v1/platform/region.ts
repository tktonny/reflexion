import { createRequire } from 'node:module'
import type { Request } from 'express'
import { normalizeRegion, type QwenRegion } from './qwen.js'

// geoip-lite is CommonJS and ships no types; load it lazily via createRequire so (a) the large MaxMind
// Lite DB is only read on first use and (b) a missing package degrades to "no IP signal" instead of a
// crash — IP is only the VALIDATOR here (device probe wins), so losing it is non-fatal.
const require = createRequire(import.meta.url)
let geoip: { lookup(ip: string): { country?: string } | null } | null | undefined

function ipCountry(ip: string): string | null {
  try {
    if (geoip === undefined) geoip = require('geoip-lite')
    if (!geoip) return null
    return geoip.lookup(ip.replace(/^::ffff:/, ''))?.country || null
  } catch {
    geoip = null
    return null
  }
}

/** Real client IP: X-Forwarded-For's first hop (we sit behind nginx, `trust proxy` is not set so
 *  request.ip would be the proxy), else request.ip, else the raw socket address. */
export function getClientIp(request: Request): string {
  const real = String(request.headers['x-real-ip'] || '').trim()
  if (real) return real.replace(/^::ffff:/, '')
  // trust-proxy is off, so request.ip is nginx. Under `proxy_add_x_forwarded_for` (append) the LAST XFF
  // hop is the client IP nginx actually saw; the LEFTMOST is client-spoofable, so never trust split[0].
  const hops = String(request.headers['x-forwarded-for'] || '').split(',').map((h) => h.trim()).filter(Boolean)
  if (hops.length) return hops[hops.length - 1].replace(/^::ffff:/, '')
  return String(request.ip || request.socket?.remoteAddress || '').replace(/^::ffff:/, '')
}

function ipToRegion(ip: string): QwenRegion | null {
  const country = ipCountry(ip)
  if (!country) return null
  const c = country.toUpperCase()
  if (c === 'CN') return 'cn'
  if (c === 'JP') return 'jp'
  return 'sg'
}

// Coarse timezone → region, used only as a fallback when neither probe nor IP is available.
const CN_TZ = new Set(['Asia/Shanghai', 'Asia/Urumqi', 'Asia/Chongqing', 'Asia/Harbin', 'Asia/Kashgar', 'PRC'])
const JP_TZ = new Set(['Asia/Tokyo', 'Japan'])
const SG_TZ = new Set([
  'Asia/Singapore', 'Singapore', 'Asia/Kuala_Lumpur', 'Asia/Kuching', 'Asia/Jakarta', 'Asia/Pontianak',
  'Asia/Makassar', 'Asia/Jayapura', 'Asia/Bangkok', 'Asia/Ho_Chi_Minh', 'Asia/Manila', 'Asia/Brunei',
])
function tzToRegion(tz?: string): QwenRegion | null {
  if (!tz) return null
  if (CN_TZ.has(tz)) return 'cn'
  if (JP_TZ.has(tz)) return 'jp'
  if (SG_TZ.has(tz)) return 'sg'
  return null
}

/** Accepts the same region vocabulary as normalizeRegion (cn/sg/sea/singapore/ap-southeast-1, any case)
 *  so the probe path treats a valid-but-aliased value as a signal instead of silently ignoring it. */
function looksLikeRegion(value: unknown): boolean {
  return ['cn', 'sg', 'sea', 'singapore', 'ap-southeast-1'].includes(String(value ?? '').trim().toLowerCase())
}

export type RegionSignals = {
  region: QwenRegion
  source: 'probe' | 'ip' | 'timezone' | 'default'
  probed: QwenRegion | null
  ip: QwenRegion | null
  timezone: QwenRegion | null
  /** probe and IP both present and disagree — probe still wins, but flag it for audit. */
  mismatch: boolean
}

/**
 * Decide a device's Qwen region at pairing. Rule (confirmed with the product owner): the device's
 * endpoint PROBE wins (it reflects real reachable latency); IP-geo VALIDATES and flags mismatches;
 * timezone is a last-resort fallback; then QWEN_DEFAULT_REGION (cn).
 */
export function resolveDeviceRegion(input: { probedRegion?: unknown; ip?: string; timezone?: string }): RegionSignals {
  const probed = looksLikeRegion(input.probedRegion) ? normalizeRegion(input.probedRegion) : null
  const ip = input.ip ? ipToRegion(input.ip) : null
  const timezone = tzToRegion(input.timezone)
  let region: QwenRegion
  let source: RegionSignals['source']
  if (probed) { region = probed; source = 'probe' }
  else if (ip) { region = ip; source = 'ip' }
  else if (timezone) { region = timezone; source = 'timezone' }
  else { region = normalizeRegion(process.env.QWEN_DEFAULT_REGION); source = 'default' }
  return { region, source, probed, ip, timezone, mismatch: Boolean(probed && ip && probed !== ip) }
}
