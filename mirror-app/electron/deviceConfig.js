// Runtime device configuration for the Linux (Ubuntu) mirror.
//
// The Android APK bakes its identity in at build time (`EXPO_PUBLIC_DEVICE_BOOTSTRAP_TOKEN` is inlined
// into the JS bundle). That is wrong for the Linux appliance, and mildly wrong for Android too: a
// bootstrap token is DEVICE-BOUND — the backend re-reads `devices` by (did, serialHash) on every call — so
// one token baked into one installer means every unit built from it claims the SAME device identity and
// they knock each other's pairing over. One AppImage has to serve the whole fleet.
//
// So the Linux build ships NO identity. Both the backend origin and the bootstrap token are resolved at
// runtime, from a file an operator drops next to the app:
//
//   <userData>/device-config.json   { "apiBase": "https://...", "bootstrapToken": "eyJ..." }
//
// A file rather than an on-screen field because the mirror is a keyboard-less appliance and the token is a
// ~300-character JWT. The in-app entry screen (app/test-device.tsx) still works and is the right path when
// someone has a phone/keyboard handy; this is the path for imaging a unit or fixing one over scp.
//
// <userData> on Linux is ~/.config/<productName>/ — printed at startup so an installer can find it.

const fs = require('fs')
const path = require('path')

const { DEFAULT_API_BASE, normalizeBase } = require('./apiProxy')

const CONFIG_FILENAME = 'device-config.json'

function configPath(userDataDir) {
  return path.join(userDataDir, CONFIG_FILENAME)
}

/**
 * Read the config file. Never throws and never returns null — a missing or corrupt file must degrade to
 * "no configuration", because throwing here would take the whole appliance down over a stray comma and
 * leave a mirror in a home with a black screen.
 */
function readDeviceConfig(userDataDir) {
  const file = configPath(userDataDir)
  try {
    if (!fs.existsSync(file)) return {}
    const parsed = JSON.parse(fs.readFileSync(file, 'utf8'))
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch (error) {
    console.warn(`[electron] ignoring unreadable ${CONFIG_FILENAME}: ${error.message}`)
    return {}
  }
}

/**
 * Backend origin. Precedence: launch env -> config file -> production default. Runtime resolution is what
 * lets a unit be re-pointed without re-exporting the web bundle, and it is what makes the same-origin /api
 * proxy possible at all: the main process must know the target before the renderer runs.
 */
function resolveApiBase(userDataDir, env = process.env) {
  const fromEnv = env.REFLEXION_API_BASE || env.EXPO_PUBLIC_API_BASE
  if (fromEnv && fromEnv.trim()) return normalizeBase(fromEnv)
  const configured = readDeviceConfig(userDataDir).apiBase
  if (typeof configured === 'string' && configured.trim()) return normalizeBase(configured)
  return DEFAULT_API_BASE
}

// A compact JWT: three base64url segments. This is a SHAPE check, not verification — only the backend can
// verify the signature, and only it knows whether the device row still exists. The point is to catch the
// ordinary operator mistakes (empty string, a pasted filename, a truncated copy) at startup with a clear
// log line, instead of letting them surface later as an opaque 401 during pairing.
const JWT_SHAPE = /^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/

/** The `did` a bootstrap token claims, for logging. Unverified by definition — never trust it for auth. */
function claimedDeviceId(token) {
  try {
    const claims = JSON.parse(Buffer.from(token.split('.')[1], 'base64url').toString('utf8'))
    return typeof claims?.did === 'string' ? claims.did : null
  } catch {
    return null
  }
}

/**
 * The per-device bootstrap token, or null when the unit has not been provisioned yet (a normal state — the
 * app boots, reports that it needs provisioning, and the network-setup flow still works without it).
 *
 * Precedence: launch env -> config file. Never a build-time constant: nothing device-bound is compiled in.
 */
function resolveBootstrapToken(userDataDir, env = process.env) {
  const candidates = [
    ['REFLEXION_BOOTSTRAP_TOKEN', env.REFLEXION_BOOTSTRAP_TOKEN],
    [CONFIG_FILENAME, readDeviceConfig(userDataDir).bootstrapToken],
  ]
  for (const [source, value] of candidates) {
    if (typeof value !== 'string' || !value.trim()) continue
    const token = value.trim()
    if (!JWT_SHAPE.test(token)) {
      console.warn(`[electron] ${source} does not look like a bootstrap token (expected three base64url segments) — ignoring`)
      continue
    }
    // Log the device id, never the token: the token IS the credential.
    console.log(`[electron] bootstrap token from ${source} (device ${claimedDeviceId(token) ?? 'unknown'})`)
    return token
  }
  return null
}

module.exports = {
  CONFIG_FILENAME,
  claimedDeviceId,
  configPath,
  readDeviceConfig,
  resolveApiBase,
  resolveBootstrapToken,
}
