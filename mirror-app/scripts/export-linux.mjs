#!/usr/bin/env node
// Export the web bundle for the Linux (Ubuntu) AppImage.
//
// This exists for ONE reason: the Linux artifact must contain no device identity.
//
// A bootstrap token is device-bound — the backend re-reads `devices` by (did, serialHash) on every call — so
// a token compiled into the installer makes every unit built from it claim the SAME device, and they knock
// each other's pairing over. One AppImage has to serve the whole fleet, so the token is runtime config
// (electron/deviceConfig.js) and must not be inlined.
//
// Plain `expo export` cannot be trusted to do that. Every `EXPO_PUBLIC_*` value is inlined at bundle time,
// and Expo loads `.env` itself — where a developer's own `EXPO_PUBLIC_DEVICE_BOOTSTRAP_TOKEN` normally lives
// for Android work. Passing an empty value on the command line does NOT win: `.env` still takes effect, and a
// verified export showed the token inlined anyway. Nothing in the build fails, so the mistake ships silently.
//
// So: read `.env` here, drop that one key, hand the rest to expo with `EXPO_NO_DOTENV=1` so expo does not
// re-add it, then FAIL if a JWT is found in the output. The check at the end is the part that matters — the
// filtering is the fix, the assertion is what keeps it fixed.

import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import process from 'node:process'

const ROOT = path.resolve(import.meta.dirname, '..')
const DIST = path.join(ROOT, 'dist')

// Device-bound: must never be inlined into a fleet-wide artifact.
const FORBIDDEN_KEYS = ['EXPO_PUBLIC_DEVICE_BOOTSTRAP_TOKEN']

/** Minimal .env reader — enough for `KEY=value` lines, which is all this file ever contains. */
function readEnvFile(file) {
  if (!fs.existsSync(file)) return {}
  const values = {}
  for (const rawLine of fs.readFileSync(file, 'utf8').split('\n')) {
    const line = rawLine.trim()
    if (!line || line.startsWith('#')) continue
    const separator = line.indexOf('=')
    if (separator < 1) continue
    const key = line.slice(0, separator).trim()
    let value = line.slice(separator + 1).trim()
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1)
    }
    values[key] = value
  }
  return values
}

const fileEnv = readEnvFile(path.join(ROOT, '.env'))
const dropped = FORBIDDEN_KEYS.filter((key) => key in fileEnv || key in process.env)
for (const key of FORBIDDEN_KEYS) delete fileEnv[key]

const env = { ...process.env, ...fileEnv, EXPO_NO_DOTENV: '1' }
for (const key of FORBIDDEN_KEYS) delete env[key]

if (dropped.length) console.log(`[export-linux] withheld from the bundle: ${dropped.join(', ')}`)
// `--clear` is not optional here. EXPO_PUBLIC_* values are inlined into the transform output, so Metro's
// cache holds a bundle that still carries whatever the last export inlined — a cached build reproduced the
// exact same bundle hash, token and all, and the withheld variable had no effect whatsoever.
console.log('[export-linux] exporting web bundle (cache cleared — inlined env is part of the transform)…')
execFileSync('npx', ['expo', 'export', '--platform', 'web', '--clear'], { cwd: ROOT, env, stdio: 'inherit' })

// --- the assertion that keeps this honest -------------------------------------------------------
//
// Grep the produced bundle for a compact JWS header. This catches the token arriving by any route — .env,
// .env.local, a shell variable, a future config layer — rather than only the one we filtered.

const webDir = path.join(DIST, '_expo', 'static', 'js', 'web')
const bundles = fs.existsSync(webDir) ? fs.readdirSync(webDir).filter((name) => name.endsWith('.js')) : []
if (!bundles.length) {
  console.error('\n✗ no JS bundle found in dist/ — the export did not produce anything to check.\n')
  process.exit(1)
}

const offenders = []
for (const name of bundles) {
  const source = fs.readFileSync(path.join(webDir, name), 'utf8')
  for (const match of new Set([...source.matchAll(/eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+/g)].map((m) => m[0]))) {
    let claims = null
    try {
      claims = JSON.parse(Buffer.from(match.split('.')[1], 'base64url').toString('utf8'))
    } catch {
      continue // not a JWT, just base64-looking data
    }
    if (claims && typeof claims === 'object' && ('did' in claims || claims.kind === 'bootstrap')) {
      offenders.push({ bundle: name, did: claims.did ?? '(unknown)' })
    }
  }
}

if (offenders.length) {
  console.error('\n✗ a device credential was inlined into the Linux bundle — refusing to package it.')
  for (const offender of offenders) console.error(`    ${offender.bundle}: device ${offender.did}`)
  console.error('\n  Every unit built from this artifact would claim that same device identity.')
  console.error('  Provision per unit instead: <userData>/device-config.json → "bootstrapToken".\n')
  process.exit(1)
}

console.log('✓ dist/ contains no device credential — safe to package for the fleet.')
