// Prove the startup hardware self-check's decision table WITHOUT a device: bundle the pure
// recommendMode logic (esbuild, no react-native) and assert every capability combination maps
// to the right conversation version. Run: node server/smoke-hwcheck.mjs   (exit 0 = PASS)

import { build } from 'esbuild'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const out = join(mkdtempSync(join(tmpdir(), 'hwcheck-')), 'recommendMode.mjs')
await build({
  entryPoints: ['src/lib/recommendMode.ts'],
  outfile: out,
  bundle: true,
  format: 'esm',
  platform: 'node',
  logLevel: 'silent',
})
const { recommendMode } = await import(out)

const T = true, F = false
// [platform, {micOk, relayOk, turnOk, rtOk}, expectedMode]
const cases = [
  ['web',     { micOk: T, relayOk: T, turnOk: T, rtOk: F }, 'relay'], // web + relay + mic → v1 realtime
  ['web',     { micOk: T, relayOk: F, turnOk: T, rtOk: F }, 'http'],  // web, no relay → v2 turn-based
  ['web',     { micOk: F, relayOk: F, turnOk: F, rtOk: F }, 'none'],  // web, nothing usable
  ['android', { micOk: T, relayOk: F, turnOk: T, rtOk: T }, 'ws'],    // native + PCM stream → v3
  ['android', { micOk: T, relayOk: F, turnOk: T, rtOk: F }, 'http'],  // native, no PCM → v2
  ['android', { micOk: T, relayOk: T, turnOk: F, rtOk: F }, 'relay'], // native fallback to relay
  ['android', { micOk: F, relayOk: T, turnOk: T, rtOk: T }, 'none'],  // no mic → nothing
]

let pass = true
for (const [platform, caps, expected] of cases) {
  const { mode, reason } = recommendMode(platform, caps)
  const ok = mode === expected
  if (!ok) pass = false
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${platform.padEnd(8)} ${JSON.stringify(caps)} → ${mode} (want ${expected})  ${reason}`)
}
console.log(`\nRESULT: ${pass ? 'PASS' : 'FAIL'} — hardware-check decision table`)
process.exit(pass ? 0 : 1)
