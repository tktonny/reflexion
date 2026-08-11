import { Platform } from 'react-native'
import * as Updates from 'expo-updates'

// Manual over-the-air updates for the mirror.
//
// WHY MANUAL: the mirror is a kiosk that can stay powered for days, so "check on launch" (the
// expo-updates default) would almost never fire. The opposite — checking periodically and reloading
// whenever something arrives — risks reloading the app **mid-conversation**, cutting an elder off in the
// middle of a check-in. So the app never checks by itself (`checkAutomatically: NEVER` in app.json) and
// an operator triggers it from the settings screen, where they can see the mirror is idle.
//
// WHAT OTA CAN AND CANNOT SHIP: the JS bundle and its assets, which includes every `EXPO_PUBLIC_*` value
// because those are inlined at bundle time — so prompts, conversation flow, self-check logic and the
// barge-in thresholds are all updatable without a 277 MB APK. Native changes are NOT: touching
// modules/expo-pcm-audio, adding a native dependency, or editing the Android config still needs a build.
// The `fingerprint` runtimeVersion policy is what keeps a JS bundle from ever landing on an incompatible
// native runtime; expo-updates rolls back to the last working update if a bad one does get through.
//
// TWO IMPLEMENTATIONS BEHIND ONE INTERFACE. expo-updates has **no web implementation**, so in the Linux
// (Electron) build `Updates.isEnabled` is always false and this module used to report "OTA disabled" on
// every Ubuntu unit — the Linux fleet had no update path at all. There, the Electron shell does the work
// (electron/bundleUpdates.js) and this module talks to it over the preload bridge. The distinction the
// Linux side adds is that the shell itself is separately updatable: replacing the renderer bundle cannot
// change Chromium, the main process, or the network stack, so that is a second, rarer channel.

export type UpdateOutcome =
  | { kind: 'disabled'; detail: string }
  | { kind: 'up_to_date'; detail: string }
  | { kind: 'downloaded'; detail: string }
  | { kind: 'failed'; detail: string }

type ShellBundleState = {
  version: string | null
  embedded: boolean
  pending: string | null
  lastFailed: string | null
  packagedVersion: string | null
}

type ShellUpdatesState = {
  updateBase: string
  bundle: ShellBundleState
  shell: { version: string; supported: boolean; arch: string }
}

type ShellUpdatesBridge = {
  state(): Promise<ShellUpdatesState>
  bundleCheck(): Promise<{ kind: UpdateOutcome['kind']; detail: string; version?: string }>
  bundleApply(): Promise<{ ok: boolean; error?: string; version?: string }>
  bundleRollback(): Promise<{ ok: boolean; from?: string | null; to?: string | null }>
  shellCheck(): Promise<{ kind: UpdateOutcome['kind']; detail: string; version?: string }>
  shellApply(): Promise<{ ok: boolean; error?: string; restartRequired?: boolean }>
  shellDiscard(): Promise<{ ok: boolean }>
  reload(): Promise<{ ok: boolean }>
  relaunch(): Promise<{ ok: boolean }>
  markBooted(): Promise<{ ok: boolean }>
}

function shell(): ShellUpdatesBridge | null {
  if (Platform.OS !== 'web' || typeof window === 'undefined') return null
  return (window as unknown as { reflexionUpdates?: ShellUpdatesBridge }).reflexionUpdates ?? null
}

/** True on the Linux appliance, where OTA is the shell's job rather than expo-updates'. */
export function shellUpdatesAvailable(): boolean {
  return Boolean(shell())
}

// `currentUpdateLabel` is called during render and must stay synchronous, but the shell's state only comes
// over async IPC — so it is cached here and refreshed by initOtaUpdates() and after every operation.
let cachedShellState: ShellUpdatesState | null = null

async function refreshShellState(): Promise<ShellUpdatesState | null> {
  const bridge = shell()
  if (!bridge) return null
  try {
    cachedShellState = await bridge.state()
  } catch {
    /* keep whatever we had; a failed status read must not affect the running app */
  }
  return cachedShellState
}

/**
 * Call once at app boot.
 *
 * On Linux this is load-bearing, not diagnostic: it tells the shell that this bundle actually rendered. The
 * shell rolls a freshly applied bundle back on the next launch if that signal never arrives, which is what
 * makes a bad update self-healing instead of a support visit — so it must run unconditionally and early,
 * before any screen can fail for an unrelated reason.
 */
export async function initOtaUpdates(): Promise<void> {
  const bridge = shell()
  if (!bridge) return
  try {
    await bridge.markBooted()
  } catch {
    /* never block boot on this */
  }
  await refreshShellState()
}

/** Short human line describing which bundle is running — shown next to the update button. */
export function currentUpdateLabel(): string {
  if (shellUpdatesAvailable()) {
    const state = cachedShellState
    if (!state) return 'checking…'
    const bundle = state.bundle.embedded
      ? `packaged bundle${state.bundle.packagedVersion ? ` ${state.bundle.packagedVersion}` : ''}`
      : `OTA bundle ${state.bundle.version}`
    const pending = state.bundle.pending ? ` · ${state.bundle.pending} ready` : ''
    return `${bundle} · shell ${state.shell.version}${pending}`
  }
  if (!Updates.isEnabled) return 'OTA disabled (dev build)'
  if (Updates.isEmbeddedLaunch) return `embedded bundle · ${Updates.runtimeVersion ?? 'unknown runtime'}`
  const id = Updates.updateId ? Updates.updateId.slice(0, 8) : 'unknown'
  const created = Updates.createdAt ? ` · ${Updates.createdAt.toISOString().slice(0, 16).replace('T', ' ')}` : ''
  return `OTA ${id}${created}`
}

/**
 * Check for an update and download it if there is one. Does NOT reload — applying is a separate,
 * explicit step so the caller can confirm before restarting the app.
 */
export async function checkAndDownload(): Promise<UpdateOutcome> {
  const bridge = shell()
  if (bridge) {
    try {
      const outcome = await bridge.bundleCheck()
      await refreshShellState()
      return { kind: outcome.kind, detail: outcome.detail } as UpdateOutcome
    } catch (error) {
      return { kind: 'failed', detail: error instanceof Error ? error.message.slice(0, 140) : 'Update check failed.' }
    }
  }
  if (!Updates.isEnabled) {
    // True in Expo Go and in any build made without expo-updates configured; not an error.
    return { kind: 'disabled', detail: 'Updates are not enabled in this build.' }
  }
  try {
    const check = await Updates.checkForUpdateAsync()
    if (!check.isAvailable) {
      return { kind: 'up_to_date', detail: `Already on the newest bundle (${currentUpdateLabel()}).` }
    }
    const fetched = await Updates.fetchUpdateAsync()
    if (!fetched.isNew) {
      return { kind: 'up_to_date', detail: 'Nothing new to install.' }
    }
    return { kind: 'downloaded', detail: 'Update downloaded. Restart to apply it.' }
  } catch (error) {
    // A failed check must never take the mirror down — it keeps running the current bundle.
    const message = error instanceof Error ? error.message : String(error)
    return { kind: 'failed', detail: message.slice(0, 140) }
  }
}

/**
 * Apply a downloaded update by reloading the app. ONLY call when the mirror is idle — this restarts the
 * JS runtime, so triggering it during a conversation would cut the elder off mid-sentence.
 */
export async function applyDownloadedUpdate(): Promise<void> {
  const bridge = shell()
  if (bridge) {
    const applied = await bridge.bundleApply()
    // Reloading is what actually swaps the bundle: the shell re-resolves which directory it serves. Skipping
    // it on failure keeps the working bundle on screen rather than reloading into the same state.
    if (applied.ok) await bridge.reload()
    return
  }
  await Updates.reloadAsync()
}

/**
 * Check for a new Electron SHELL (Linux only) — Chromium, the main process, the network stack, the relay.
 * Separate from the bundle channel because it is a ~97 MB download and needs a full relaunch, so it is run
 * rarely and deliberately rather than as part of a routine update check.
 */
export async function checkShellUpdate(): Promise<UpdateOutcome> {
  const bridge = shell()
  if (!bridge) return { kind: 'disabled', detail: 'Shell updates only apply to the Ubuntu build.' }
  try {
    const outcome = await bridge.shellCheck()
    await refreshShellState()
    return { kind: outcome.kind, detail: outcome.detail } as UpdateOutcome
  } catch (error) {
    return { kind: 'failed', detail: error instanceof Error ? error.message.slice(0, 140) : 'Shell update check failed.' }
  }
}

/** Install a downloaded shell and relaunch. ONLY when the mirror is idle — this restarts the whole app. */
export async function applyShellUpdate(): Promise<{ ok: boolean; error?: string }> {
  const bridge = shell()
  if (!bridge) return { ok: false, error: 'Shell updates only apply to the Ubuntu build.' }
  const applied = await bridge.shellApply()
  if (!applied.ok) return { ok: false, error: applied.error }
  await bridge.relaunch()
  return { ok: true }
}

/** Go back to the previous renderer bundle. The manual counterpart to automatic rollback. */
export async function rollbackBundle(): Promise<{ ok: boolean; to?: string | null }> {
  const bridge = shell()
  if (!bridge) return { ok: false }
  const result = await bridge.bundleRollback()
  if (result.ok) await bridge.reload()
  return result
}
