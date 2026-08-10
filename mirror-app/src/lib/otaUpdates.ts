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

export type UpdateOutcome =
  | { kind: 'disabled'; detail: string }
  | { kind: 'up_to_date'; detail: string }
  | { kind: 'downloaded'; detail: string }
  | { kind: 'failed'; detail: string }

/** Short human line describing which bundle is running — shown next to the update button. */
export function currentUpdateLabel(): string {
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
  await Updates.reloadAsync()
}
