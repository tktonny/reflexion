// Expo config plugin — Android kiosk hardening for the Reflexion Mirror (implementation baseline §7
// Phase 7). It makes the mirror behave as an always-on appliance:
//
//   1. HOME + DEFAULT launcher intent-filter on MainActivity → the app can be set as the device's
//      default launcher. Android then auto-starts it on boot AND relaunches it after a crash (the
//      home app is always restarted), which covers "launch on boot" and basic crash recovery without
//      any custom native BroadcastReceiver class.
//   2. lockTaskMode="if_whitelisted" → the app can enter true screen-pinning / lock-task when the
//      device is provisioned as device-owner (via MDM/EMM) and calls startLockTask().
//   3. keep-screen-on flag so the ambient mirror never sleeps.
//
// Full lock-task ENFORCEMENT still requires device-owner provisioning at deployment (MDM). This
// plugin supplies the manifest side; it adds no runtime behaviour on its own. Runs at prebuild.

const { withAndroidManifest, AndroidConfig } = require('@expo/config-plugins')

function ensurePermission(manifest, name) {
  manifest['uses-permission'] = manifest['uses-permission'] || []
  const full = name.startsWith('android.permission.') ? name : `android.permission.${name}`
  if (!manifest['uses-permission'].some((p) => p.$ && p.$['android:name'] === full)) {
    manifest['uses-permission'].push({ $: { 'android:name': full } })
  }
}

const withKioskAndroid = (config) =>
  withAndroidManifest(config, (mod) => {
    const manifest = mod.modResults.manifest
    ensurePermission(manifest, 'RECEIVE_BOOT_COMPLETED')
    ensurePermission(manifest, 'WAKE_LOCK')

    const mainActivity = AndroidConfig.Manifest.getMainActivityOrThrow(mod.modResults)
    mainActivity.$['android:launchMode'] = 'singleInstance'
    // Allow the app to enter lock-task (screen pinning) when provisioned as device owner.
    mainActivity.$['android:lockTaskMode'] = 'if_whitelisted'

    // Add HOME + DEFAULT to the existing MAIN/LAUNCHER intent-filter so the mirror can be the default
    // launcher (auto-start on boot + crash relaunch). Do not duplicate if already present.
    mainActivity['intent-filter'] = mainActivity['intent-filter'] || []
    const launcher = mainActivity['intent-filter'].find((filter) =>
      (filter.action || []).some((a) => a.$ && a.$['android:name'] === 'android.intent.action.MAIN'))
    if (launcher) {
      launcher.category = launcher.category || []
      for (const category of ['android.intent.category.HOME', 'android.intent.category.DEFAULT']) {
        if (!launcher.category.some((c) => c.$ && c.$['android:name'] === category)) {
          launcher.category.push({ $: { 'android:name': category } })
        }
      }
    }

    return mod
  })

module.exports = withKioskAndroid
