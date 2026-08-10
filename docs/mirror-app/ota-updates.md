# Mirror OTA updates (EAS Update, manual trigger)

Ship JS changes to a mirror without building and side-loading a 277 MB APK.

## What OTA can and cannot ship

| Change | OTA? |
|---|---|
| Prompts, conversation flow, memory logic, self-check logic, screens | ✅ |
| **`EXPO_PUBLIC_*` values** (barge-in thresholds, feature flags, API base) — they are inlined into the bundle at build time, so they ride along | ✅ |
| `modules/expo-pcm-audio`, a new native dependency, Android permissions / `app.json` native config, app version bump | ❌ full APK |

In practice most iteration is JS: the barge-in tuning loop, prompt wording, and flow changes are all OTA-able.

## Design decisions (and why)

- **Manual trigger, never automatic.** The mirror is a kiosk that can stay powered for days, so "check on launch" would rarely fire; but checking periodically and reloading whenever something lands risks restarting the JS runtime **mid-conversation** and cutting an elder off. So `checkAutomatically: "NEVER"` and an operator presses **Admin diagnostics → Check for app update**. Download and apply are two separate steps, with a confirmation before the reload.
- **`runtimeVersion` policy `fingerprint`.** It bumps whenever anything affecting the native runtime changes, which is what stops a JS bundle from landing on an incompatible native build. expo-updates also rolls back to the last working update if a bad one gets through.
- **Two channels.** Test devices on `preview`, real users on `production`, so nothing reaches a household before a device has run it.

## One-time setup

1. **Create the EAS project** (interactive — needs the Expo account):
   ```bash
   cd mirror-app && eas init
   ```
   Copy the printed project id into `app.json` → `updates.url`, replacing `REPLACE_WITH_EAS_PROJECT_ID`:
   ```json
   "updates": { "url": "https://u.expo.dev/<project-id>", … }
   ```
2. **Create the channels** (a local build cannot have one injected by EAS Build, so they must exist server-side):
   ```bash
   eas channel:create preview
   eas channel:create production
   ```

## Building an OTA-capable APK

`mirror-app/android/` is hand-maintained (it carries the release-signing guard and the local native module), so **never run `expo prebuild`** — it would overwrite `build.gradle`. Use the updates-only sync instead:

```bash
cd mirror-app
npx expo-updates configuration:syncnative --platform android --workflow generic
cd android && ./gradlew assembleRelease
```

`syncnative` writes only the updates keys into `AndroidManifest.xml` — verified to leave `build.gradle` untouched:

```xml
<meta-data android:name="expo.modules.updates.ENABLED" android:value="true"/>
<meta-data android:name="expo.modules.updates.EXPO_UPDATES_CHECK_ON_LAUNCH" android:value="NEVER"/>
<meta-data android:name="expo.modules.updates.EXPO_UPDATE_URL" android:value="https://u.expo.dev/<project-id>"/>
<meta-data android:name="expo.modules.updates.UPDATES_CONFIGURATION_REQUEST_HEADERS_KEY" android:value="{&quot;expo-channel-name&quot;:&quot;preview&quot;}"/>
<meta-data android:name="expo.modules.updates.EXPO_RUNTIME_VERSION" android:value="@string/expo_runtime_version"/>
```

Since `android/` is gitignored, this step is **per build host** — it is not carried by the repo.

To build a device for the production channel, change `expo-channel-name` in `app.json` before syncing.

## Publishing an update

```bash
cd mirror-app
eas update --branch preview --message "lower barge-in threshold to 0.06"
```

Then on the mirror: **Admin diagnostics → Check for app update** → *Restart now* (only while nobody is mid-conversation). The **Running bundle** row on the same screen shows which bundle is live (`embedded bundle · <runtime>` or `OTA <id> · <date>`).

Promote a verified update to real users:
```bash
eas channel:edit production --branch preview   # or publish to --branch production directly
```

## Rollback

```bash
eas update:republish --group <previous-update-group>
```
Then trigger the update on the device. A bundle that crashes on launch is rolled back automatically by expo-updates, but do not rely on that as the plan.

## Limits / not done yet

- **Not verified end-to-end on a device.** The build recipe (`syncnative` → gradle) is verified; publishing and applying an update needs the EAS project id from step 1.
- `runtimeVersion` uses `fingerprint`, so **any native change produces a new runtime and older installed APKs stop receiving updates** — that is the intended safety behaviour, but it means native changes still require distributing a new APK.
- Automatic apply-when-idle was considered and rejected (see above); if it is ever added, it must gate on "no conversation active" and on the mirror being visibly idle.
