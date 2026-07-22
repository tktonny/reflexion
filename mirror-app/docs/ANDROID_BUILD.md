# Android Studio / release build

## 1. Prerequisites

- Android Studio with the SDK/NDK versions requested by Gradle.
- JDK 17.
- A reachable HTTPS deployment of `caregiver-server`.
- A unique device bootstrap token from `npm run provision:device` in that server project.
- A private release keystore. Never publish an APK signed with the Android debug key.

## 2. Mirror configuration

Create `mirror-app/.env` from `.env.example`:

```bash
EXPO_PUBLIC_API_BASE=https://api.example.com
EXPO_PUBLIC_CONVERSATION_MODE=ws
EXPO_PUBLIC_DEVICE_BOOTSTRAP_TOKEN=<this-device-only-bootstrap-token>
```

Do not define `EXPO_PUBLIC_QWEN_API_KEY`, `MONGODB_URI`, `QWEN_API_KEY` or any other provider/database key in the Mirror project. Expo public variables are compiled into the application.

Camera and microphone are enabled in `app.json` and the generated Android manifest. The release is not muted: native PCM playback uses `AudioTrack`, and capture uses `AudioRecord` with AEC/NS/AGC where the device supports them.

## 3. Release signing

The Gradle project fails a release build when release signing is missing. Configure either the existing macOS Keychain entry or these environment variables before opening Android Studio/building:

```bash
export REFLEXION_MIRROR_STORE_FILE=/absolute/path/reflexion-mirror-release.p12
export REFLEXION_MIRROR_STORE_PASSWORD='<store password>'
export REFLEXION_MIRROR_KEY_ALIAS='reflexion-mirror'
export REFLEXION_MIRROR_KEY_PASSWORD='<key password>'
```

Keep the keystore and passwords outside Git. Preserve the same key for every future update of `com.reflexion.mirror`.

## 4. Build

The native dependency graph needs a larger Gradle heap during release dex merge. Keep the generated
`android/gradle.properties` at `-Xmx4096m`, `MaxMetaspaceSize=1024m`, with at most two workers.

From Android Studio, open `mirror-app/android`, select the release variant, then use **Build > Generate Signed App Bundle or APK**.

Equivalent commands:

```bash
cd mirror-app/android
./gradlew clean assembleRelease
./gradlew bundleRelease
```

Outputs:

- APK: `android/app/build/outputs/apk/release/app-release.apk`
- Play/managed distribution bundle: `android/app/build/outputs/bundle/release/app-release.aab`

An APK embeds its API URL and device bootstrap value. If every mirror receives a different factory token, build/provision per device or replace the environment-token channel with your MDM/secure factory injection mechanism before fleet scale.

## 5. Device acceptance

1. Install with `adb install -r app-release.apk`.
2. Confirm Android asks for microphone permission when a voice session starts; grant it. Camera permission must only be requested by an explicit diagnostics or separately consented camera flow, not by a daily voice conversation.
3. Confirm the pairing screen displays a six-digit code and the Caregiver app can claim it.
4. Start daily assistant and daily check-in sessions; confirm audio plays, microphone-active disclosure remains visible, barge-in works and neither side is cut off. Verify camera availability separately from the daily conversation flow.
5. End manually and automatically; confirm the goodbye finishes before the session is saved.
6. Force-stop/relaunch during a session; confirm the stale backend session is abandoned and only one conversation remains active.
7. Disconnect/reconnect the network; confirm the transcript completion is queued and later uploaded once, without clearing a newer active session.
8. Check the caregiver backend monitoring summary after the worker processes the session.
