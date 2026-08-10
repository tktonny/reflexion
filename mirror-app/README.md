# Reflexion Mirror

Expo/React Native application for the physical Reflexion mirror. Android production uses native PCM capture/playback and Qwen realtime over a direct authenticated WebSocket; the shared Reflexion backend handles device identity, pairing, session tickets, transcript ingestion and longitudinal processing.

## Security boundary

- The APK contains no MongoDB, Qwen account, object-store, embedding or web-search key.
- A fresh mirror generates a device-local identity in platform secure storage and uses it only to begin pairing.
- Existing factory bootstrap tokens remain supported as a backwards-compatible installation path.
- Pairing exchanges a one-time ticket for rotating device credentials stored in Android SecureStore.
- Every conversation creates a backend session, then obtains a short-lived Qwen ticket for that session.
- Ordered transcript events are uploaded to `/api/v1`; failed completions remain in a local durable outbox.
- Client-side screening previews are disabled in release builds. Clinical/research observations come from the backend pipeline.

## Configure

```bash
cp .env.example .env
```

Set at minimum:

```bash
EXPO_PUBLIC_API_BASE=https://reflexion.production.tktonny.top
EXPO_PUBLIC_CONVERSATION_MODE=ws
# Optional legacy path for an already-provisioned unit:
# EXPO_PUBLIC_DEVICE_BOOTSTRAP_TOKEN=<token returned by reflexion-server provision:device>
```

Fresh mirrors do not need a pre-provisioned token. On first boot they generate and retain a local identity, register with the API, and show a short-lived pairing QR/code. Provider credentials belong only in `reflexion-server/.env`.

## Verify

```bash
npm ci
npm run typecheck
npm run test:turn-taking
```

## Local web demo

The browser demo is a self-contained simulation of the mirror experience. It uses local browser storage and does not pair with a caregiver account or send production requests.

```bash
npm ci
npm run web:demo
```

Open the URL shown by Expo. The demo opens automatically and its `Test menu` exposes the boot, Wi-Fi, pairing, conversation, routine, family-message, consent and research fixtures. Browser speech and microphone support vary by browser; use `Continue check-in` when speech recognition is unavailable. Microphone testing requires `localhost` or HTTPS.

To create a static web build instead:

```bash
npm run export:web-demo
npx serve dist
```

## Android Studio / Gradle

The checked-out `android/` project includes the local `expo-pcm-audio` native module and release-signing guard. See [`docs/ANDROID_BUILD.md`](./docs/ANDROID_BUILD.md) for configuration, signing and APK/AAB commands.

## Provisioning a device

Pairing is now a first-boot flow: connect the mirror to Wi-Fi, wait for its QR/code, scan or enter it in the caregiver app, choose the loved one, and leave the mirror to finish automatically. A factory bootstrap token is still supported for existing fleets; see [`../docs/mirror-app/device-provisioning.md`](../docs/mirror-app/device-provisioning.md) for migration and recovery details.

## Linux (Ubuntu) via Electron

The same app also ships as a Linux desktop app — the web build (`react-native-web`) wrapped in Electron (`electron/`).

```bash
npm install                 # electron + electron-builder devDeps
QWEN_API_KEY=sk-... npm run electron:dev     # run locally (export dist/ first: npm run electron:export)
npm run electron:build      # → dist-linux/*.AppImage + *.deb
```

It's functionally lighter than Android (relay transport + Web Audio, tap-to-start, no native wake word). See [`../docs/mirror-app/linux-electron.md`](../docs/mirror-app/linux-electron.md) for the architecture, deployment and production-hardening notes.

The production path is `ws` or `webrtc`. The local Node relay remains a web-development diagnostic only and is not needed by the Android release.
