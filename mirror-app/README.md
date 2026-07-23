# Reflexion Mirror

Expo/React Native application for the physical Reflexion mirror. Android production uses native PCM capture/playback and Qwen realtime over a direct authenticated WebSocket; the shared Reflexion backend handles device identity, pairing, session tickets, transcript ingestion and longitudinal processing.

## Security boundary

- The APK contains no MongoDB, Qwen account, object-store, embedding or web-search key.
- A device-specific bootstrap token is used only to begin pairing.
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
EXPO_PUBLIC_DEVICE_BOOTSTRAP_TOKEN=<token returned by reflexion-server provision:device>
```

The bootstrap token must belong to this physical device. Provider credentials belong only in `reflexion-server/.env`.

## Verify

```bash
npm ci
npm run typecheck
npm run test:turn-taking
```

## Android Studio / Gradle

The checked-out `android/` project includes the local `expo-pcm-audio` native module and release-signing guard. See [`docs/ANDROID_BUILD.md`](./docs/ANDROID_BUILD.md) for configuration, signing and APK/AAB commands.

The production path is `ws` or `webrtc`. The local Node relay remains a web-development diagnostic only and is not needed by the Android release.
