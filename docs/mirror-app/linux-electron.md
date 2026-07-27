# Mirror app — Linux (Ubuntu) build via Electron

The mirror runs on two kinds of hardware: **Android** smart-mirror units (the primary target) and **Ubuntu OS** units. The app is one Expo / React Native codebase; this doc covers the **Linux** delivery.

## Why Electron (and not a native Linux app)

React Native has no mature native **Linux desktop** target, and the mirror's production Android path leans on native modules that don't exist on Linux:

- `modules/expo-pcm-audio` — native 16 kHz PCM capture/playback (Android/iOS only)
- `onnxruntime-react-native` — the "Hello Aria" wake word (native only)
- `react-native-webrtc` + the `ws`/`webrtc` direct-to-Qwen realtime transports (native only)

So the Linux build is the **same app compiled for web** (`react-native-web`, already a dependency) and wrapped in **Electron** (Chromium + Node). One codebase → two apps: **Android APK** and **Linux AppImage/deb**.

```
Android unit                          Ubuntu unit
────────────                          ────────────
RN native                             Electron (Chromium)
 ├─ expo-pcm-audio (native 16k PCM)    ├─ dist/ (react-native-web SPA)  ← same UI
 ├─ onnxruntime wake word             ├─ Web Audio (getUserMedia)
 └─ ws/webrtc direct → Qwen           └─ relay transport → local Node relay → Qwen
```

## What differs from Android (read this before shipping)

The Linux build is **functionally lighter** than Android. It runs the `relay` conversation transport with browser audio:

| Capability | Android | Linux (Electron) |
|---|---|---|
| Conversation transport | `ws`/`webrtc` direct realtime to Qwen | `relay` (browser ↔ local Node relay ↔ Qwen) |
| Mic capture | native `expo-pcm-audio` (16 kHz PCM) | Web Audio `getUserMedia` |
| Wake word | native onnxruntime ("Hello Aria") | **none yet** → tap-to-start (future: web wake word) |
| Echo cancellation | native / semantic_vad on device | Chromium AEC + relay-side handling |
| Provider auth | short-lived **ticket** from backend (key never on device) | relay holds a **Qwen key** (appliance config, see below) |

These are the known gaps to close before a Linux unit is production-equivalent — tracked in "Production hardening" below.

## Files

- `electron/main.js` — Electron main process. Serves the exported SPA (`dist/`) over a local HTTP server (file:// breaks Expo Router history routing), auto-grants microphone permission (the appliance needs it), and optionally spawns the relay.
- `electron/preload.js` — minimal, context-isolated; the seam for any future Linux-native bridge.
- `package.json` → `build` block — electron-builder config (Linux `AppImage` + `deb`, `extraMetadata.main` points the packaged app at `electron/main.js` so Expo's own `main` is untouched).
- `app.json` → `web.output: "single"` — the mirror is a client-only SPA (no `+api` routes), so single-file output is correct and is what Electron loads.

## Build (on a build host — Linux or macOS)

```bash
cd mirror-app
npm install                      # pulls electron + electron-builder (devDeps)
npm run electron:build           # export web SPA → build orch bundle → electron-builder --linux
# → dist-linux/Reflexion Mirror-<ver>.AppImage  and  .deb
```

`electron:build` runs, in order: `expo export --platform web` (→ `dist/`), `build:orch` (→ `server/generated/orchestration.mjs`, needed by the relay), then `electron-builder`. Cross-building the Linux targets from macOS works for AppImage; `.deb` is most reliable built on Linux.

## Run / dev

```bash
npm run electron:export          # produce dist/ (once, or after any app change)
QWEN_API_KEY=sk-... npm run electron:dev
```

Environment (all optional):

| Var | Default | Effect |
|---|---|---|
| `QWEN_API_KEY` / `DASHSCOPE_API_KEY` | — | relay's provider key. **Without it the UI still loads but the conversation is unavailable.** |
| `REFLEXION_MIRROR_KIOSK` | `1` | `0` = windowed (not fullscreen kiosk) |
| `REFLEXION_MIRROR_SKIP_RELAY` | `0` | `1` = don't spawn the local relay (point at an external one) |
| `REFLEXION_MIRROR_DEVTOOLS` | `0` | `1` = open Chromium devtools |
| `REFLEXION_MIRROR_WEB_PORT` / `REFLEXION_RELAY_PORT` | `8899` / `8787` | local ports |

## Deploy on an Ubuntu unit

1. Install: `sudo dpkg -i "Reflexion Mirror-<ver>.deb"` (or run the AppImage directly).
2. Put the Qwen key in appliance config (e.g. a root-only `/etc/reflexion/mirror.env` with `QWEN_API_KEY=...`) and launch the app with that environment — **never bake the key into the distributed artifact.**
3. Autostart in kiosk: a `systemd` user service or the desktop's autostart, launching `reflexion-mirror` on boot; kiosk mode is on by default.

## Security note

Android devices never hold a provider key — they mint short-lived **tickets** from the backend. The relay transport used here holds a **Qwen key** on the unit. That is acceptable for a **controlled appliance** (key in a root-only config file, not in the shipped app), but it is weaker than the Android model. Moving the Linux build onto the ticket flow is the top production-hardening item.

## Production hardening (follow-ups, not in the first cut)

- **Ticket-based auth** for the Linux relay (parity with Android; remove the on-device key).
- **Wake word** on Linux (a web/wasm wake-word, or a small native helper via the preload bridge) — today it's tap-to-start.
- **Echo/AEC** validation on the real speaker+mic hardware; tune relay-side turn detection.
- **Auto-update** for the AppImage (electron-updater) and crash/telemetry reporting.
