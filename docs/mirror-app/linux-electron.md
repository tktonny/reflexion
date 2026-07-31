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
| Echo cancellation | native / semantic_vad on device | Chromium AEC + semantic_vad (ticket model) |
| Provider auth | short-lived **ticket** from backend (key never on device) | **same** — short-lived ticket, **no Qwen key on device** (see "Keyless auth") |
| Backend calls | direct `fetch` to the API origin | **proxied** through the local server (same-origin, no CORS — see "Backend proxy") |
| Network setup | opens the OS settings panel | **in-app**: Wi-Fi / hotspot / Bluetooth tethering via `nmcli` + `bluetoothctl` (see `network-setup.md`) |

These are the known gaps to close before a Linux unit is production-equivalent — tracked in "Production hardening" below.

## Keyless auth (same model as the Android APK)

**No Qwen key lives on the Linux device** — identical trust model to Android:

1. The AppImage embeds only the **backend API URL** (`EXPO_PUBLIC_API_BASE`) and a **per-device bootstrap token**.
2. The renderer pairs with the backend (bootstrap → device credential) and, per conversation, mints a **short-lived Qwen realtime ticket** from `POST /api/v1/sessions/:id/realtime-tickets` (device-authenticated, plain `fetch` — works in the renderer).
3. The renderer hands that ticket + region endpoint + model to the local relay in the **first WS message** (`reflexion.auth`; never in the URL/logs).
4. The relay opens the header-authed Qwen WS **with the ticket** (`server/relay.mjs` `connectUpstream(url, ticket)`), using semantic_vad to match the qwen3.5-omni ticket. Chromium WebSockets can't set the `Authorization` header — that is the *only* reason a Node relay sits in the middle.

The relay's raw-`QWEN_API_KEY` path still exists **for local web dev only** and is never used by the shipped Electron build.

## Backend proxy (why the mirror is same-origin)

The renderer is a page on `http://127.0.0.1:8899`, so calling the backend directly was **cross-origin**, and because the mirror sends `Authorization` / `Idempotency-Key` / `X-Device-Bootstrap` it was a *preflighted* request. Production's `CORS_ALLOWED_ORIGINS` listed only the admin SPA, so every preflight came back without `Access-Control-Allow-Origin` and **Chromium blocked every request before it left the device** — nothing reached the server, nothing was logged, and the unit reported "unable to reach the Reflexion service" against a healthy API. That is the failure a shipped AppImage hit in the field.

The local server now forwards `/api/*`, `/health` and `/healthcheck` upstream (`electron/apiProxy.js`), so every backend call is **same-origin**: no preflight, no allowlist to be missing from. The proxy strips `Origin`/`Referer`, sets the upstream `Host`, and turns an unreachable backend into the API's own `{error:{code,message,retryable}}` envelope instead of an HTML error page.

Consequences:

- **`EXPO_PUBLIC_API_BASE` is no longer required at export time** for the Electron build. The origin is resolved at *runtime* — `REFLEXION_API_BASE` env → `apiBase` in `<userData>/device-config.json` → production default — so a unit can be re-pointed without re-exporting the bundle. The startup log line states the resolved target.
- The renderer detects the shell via `window.reflexionMirror.apiProxy` and uses relative URLs, so even a baked `EXPO_PUBLIC_API_BASE` cannot reintroduce cross-origin calls (`src/config/apiUrl.ts`).
- Independently, the server always allows the mirror's loopback origins (`reflexion-server/src/app.ts`, `MIRROR_LOOPBACK_ORIGINS`) so **AppImages built before this change keep working**.

Tests: `npm run test:network`.

## Files

- `electron/main.js` — Electron main process. Serves the exported SPA (`dist/`) over a local HTTP server (file:// breaks Expo Router history routing), proxies backend paths, auto-grants microphone permission (the appliance needs it), registers the network-setup IPC handlers, and optionally spawns the relay.
- `electron/apiProxy.js` — the backend proxy above; a separate module so it is testable without the Electron binary.
- `electron/network.js` — privileged network control (`nmcli` / `bluetoothctl`) behind named operations, `execFile` with argument arrays only. See `network-setup.md`.
- `electron/preload.js` — context-isolated bridge: `reflexionMirror` (platform + `apiProxy` flag) and `reflexionNetwork` (one fixed IPC channel per network operation).
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

Set the bootstrap token at **export** time (baked into the SPA, like the APK — NO Qwen key). The API origin is runtime config now, so it no longer has to be baked:

```bash
EXPO_PUBLIC_DEVICE_BOOTSTRAP_TOKEN=<per-device token from `provision:device`> \
  npm run electron:export        # produce dist/
REFLEXION_API_BASE=https://reflexion.production.tktonny.top \
  npm run electron:dev           # launch the shell (no Qwen key needed)
```

Environment:

| Var | Where | Effect |
|---|---|---|
| `REFLEXION_API_BASE` | **runtime** | backend origin the shell proxies to; falls back to `device-config.json`, then the production default |
| `EXPO_PUBLIC_API_BASE` | export (optional) | still read as a runtime fallback for the proxy target; the renderer ignores it for URL building (it goes through the proxy) |
| `EXPO_PUBLIC_DEVICE_BOOTSTRAP_TOKEN` | export or first-run | per-device pairing token (see "Per-device provisioning") |
| ~~`QWEN_API_KEY`~~ | — | **not used** — the device is keyless; the relay authenticates with the renderer's ticket |
| `REFLEXION_MIRROR_KIOSK` | runtime | `0` = windowed (not fullscreen kiosk) |
| `REFLEXION_MIRROR_SKIP_RELAY` | runtime | `1` = don't spawn the local relay (point at an external one) |
| `REFLEXION_MIRROR_DEVTOOLS` | runtime | `1` = open Chromium devtools |
| `REFLEXION_MIRROR_WEB_PORT` / `REFLEXION_RELAY_PORT` | runtime | local ports (`8899` / `8787`) |

**Backend CORS:** no longer a deployment step — backend calls go through the local proxy and are same-origin, and the server allows the mirror's loopback origins regardless. See "Backend proxy" above for the failure this replaced.

**Per-device provisioning:** don't bake one shared bootstrap token into the shipped AppImage (violates the per-device rule). Either set `EXPO_PUBLIC_DEVICE_BOOTSTRAP_TOKEN` per unit at export, or leave it unset and have `electron/main.js` read a device-local token file and pass it to the renderer via the preload bridge to `persistBootstrapCredential()` once on first run.

## Deploy on an Ubuntu unit

1. Install: `sudo dpkg -i "Reflexion Mirror-<ver>.deb"` (or run the AppImage directly).
2. **Connect the unit to the internet.** A unit arrives with no network configured; the app's own setup screen joins Wi-Fi (home router or a phone hotspot), starts the mirror's hotspot, or tethers to a phone over Bluetooth. Requires `network-manager` (and `bluez` for Bluetooth) with the kiosk user permitted to control networking via polkit. Full detail and the on-device acceptance checklist: `network-setup.md`.
3. Pair the unit (provision a per-device bootstrap token via `provision:device`, then pair from the app). **No Qwen key is placed on the unit** — the device mints short-lived tickets from the backend.
4. Autostart in kiosk: a `systemd` user service or the desktop's autostart, launching `reflexion-mirror` on boot; kiosk mode is on by default.

## Security note

Keyless, same trust model as Android: the unit holds only a per-device bootstrap token → a rotating device credential → short-lived Qwen tickets. **No raw provider key is on the device or in the shipped app.** Remaining at-rest gap: on Electron the device credential lives in `localStorage` (plaintext on disk) rather than an OS keyring — route it through Electron main-process `safeStorage` to fully match Android's Keystore (listed below).

## Production hardening (follow-ups, not in the first cut)

- **Device-test the ticketed conversation** — the relay's orchestration was built for `qwen3-omni-flash`/`server_vad`; it now drives the ticket's `qwen3.5-omni`/`semantic_vad`. The plumbing is keyless and typechecks, but the live conversation (turn-taking, echo) needs validation on real Ubuntu speaker/mic hardware.
- **At-rest secrets:** move the bootstrap token + rotating credential from `localStorage` to Electron `safeStorage` (OS keyring) via the preload bridge.
- **Wake word** on Linux (a web/wasm wake-word, or a small native helper via the preload bridge) — today it's tap-to-start.
- **Auto-update** for the AppImage (electron-updater) and crash/telemetry reporting.
- **Device-test the network setup screen** — the `nmcli`/`bluetoothctl` wiring is unit-tested for parsing and error translation only; joining a real network, the hotspot, and Bluetooth PAN need a real Ubuntu unit (checklist in `network-setup.md`).
