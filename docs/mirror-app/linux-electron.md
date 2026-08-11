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
| Device identity | baked per-APK, or entered on screen | **runtime only** — `device-config.json`, nothing baked (see "Per-device provisioning") |
| OTA | EAS Update (expo-updates) | **shell-driven** — `expo-updates` has no web build (see "OTA updates") |
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
- `electron/preload.js` — context-isolated bridge: `reflexionMirror` (platform + `apiProxy` flag), `reflexionNetwork` (one fixed IPC channel per network operation), `reflexionProvisioning` (the runtime bootstrap token) and `reflexionUpdates` (both OTA channels).
- `electron/deviceConfig.js` — runtime identity/config from `<userData>/device-config.json`: backend origin, bootstrap token, update host. Why nothing device-bound may be compiled in.
- `electron/bundleUpdates.js` / `electron/shellUpdates.js` — the two OTA channels (see "OTA updates").
- `scripts/publish-linux-update.mjs` — builds the update payloads + manifests (`npm run electron:publish-update`).
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
| `REFLEXION_BOOTSTRAP_TOKEN` | **runtime** | per-device pairing token; overrides `device-config.json` (see "Per-device provisioning") |
| `REFLEXION_UPDATE_BASE` | runtime | OTA manifest host; defaults to `<backend origin>/mirror-updates` |
| ~~`EXPO_PUBLIC_DEVICE_BOOTSTRAP_TOKEN`~~ | — | **do not use for Linux** — baking an identity into the AppImage makes every unit claim the same device |
| ~~`QWEN_API_KEY`~~ | — | **not used** — the device is keyless; the relay authenticates with the renderer's ticket |
| `REFLEXION_MIRROR_KIOSK` | runtime | `0` = windowed (not fullscreen kiosk) |
| `REFLEXION_MIRROR_SKIP_RELAY` | runtime | `1` = don't spawn the local relay (point at an external one) |
| `REFLEXION_MIRROR_DEVTOOLS` | runtime | `1` = open Chromium devtools |
| `REFLEXION_MIRROR_WEB_PORT` / `REFLEXION_RELAY_PORT` | runtime | local ports (`8899` / `8787`) |

**Backend CORS:** no longer a deployment step — backend calls go through the local proxy and are same-origin, and the server allows the mirror's loopback origins regardless. See "Backend proxy" above for the failure this replaced.

## Per-device provisioning (the AppImage carries no identity)

**One AppImage serves the whole fleet.** A bootstrap token is *device-bound* — the backend re-reads `devices` by `(did, serialHash)` on every call — so baking one into the installer makes every unit built from it claim the same device and knock each other's pairing over. The Linux build therefore ships **no identity at all**: both the backend origin and the token are resolved at runtime by `electron/deviceConfig.js`.

Provision on the **production** server (a token minted anywhere else 401s — different database, different `JWT_SECRET`):

```bash
cd /www/wwwroot/reflexion/reflexion-server
npm run provision:device -- --serial=<the unit's hardware serial> --hardware=ubuntu-v1
```

Then drop the token on the unit — `<userData>` is `~/.config/Reflexion Mirror/`, printed at startup:

```json
// ~/.config/Reflexion Mirror/device-config.json
{
  "apiBase": "https://reflexion.production.tktonny.top",
  "bootstrapToken": "eyJhbGciOiJIUzI1NiI..."
}
```

Restarting the app is enough; nothing is rebuilt. `REFLEXION_BOOTSTRAP_TOKEN` in the launch environment overrides the file, which is the convenient path for a bench test.

A malformed token is rejected **at startup** with a log line naming its source, rather than surfacing later as an opaque 401 during pairing — the check is a JWT shape check only (the shell cannot verify a signature). An unprovisioned unit is a **normal state**: network setup, the hardware self-check and the update channels all work without a token; only pairing needs one.

The on-screen entry field (`app/test-device.tsx`) still works and is the right path when someone has a keyboard handy. The file exists because the appliance usually does not, and the token is a ~300-character JWT.

## OTA updates

**Two channels, both manual.** The mirror is a kiosk that stays powered for days, so "check on launch" would almost never fire — and polling would risk reloading **mid-conversation**, cutting an elder off in the middle of a check-in. An operator triggers updates from the settings screen when they can see the mirror is idle, and download/apply are always separate steps.

> **`expo-updates` does not work here.** It has no web implementation, so in Electron `Updates.isEnabled` is always `false` — before this, every Ubuntu unit silently reported "OTA disabled" and had no update path at all. The Android fleet keeps using EAS Update; Linux uses the shell equivalents below, and `src/lib/otaUpdates.ts` picks the right one so one settings screen drives both.

| | Renderer bundle | Electron shell |
|---|---|---|
| Module | `electron/bundleUpdates.js` | `electron/shellUpdates.js` |
| Manifest | `latest.json` | `shell-latest.json` |
| Size | ~5 MB | ~97 MB |
| Covers | JS, assets, screens, prompts, conversation flow, every inlined `EXPO_PUBLIC_*` | Chromium, main process, `network.js`, the relay |
| Applies by | reloading the window | replacing the AppImage + relaunch |
| Frequency | routine | rare |

Bundles install to `<userData>/bundles/<version>/` and the packaged copy inside the AppImage always remains as the fallback. Both channels verify a **sha256 from the manifest** before installing anything, and refuse an archive served off the update origin so a manifest cannot redirect a unit at an arbitrary host. That protects against corruption, truncation and redirection — it is **not** a signature, so only point `REFLEXION_UPDATE_BASE` at a host you control as tightly as the backend.

**Rollback is automatic.** After applying, the shell records a boot marker that only the new bundle can clear (`initOtaUpdates()` in `app/_layout.tsx` calls `markBooted`). If the unit restarts and that marker is still set, the bundle never rendered — it is rolled back and recorded so the same broken version is not reinstalled. Without that, a bad update on an appliance in someone's home is a site visit.

### Publishing

```bash
cd mirror-app
npm run electron:export                                  # → dist/
npm run electron:publish-update -- --version=2026.08.11-1
# large/rare: also publish the shell
npm run electron:publish-update -- --shell-only \
  --shell="dist-linux/Reflexion Mirror-1.0.0.AppImage" --arch=x64

rsync -av dist-updates/ root@<server>:/www/wwwroot/mirror-updates/
```

Serve it as a plain static path — **no backend change**:

```nginx
# in the reflexion.production.tktonny.top server block
location /mirror-updates/ {
    alias /www/wwwroot/mirror-updates/;
    autoindex off;
    # latest.json must never be cached, or units keep seeing the old version
    location ~ \.json$ { add_header Cache-Control "no-store"; }
}
```

An x64 AppImage installed on an arm64 unit would leave a machine that cannot execute, so the shell refuses a manifest whose `arch` does not match. A mixed fleet needs `shell-latest.json` mapped per architecture (the publish script writes `shell-latest-<arch>.json` alongside it).

Tests: `npm run test:network` (69 unit + 5 end-to-end against a real local update host, covering bad digest, missing `index.html`, off-origin manifest, and the never-boots rollback).

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
- **Sign the update manifests.** Both OTA channels trust HTTPS + a sha256 from the manifest, so a compromised update host could serve a bundle the shell would accept. Signing the manifest with a key pinned in the shell closes that; it matters more for the shell channel, which replaces an executable.
- **Crash/telemetry reporting** — OTA landed, but a unit that rolls back still only says so in its local log.
- **Device-test the network setup screen** — the `nmcli`/`bluetoothctl` wiring is unit-tested for parsing and error translation only; joining a real network, the hotspot, and Bluetooth PAN need a real Ubuntu unit (checklist in `network-setup.md`).
