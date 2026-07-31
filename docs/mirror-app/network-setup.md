# Mirror app — connecting a unit to the internet

A mirror is delivered to a home with **no network configured**. Until this existed the app could only
report the consequence — "unable to reach the Reflexion service" — and there was no way to fix it from the
device: the Linux appliance runs fullscreen kiosk with no desktop to escape to, and the Android unit sits
behind a locked-down launcher. This doc covers the setup surface and how to debug a unit that will not
connect.

## The screen

`mirror-app/app/network-setup.tsx` — reachable three ways:

- **Boot screen**, when an unpaired mirror cannot reach the backend → "Set up the connection" (primary
  action on `NoConnectionScreen`).
- **Offline screen**, when a *paired* mirror is offline → a quiet "Set up the internet connection" link
  (the check-in still works offline and queues, so this must not shout).
- **Settings** (`app/settings.tsx`) → "Set up the internet connection".

It is an **installer/caregiver** surface, not an elder one: denser than the check-in screens, and it names
"Wi-Fi" and "Bluetooth" plainly. It never blocks a check-in.

Three tabs, because homes differ:

| Tab | What it does | Gives the mirror internet? |
|---|---|---|
| **Wi-Fi** | Scan, join (with an on-screen keyboard), rejoin a saved network, forget one | **Yes** — home router or a phone's personal hotspot |
| **Bluetooth** | Power/discoverable, scan, pair a phone, then "Use for internet" (PAN/`panu`) | **Yes** — when there is no usable Wi-Fi |
| **Mirror hotspot** | The mirror broadcasts its *own* AP, showing SSID + generated passphrase | **No** — it is a way to reach the unit |

The hotspot tab warns *before* the tap that starting an AP takes over the Wi-Fi radio and will drop a
Wi-Fi-connected mirror offline — an installer who is not warned reads that as a new fault.

## Platform capabilities

| | Linux (Electron) | Android |
|---|---|---|
| Join a Wi-Fi network from the app | **yes** (`nmcli`) | no — OS-privileged since Android 10 |
| Start a hotspot from the app | **yes** (`nmcli device wifi hotspot`) | no |
| Pair Bluetooth / tether from the app | **yes** (`bluetoothctl` + `nmcli bt-type panu`) | no |
| Fallback | — | opens the OS settings panel (`Linking.sendIntent`) |

Android cannot let an app join a network or enable tethering on its behalf, so the honest offer there is to
open the right settings panel. `getNetworkCapabilities()` returns `settingsOnly: true` and the screen
renders the three deep links instead of live controls.

## How it is wired

```
app/network-setup.tsx            renderer UI (Expo Router screen)
  └─ src/native/networkSetup.ts  cross-platform facade + types; Android → OS settings intents
       └─ window.reflexionNetwork      (Linux only)
            └─ electron/preload.js     contextBridge, one fixed IPC channel per operation
                 └─ electron/main.js   ipcMain.handle registry (registerNetworkIpc)
                      └─ electron/network.js   nmcli / bluetoothctl
```

`electron/network.js` notes:

- Every command runs through `execFile` with an **argument array**, never a shell string — an SSID or
  Wi-Fi password is free text typed on a touch keyboard, and `"; rm -rf ~` is a legal Wi-Fi password.
- Wi-Fi passwords are redacted before anything is logged (`logArgs`).
- Only named operations are exposed. There is deliberately **no** generic "run this command" channel.
- `friendlyError()` translates nmcli stderr into something an installer can act on (wrong password vs.
  polkit refusal vs. missing NetworkManager) instead of surfacing raw stderr.

Pure-logic tests: `npm run test:network` (terse-output parsing, error translation, hotspot passphrase
rules, and the API proxy). The command wiring itself needs real hardware — see the checklist below.

## On-device checklist (needs a real Ubuntu unit)

Not yet validated on hardware — this is the acceptance list:

1. `nmcli` present and the kiosk user may control networking (polkit). If not, the screen reports "not
   allowed to change network settings" rather than failing silently.
2. Wi-Fi: scan lists networks; joining a WPA2 network with a correct password connects; a wrong password
   reports the password message and does **not** leave a broken saved profile behind.
3. Phone hotspot: a personal hotspot appears in the list and joins like any other network.
4. Bluetooth: the mirror appears on a phone; pair succeeds (accept the prompt on the phone); with
   Bluetooth tethering switched on on the phone, "Use for internet" brings up the `panu` link and
   `connectivity` becomes `full`.
5. Mirror hotspot: starts, SSID + passphrase display, a laptop can join, stopping it restores the previous
   Wi-Fi connection.
6. After any of the above, "Check again" on the boot screen proceeds to pairing without an app restart.

## Debugging "the mirror cannot connect to the server"

Work down this list — the first two are the ones that have actually bitten.

1. **CORS blocking the renderer (fixed, but check the build).** The Electron renderer is a page on
   `http://127.0.0.1:8899`. Before the proxy existed it called the backend cross-origin, which preflights
   because of `Authorization` / `Idempotency-Key` / `X-Device-Bootstrap`. Production's
   `CORS_ALLOWED_ORIGINS` named only the admin SPA, so **every request was blocked in the browser and
   never reached the server** — nothing appeared in the server logs, and the mirror said "unable to reach
   the Reflexion service" against a perfectly healthy API. Two independent fixes now cover this:
   - the Electron shell proxies `/api` and `/health` from its own origin, so backend calls are
     same-origin and no allowlist is consulted (`electron/apiProxy.js`);
   - the server always allows the mirror's loopback origins, so **AppImages built before the proxy still
     work** (`reflexion-server/src/app.ts`, `MIRROR_LOOPBACK_ORIGINS`).

   Confirm with: `curl -i -X OPTIONS <api>/api/v1/device-pairings -H 'Origin: http://127.0.0.1:8899'
   -H 'Access-Control-Request-Method: POST' -H 'Access-Control-Request-Headers: x-device-bootstrap'` —
   an `access-control-allow-origin` header must come back. Its **absence** is the bug.

2. **Wrong or missing API origin.** Older builds required `EXPO_PUBLIC_API_BASE` at export time and fell
   closed to the unreachable `http://127.0.0.1:9` without it. The Electron build no longer needs it (see
   below). Check the Electron log line at startup:
   `[electron] backend proxied at http://127.0.0.1:8899/api -> https://…`

3. **No network at all** — the case this whole doc is about. Open the setup screen and connect the unit.

4. **A network with no route out.** `connectivity: portal` (captive portal needing a browser sign-in) or
   `limited` looks identical to a dead server from the outside. The setup screen's status pill now names
   both, and the boot screen distinguishes "not connected" from "on a network but cannot reach Reflexion".

## API origin is now runtime config

The Electron build resolves the backend origin at **runtime**, so a unit can be re-pointed without
re-exporting the web bundle. Precedence:

1. `REFLEXION_API_BASE` (or `EXPO_PUBLIC_API_BASE`) in the launch environment;
2. `apiBase` in `<userData>/device-config.json` (`~/.config/Reflexion Mirror/device-config.json`);
3. the production default.

The renderer detects the shell via `window.reflexionMirror.apiProxy` and uses **relative** URLs, so a
baked `EXPO_PUBLIC_API_BASE` no longer forces cross-origin calls (`src/config/apiUrl.ts`).
