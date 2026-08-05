# Mirror app — connecting a unit to the internet

A mirror is delivered to a home with **no network configured**. Until this existed the app could only
report the consequence — "unable to reach the Reflexion service" — and there was no way to fix it from the
device: the Linux appliance runs fullscreen kiosk with no desktop to escape to, and the Android unit sits
behind a locked-down launcher. This doc covers the setup surface and how to debug a unit that will not
connect.

## The primary path: phone setup over the mirror's own hotspot

**The Ubuntu unit has no keyboard and no mouse.** A technician can attach one over USB; a family cannot. So
the *default* recovery path assumes **no input on the mirror at all**:

1. The mirror boots straight into the app. If it is still offline after a grace period (45 s by default —
   long enough for NetworkManager to rejoin a known network on its own), the Electron shell brings up its
   **own Wi-Fi hotspot** and serves a small setup page on it.
2. The mirror's **screen** shows: the hotspot name, its passphrase, a `WIFI:` QR for one-tap joining, the
   portal address (+ QR), and a **6-digit code**.
3. The caregiver joins that hotspot from a phone and opens the page — most phones pop it automatically,
   because the portal answers the captive-portal probe paths (`/hotspot-detect.html`, `/generate_204`, …).
4. They pick the home Wi-Fi, type its password **on the phone's keyboard**, enter the 6-digit code, submit.
5. The mirror joins the network, and the app re-runs its boot checks and continues into pairing on its own.

Nothing is ever typed on the mirror. No app install, no App Store, identical on iOS and Android.

### Why not voice

The mirror's voice pipeline is **Qwen realtime — a cloud service**. Using voice to configure the network
that the cloud connection depends on is circular: with no internet there is no ASR, so there is no voice.
Network setup has to work fully offline, which rules voice out as the mechanism. Voice stays where it
belongs — the daily check-in, which needs internet anyway.

### Why the result appears on the mirror, not the phone

Joining a network takes the **same radio the hotspot runs on**, so the phone's connection to the mirror dies
the instant the switch begins. The portal therefore answers `202 Accepted` *before* applying, and the mirror's
screen reports the outcome. A failed join (a mistyped password is the common case) brings the hotspot and
portal straight back so the caregiver can retry — otherwise one typo would strand a unit with no hotspot, no
network, and no input device.

### Security of the portal

It binds to all interfaces, so it is deliberately minimal:

- serves **only** the page and `/api/networks`, `/api/status`, `/api/connect`. It does **not** serve the
  mirror SPA and does **not** expose the backend proxy — a LAN neighbour must never be able to borrow the
  device's credentials (pinned by test).
- runs **only** while setup mode is active; torn down as soon as the mirror has real internet, and on exit.
- applying a network requires the **6-digit code shown on the mirror's screen**, so being in radio range is
  not enough — you have to be able to see the device. Attempts are rate-limited, because six digits is
  otherwise brute-forceable in seconds.
- Wi-Fi passwords travel over plain HTTP inside the mirror's own **WPA2-encrypted** hotspot. A self-signed
  cert would trade that for a browser warning an installer has to click through anyway.

Environment knobs: `REFLEXION_SETUP_PORTAL_PORT` (default 8900), `REFLEXION_SETUP_GRACE_MS` (default 45000).

### Still open: a phone app that controls mirror settings over Bluetooth

The hotspot portal covers *network* setup, which is what blocks a unit from working at all. Full settings
control from a paired phone app over Bluetooth (BLE GATT on the mirror, `react-native-ble-plx` in
caregiver-app) is **not** built yet — it needs a new native dependency in the caregiver app and its own
pairing model. The portal was built first because it needs no app install and no new dependency anywhere.

## The touch screen (secondary path)

Used when the unit *does* have a touchscreen, or by an installer with a USB keyboard/mouse attached.
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
PHONE PATH (no input on the mirror)          TOUCH PATH (touchscreen / attached keyboard)
electron/setupPortal.js  ── page + 3 routes  app/network-setup.tsx        renderer UI
electron/main.js         ── setup-mode                └─ src/native/networkSetup.ts  facade + types
   state machine + connectivity watch                      └─ window.reflexionNetwork   (Linux only)
   │  auto-starts hotspot when offline                          └─ electron/preload.js  contextBridge
   └─ pushes state → PhoneSetupInstructions                          └─ electron/main.js registerNetworkIpc
        (mirror screen: SSID, passphrase,                                 └─ electron/network.js
         QR codes, portal URL, 6-digit PIN)                                    nmcli / bluetoothctl
```

Both paths bottom out in the same `electron/network.js`.

`electron/network.js` notes:

- Every command runs through `execFile` with an **argument array**, never a shell string — an SSID or
  Wi-Fi passphrase is free text supplied by whoever is setting the mirror up, and `"; rm -rf ~` is a legal
  Wi-Fi passphrase.
- `run()` **never logs an argument list**, only a short caller-supplied label plus stderr. A live home Wi-Fi
  passphrase is handed to nmcli as argv, so anything that prints argv can leak it into the log of an
  appliance sitting in someone's living room. (This replaced an allowlist of "flags whose next argument is
  secret", which fails open the moment a new secret-bearing flag is added and nobody updates the list.)
- The hotspot passphrase is held in memory by the process that generated it rather than read back with
  `nmcli -s`, so nmcli is never asked to print a live secret to stdout.
- Only named operations are exposed. There is deliberately **no** generic "run this command" channel.
- `friendlyError()` translates nmcli stderr into something an installer can act on (wrong password vs.
  polkit refusal vs. missing NetworkManager) instead of surfacing raw stderr.

Pure-logic tests: `npm run test:network` (terse-output parsing, error translation, hotspot passphrase
rules, and the API proxy). The command wiring itself needs real hardware — see the checklist below.

## On-device checklist (needs a real Ubuntu unit)

Not yet validated on hardware — this is the acceptance list:

0. **Phone path, no input attached** — unplug keyboard/mouse, boot with unconfigured Wi-Fi, wait out the
   grace period: the hotspot comes up, the mirror screen shows name/passphrase/QRs/PIN, a phone joins and
   opens the page (check it auto-opens), the home Wi-Fi is applied, and the mirror continues into pairing on
   its own. Then repeat with a **wrong** password and confirm the hotspot and portal come back for a retry.
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
