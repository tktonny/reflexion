# Provisioning a mirror (device code / bootstrap credential)

How a mirror gets its identity, how to run a test fleet, and the two traps that cost real debugging time.

## The credential chain

```
provision:device  →  bootstrap token (30-day JWT, per device)
                     ↓ entered on the device
                   pairing (6-digit code claimed by the caregiver app)
                     ↓
                   rotating device credential  →  short-lived Qwen tickets per conversation
```

The device never holds a provider key. That is the whole point of the chain — see `docs/ARCHITECTURE-AND-API.md`.

## Trap 1: the bootstrap token is bound to ONE device

Its payload carries a `did` (device id) claim. So if you bake one token into an APK and install that APK on several units, **they all claim the same device record** — the later install silently takes over the earlier one's identity, and both devices' conversations land on the same patient.

**Therefore: do not embed the token in a shared build.** Leave `EXPO_PUBLIC_DEVICE_BOOTSTRAP_TOKEN` unset and enrol each unit on the device. One universal APK then serves every unit.

The startup self-check has an `identity` item that fails loudly if the embedded token's device id ever disagrees with the paired credential's.

## Trap 2: reinstalling an APK does NOT lose the credential

`adb install -r` (same signing key, update-install) **preserves** app data, so SecureStore/AsyncStorage keep the credential. Frequent test APKs therefore need **no re-pairing**.

Credentials are only lost by: `adb uninstall`, clearing app data, or changing the signing key.

## Test-fleet recipe

**1. Mint one token per unit** — different `--serial` each time:

```bash
cd reflexion-server
for i in 01 02 03; do
  npm run provision:device -- --serial=mirror-test-$i --hardware=v1 --software=1.0.0
done
```

Output per unit: `{ "deviceId": …, "bootstrapToken": …, "expiresInDays": 30 }`. The token is a secret — do not commit or paste it into chat.

`--serial` is hashed and upserted, so:

| | effect |
|---|---|
| **different** serial | a new device record → an independent identity |
| **same** serial | the *same* device, re-issued token → this is how you **renew**, not how you add a unit |

Prefer the unit's real hardware serial: it makes "which physical mirror is this?" answerable, and re-running the same serial later renews rather than creating a duplicate.

**2. Build a universal APK** — the enrolment screen is gated, and a release build has `__DEV__ === false`, so it needs:

```
EXPO_PUBLIC_ENABLE_INSTALLER_SETUP=true
# EXPO_PUBLIC_DEVICE_BOOTSTRAP_TOKEN=   ← leave unset
```

**3. Install** — `adb install -r <apk>`

**4. Enrol on the device** — from the pairing screen, open the device-test screen and either paste the token or scan a QR of it. The QR payload may be the raw token or `{"bootstrapToken":"…"}`. Expired/invalid tokens are reported explicitly.

**5. Pair** — the mirror shows a 6-digit code + QR; the caregiver app claims it; the mirror polls, receives the rotating credential, and enters the conversation.

Steps 1–4 are once per unit. After that, flash as many APKs as you like.

## Expiry

The bootstrap token lasts **30 days** (hard-coded in `reflexion-server/src/scripts/provisionDevice.ts`). To renew, re-run `provision:device` with the **same** `--serial`; the existing pairing is unaffected.

The `identity` self-check warns at ≤7 days remaining, so a fleet no longer discovers this by suddenly being unable to pair.

## ⚠️ Open issue: the 30-day TTL does not survive a factory flow

If the token is written at the factory, the clock starts on the production line. Factory → warehouse → shipping → retail → the customer opening the box **routinely exceeds 30 days**, and the elder would then be unable to pair a brand-new mirror, with no on-site recovery.

The self-check's expiry warning cannot help here: a boxed mirror is not powered on to show it.

Options, in preference order:
1. **Give factory tokens a long TTL** — add a `--ttl-days` flag and use ≥365 on the line (tests keep 30). Small change; the token is pair-only (`scopes: ['device:pair']`) and is exchanged for a rotating credential immediately, so a long-lived pair-only token is an acceptable risk.
2. **Mint on first boot** from the hardware serial — cleanest, but needs a new backend endpoint plus serial registration on the line.
3. **Keep the on-site enrolment path as the fallback** — already works today; costs a technician visit.

Also unresolved: **how the line writes the token**. Rebuilding the APK per unit is the wrong answer (a build per device, and it re-introduces Trap 1). Writing it to device-local storage for the app to read and `persistBootstrapCredential()` once on first boot is the better shape — the same approach the Linux build doc proposes.
