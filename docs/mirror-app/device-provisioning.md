# Pairing and provisioning a Reflexion mirror

The normal setup is a first-boot flow. A caregiver does not need to handle a device credential or configure a mirror name.

## Normal first-boot flow

1. Build the Mirror and Caregiver apps with the same `EXPO_PUBLIC_API_BASE` (staging and production must never be mixed).
2. Connect the mirror to Wi-Fi. On first use it generates a stable `dev_...` local device ID and a random install secret, then stores both in platform secure storage.
3. The mirror calls `POST /api/v1/device-pairings` with `X-Device-Id`, `X-Device-Install-Secret`, and a persisted `Idempotency-Key`. The server creates the unclaimed device record if needed and returns a ten-minute pairing session containing a six-digit display code and a separate short-lived QR token.
4. The mirror shows the code and a QR payload containing only the pairing token. It polls `GET /api/v1/device-pairings/{pairingId}`.
5. In the Caregiver app, scan the QR or enter the six-digit code, then choose the loved one. The app calls `POST /api/v1/device-pairing-claims` with the caregiver bearer token and the selected patient.
6. The server transaction checks caregiver permission, device ownership and one-active-assignment rules, creates the assignment and permanent device credential together, invalidates the pairing session, and gives the mirror a short-lived exchange ticket.
7. The mirror exchanges the ticket at `POST /api/v1/device-credentials/exchange`, stores the access and refresh credentials in SecureStore, fetches the loved one’s configuration, and starts the conversation automatically.

The QR never contains the permanent access token, refresh credential, install secret, or database identifier beyond the opaque pairing token. The caregiver sees only a pairing action, code/scan input, and loved-one selection.

## Identity and storage

- The local device ID is created once per app installation. The install secret is never displayed, logged, or sent in a QR code.
- The server stores the install-secret hash, not the secret. Pairing tokens and exchange tickets are stored as hashes/ciphertext as appropriate, and the permanent refresh credential is stored hashed.
- Access and refresh credentials are stored in Android/iOS SecureStore. Non-secret metadata such as the active patient and expiry timestamps remains in app storage.
- A normal update install preserves app data, so the mirror keeps its identity and credential. A credential exchange response can be retried with the same idempotency key if the network drops after delivery.
- Clearing SecureStore while leaving the old local ID behind fails closed into recovery. A full app-data wipe or factory reset can create a new local identity; support must then unlink/reset the old server assignment before pairing that physical mirror again.

## Compatibility path

Existing units may still use the legacy per-device `X-Device-Bootstrap` token from `provision:device`. It is accepted by the pairing, status, and exchange endpoints, but it is no longer required for a fresh mirror. Never embed one bootstrap token in a shared APK: it is bound to one device record.

For an existing factory-provisioned unit:

```bash
cd reflexion-server
npm run provision:device -- --serial=mirror-test-01 --hardware=v1 --software=1.0.0
```

Keep the returned token secret. The installer screen remains available only for the explicit installer setup path; it is not part of the ordinary elder-facing flow.

## User-visible recovery states

- Wi-Fi unavailable: “Connect your mirror to Wi-Fi to continue.”
- Wi-Fi/internet available but backend unreachable: retry the Reflexion service; this is not reported as a generic unprovisioned-device error.
- Backend reachable but credential rejected: show the controlled recovery message and contact support; do not silently register a replacement device.
- Pairing code expired or cancelled: the mirror generates a fresh code.
- Caregiver has no permission or the loved one already has a mirror: the caregiver app explains the conflict and leaves the existing assignment unchanged.
- Mirror power/network drops during pairing: the pending pairing is retained until expiry; restart resumes the same session when possible.
- Exchange response is lost: the mirror retains the exchange ticket and idempotency key and retries; the server returns the same credential response instead of issuing a second credential.

## Diagnostics and security controls

The mirror records structured, non-sensitive stages: `wifi_connected`, `internet_reachable`, `backend_reachable`, `authenticated`, `paired`, `assigned`, `pairing_session_created`, `pairing_code_displayed`, `credential_saved`, `ready`, and `pairing_failed` with a safe reason code. The backend audit records claim initiation/completion and the atomic credential-issued milestone. After authentication, the next heartbeat sends the bounded diagnostic queue to device telemetry; tokens, codes, install secrets, and credentials are excluded and the server allowlists the diagnostic fields.

Pairing codes and QR tokens expire after ten minutes, are single-use, and are rate-limited. Claiming requires an authenticated caregiver with `device:assign` access to the selected loved one. An active device or patient assignment cannot be silently replaced. Credential exchange tickets expire after five minutes, and exchange/claim/pairing mutations use idempotency keys.

## Environment check

The mirror and caregiver clients each read their own `EXPO_PUBLIC_API_BASE`; there is no safe cross-environment pairing. Release builds must inject the same staging or production URL into both clients, and deployment smoke tests should verify `/health` plus a pairing round trip before distributing an APK.
